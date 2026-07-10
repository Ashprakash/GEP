"""
method_tool.py — GEP v2: grounded, tool-verified, self-calibrated decisions.

Fixes the bottleneck the pilot exposed: small models can't do the arithmetic even
when the numbers are present (numeric_accuracy ~0.2 WITH gold evidence). Instead of
compressing evidence into prose, we:

    1. EXTRACT typed variables + a formula from raw evidence (LLM, no gold answer)
    2. COMPUTE the answer with a deterministic executor (tool / program-of-thought)
    3. CONFIDENCE = grounding-completeness x execution-success (a real signal, unlike
       token-overlap priors) -> used for SELECTIVE prediction (answer / abstain)

The pure-python core (execute_program, grounding_confidence, accuracy_at_coverage)
has no heavy deps and is unit-testable. run_tool_cascade wires it to a HF extractor.
"""

from __future__ import annotations

import json
import os
import re

_ALLOWED = re.compile(r"^[0-9a-zA-Z_+\-*/().,%\s]+$")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SAFE = {"max": max, "min": min, "abs": abs, "round": round, "sum": sum,
         "Max": max, "Min": min, "Abs": abs}
_STOP = {"total", "net", "the", "of", "fy", "in", "and", "current",
         "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"}


def _strip_assignment(formula):
    """'gross_margin = expr' -> 'expr' (leave ==, <=, >= alone)."""
    m = re.match(r"\s*[A-Za-z_][A-Za-z0-9_]*\s*=(?!=)\s*(.+)$", formula)
    return m.group(1).strip() if m else formula


def _norm_tokens(name):
    return {t for t in name.lower().split("_") if t and t not in _STOP}


def _best_var(name, variables):
    """Fuzzy-match a formula identifier to the closest extracted variable name."""
    nt = _norm_tokens(name)
    if not nt:
        return None
    best, score = None, 0.0
    for v in variables:
        vt = _norm_tokens(v)
        if not vt:
            continue
        j = len(nt & vt) / len(nt | vt)
        if j > score:
            best, score = v, j
    return best if score >= 0.5 else None


def execute_program(variables: dict, formula: str):
    """Compute `formula` over `variables` deterministically. Returns float or None.

    Robustness for FAIR cross-model measurement (not answer inflation): strips a
    leading 'name =' assignment, allows max/min/abs, and reconciles formula
    identifiers to the closest extracted variable (e.g. total_current_assets ->
    current_assets) when unambiguous.
    """
    if not formula:
        return None
    formula = _strip_assignment(formula)
    if not _ALLOWED.match(formula):
        return None
    env = {}
    for k, v in variables.items():
        try:
            env[k] = float(v)
        except (TypeError, ValueError):
            return None
    for nm in set(_IDENT.findall(formula)):
        if nm in env or nm in _SAFE:
            continue
        match = _best_var(nm, variables)
        if match is not None:
            env[nm] = env[match]
    try:
        return float(eval(formula, {"__builtins__": {}, **_SAFE}, env))  # noqa: S307
    except Exception:
        return None


def _num_str(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return str(int(f)) if f == int(f) else str(f)


def grounding_confidence(variable_values, evidence_text, executed: bool) -> float:
    """Confidence = fraction of input values found in the evidence, gated on execution.

    A program that (a) executed and (b) drew every input from the evidence is usually
    right; one that hallucinated inputs or failed to compute is usually wrong. That
    correlation is what makes selective prediction work (unlike token-overlap priors).
    """
    if not executed or not variable_values:
        return 0.0
    ev = (evidence_text or "").replace(",", "")
    found = sum(1 for v in variable_values if (_num_str(v) or "\0") in ev)
    return round(found / len(variable_values), 3)


def accuracy_at_coverage(scored, coverages=(0.3, 0.5, 0.7, 1.0)):
    """scored: list of (confidence, correct_bool). Answer highest-confidence first."""
    order = sorted(scored, key=lambda x: x[0], reverse=True)
    n = len(order)
    if n == 0:
        return {f"acc@{int(c*100)}": None for c in coverages}
    cum, curve = 0.0, []
    for k, (_c, correct) in enumerate(order, start=1):
        cum += 1.0 if correct else 0.0
        curve.append(cum / k)
    out = {}
    for c in coverages:
        k = max(1, min(n, round(c * n)))
        out[f"acc@{int(c*100)}"] = round(curve[k - 1], 3)
    out["aurc"] = round(sum(1 - a for a in curve) / n, 3)
    out["full_accuracy"] = round(curve[-1], 3)
    return out


# --- LLM-facing prompt + parsing (extract program, do NOT answer) ---

def make_program_prompt(row, max_evidence_chars=6000):
    ev = row["evidence_text"]
    if max_evidence_chars and len(ev) > max_evidence_chars:
        ev = ev[:max_evidence_chars] + "\n\n[TRUNCATED]"
    return f"""You are a financial analyst. From the raw evidence, extract ONLY the
variables and the formula needed to answer the question. Do NOT compute the final answer.
Output ONLY one JSON object and nothing else.

Example question: What was the operating margin for FY2022?
Example evidence: Operating income was $9,570 million. Total revenue was $52,862 million.
Example output:
{{"variables": {{"operating_income": 9570, "total_revenue": 52862}}, "formula": "operating_income / total_revenue * 100", "unit": "percent"}}

Rules:
- "variables" is a JSON object mapping short snake_case names to the NUMBERS you pulled from the evidence.
- "formula" is an arithmetic expression that uses ONLY those variable names (or "" if the answer is a direct lookup — then put the looked-up value as a single variable and set formula to that variable name).
- Every name in "formula" must appear in "variables".
- If the evidence is insufficient, output {{"variables": {{}}, "formula": "", "unit": "none"}}.

Question:
{row["question"]}

Raw evidence:
{ev}

JSON:"""


def _first_json_object(text):
    """Return the first balanced {...} substring (robust to trailing garbage)."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _coerce_number(v):
    """Value -> float, evaluating simple arithmetic strings like '37857 - 23977'."""
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    if isinstance(v, str):
        return execute_program({}, v)
    return None


def parse_program(prediction):
    """Extractor output -> (variables dict, formula str, unit str). Robust to prose
    around the JSON, trailing garbage, list-form variables, and expression values."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(prediction).strip(), flags=re.I)
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        block = _first_json_object(text)
        if block:
            try:
                obj = json.loads(block)
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return {}, "", "none"

    raw_vars = obj.get("variables", {})
    if isinstance(raw_vars, dict):
        pairs = list(raw_vars.items())
    elif isinstance(raw_vars, list):  # [{"name":..,"value":..}, ...]
        pairs = [(e.get("name"), e.get("value")) for e in raw_vars
                 if isinstance(e, dict) and "name" in e and "value" in e]
    else:
        pairs = []

    clean = {}
    for k, v in pairs:
        if k is None:
            continue
        fv = _coerce_number(v)
        if fv is not None:
            clean[str(k)] = fv
    # models sometimes rename the key ("formulas"/"formual") — accept common variants
    formula = obj.get("formula")
    if formula is None:
        for alt in ("formulas", "formual", "formulae", "expression"):
            if alt in obj:
                formula = obj[alt]
                break
    formula = str(formula or "").strip()
    unit = str(obj.get("unit", "none") or "none").strip()
    return clean, formula, unit


_NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def units_tolerant_correct(val, gold_answer):
    """True if the computed value matches a gold number under any 10^3 scaling
    (so 400 == $400,000,000 and 13,200 million == 13.2 billion)."""
    if val is None:
        return False
    golds = [float(x.replace(",", "")) for x in _NUM.findall(str(gold_answer))]
    for g in golds:
        for scale in (1, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9):
            if abs(val * scale - g) <= max(0.05, 0.02 * abs(g)):
                return True
    return False



def compute_answer(variables, formula):
    """Compute the answer. Lookup path: a single variable with no formula IS the
    answer (fixes the case where the model correctly extracts one value but emits
    no formula). Otherwise evaluate the formula."""
    if formula:
        return execute_program(variables, formula)
    if len(variables) == 1:
        return float(next(iter(variables.values())))
    return None


def format_value(val, unit):
    if val is None:
        return None
    if unit == "percent":
        return f"{val:.1f}%"
    if unit == "currency":
        return f"${val:,.0f}"
    return f"{val:g}"


def _answer_logprob(gen, prompt, generated_text):
    """Parametric-confidence baseline: exp(mean token log-prob) of the model's own
    generated program given the prompt, via one forward pass. In (0,1], higher =
    more confident. This is the 'log-probability' signal DRC is compared against."""
    if not generated_text:
        return None
    try:
        import torch
        model, tok = gen.model, gen.tokenizer
        ids = tok(prompt + str(generated_text), return_tensors="pt").input_ids.to(model.device)
        plen = tok(prompt, return_tensors="pt").input_ids.shape[1]
        with torch.no_grad():
            logits = model(ids).logits
        logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        tgt = ids[0, 1:]
        tok_lp = logp[torch.arange(tgt.shape[0]), tgt]
        ans_lp = tok_lp[max(0, plen - 1):]
        if ans_lp.numel() == 0:
            return None
        return float(torch.exp(ans_lp.mean()).item())
    except Exception:
        return None


def make_verify_prompt(question, computed_answer, evidence_text, max_evidence_chars=4000):
    """Self-verification: rate confidence the computed answer is right (no gold answer)."""
    ev = evidence_text or ""
    if max_evidence_chars and len(ev) > max_evidence_chars:
        ev = ev[:max_evidence_chars] + "\n\n[TRUNCATED]"
    return f"""You are checking a computed financial answer against the evidence.

Question:
{question}

Evidence:
{ev}

Proposed answer: {computed_answer}

How confident are you the proposed answer is correct and supported by the evidence?
Return strict JSON: {{"confidence": <number 0 to 1>}}"""


def run_tool_cascade(df, extractor_model_id="Qwen/Qwen2.5-7B-Instruct",
                     n_examples=50, random_state=7, max_evidence_chars=6000,
                     max_new_tokens=192, verbalized_confidence=True, checkpoint_path=None):
    """Real run: extractor -> executor -> (grounding + verbalized) confidence.

    Captures BOTH confidence signals per row so selective accuracy can be reported
    by each (and their product) without re-running the expensive 7B pass. Reuses the
    pilot's generator loader (so run_pilot's MPS monkeypatch applies) and its scorer.

    If checkpoint_path is given, each completed example is appended to that JSONL and
    reloaded on restart — so a crash mid-run resumes instead of recomputing.
    """
    import pandas as pd
    from benchmark import financebench_pilot as pilot

    base = df[df["has_numeric_answer"]].sample(
        min(n_examples, int(df["has_numeric_answer"].sum())), random_state=random_state
    ).reset_index(drop=True)

    # Resume: load already-completed rows keyed by financebench_id.
    done = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["financebench_id"]] = r
                except Exception:
                    pass
        if done:
            print(f"[resume] {len(done)} examples loaded from {checkpoint_path}")

    ck = open(checkpoint_path, "a") if checkpoint_path else None
    gen = None  # lazy: don't load the model if everything is already checkpointed
    rows = []
    for _, row in base.iterrows():
        fid = row["financebench_id"]
        if fid in done:
            rows.append(done[fid])
            continue
        if gen is None:
            gen = pilot.load_hf_generator(model_id=extractor_model_id, max_new_tokens=max_new_tokens)

        prompt = make_program_prompt(row, max_evidence_chars)
        raw = pilot.call_hf_generator(gen, prompt)
        variables, formula, unit = parse_program(raw)
        val = compute_answer(variables, formula)
        answer = format_value(val, unit)
        grounding = grounding_confidence(list(variables.values()), row["evidence_text"], val is not None)
        # parametric-confidence baseline: model's own log-prob of the program it produced
        logprob_conf = _answer_logprob(gen, prompt, raw) if val is not None else None

        if not variables and not formula:
            fail = "no_program_parsed"
        elif val is None:
            fail = "no_formula_multivar" if not formula else "formula_did_not_execute"
        else:
            fail = ""

        verbalized = None
        if verbalized_confidence and answer is not None:
            vraw = pilot.call_hf_generator(
                gen, make_verify_prompt(row["question"], answer, row["evidence_text"], max_evidence_chars))
            verbalized = pilot.parse_model_confidence(vraw)

        pred = json.dumps({
            "answer": answer if answer is not None else "INSUFFICIENT_EVIDENCE",
            "confidence": grounding,
        })
        scores = pilot.score_prediction(pred, row["answer"])
        row_dict = {
            "extractor_id": extractor_model_id,
            "condition": "tool_program",
            "task_type": pilot.infer_task_type(row),
            "financebench_id": fid,
            "question": row["question"],
            "gold_answer": row["answer"],
            "formula": formula,
            "n_vars": len(variables),
            "computed": answer,
            "executed": val is not None,
            "match_units": units_tolerant_correct(val, row["answer"]),
            "fail_reason": fail,
            "extractor_raw": str(raw)[:4000],  # keep enough to re-score offline later
            "support_probability": grounding,
            "verbalized_confidence": verbalized,
            "logprob_confidence": logprob_conf,
            **scores,
        }
        rows.append(row_dict)
        if ck:
            ck.write(json.dumps(row_dict, default=str) + "\n")
            ck.flush()
    if ck:
        ck.close()
    return pd.DataFrame(rows)
