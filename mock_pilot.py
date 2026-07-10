#!/usr/bin/env python3
"""
mock_pilot.py — offline mimic of the GEP pilot. No HF, no network, no deps.

Why: the real pilot (run_pilot.py) needs torch/transformers + a HuggingFace model
download. This script mimics the SAME pipeline on this Mac with pure stdlib:

    synthetic FinanceBench-like sample
        -> prompt per condition
        -> MockModel (stubbed 'small model' behavior)
        -> REAL scoring (functions copied verbatim from financebench_pilot.py)
        -> summary tables + CSVs

The MockModel is tuned to reproduce the qualitative findings recorded in
benchmark/results_log.md (gold evidence doesn't rescue a tiny model and worsens
calibration; the model can copy short/direct/counterfactual evidence but fails
raw evidence; it mostly abstains on missing evidence). Swap MockModel for a real
HF generator and the rest of the pipeline is identical.

    python mock_pilot.py                 # baseline + probe, writes results/
    python mock_pilot.py --stages baseline
"""

import argparse
import csv
import json
import math
import os
import re

# ---------------------------------------------------------------------------
# Scoring — copied verbatim from benchmark/financebench_pilot.py so the mimic
# scores exactly like the real pilot (these functions are pure stdlib).
# ---------------------------------------------------------------------------


def normalize_text(s):
    s = str(s).lower().strip()
    s = re.sub(r"[$,%]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def extract_numbers(s):
    return [
        float(x.replace(",", ""))
        for x in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(s))
    ]


def numeric_close(pred, gold, rel_tol=0.02, abs_tol=0.05):
    pnums = extract_numbers(pred)
    gnums = extract_numbers(gold)
    if not pnums or not gnums:
        return False
    for p in pnums:
        for g in gnums:
            if abs(p - g) <= max(abs_tol, rel_tol * max(1.0, abs(g))):
                return True
    return False


def weak_answer_match(pred, gold):
    pn = normalize_text(pred)
    gn = normalize_text(gold)
    if numeric_close(pred, gold):
        return True
    if len(pn) >= 12 and len(gn) >= 12 and (gn in pn or pn in gn):
        return True
    return False


def parse_model_answer(prediction):
    text = str(prediction).strip()
    cleaned = text
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "answer" in obj:
            return str(obj["answer"]).strip()
    except Exception:
        pass
    match = re.search(r'"answer"\s*:\s*"([^"]+)"', cleaned)
    if match:
        return match.group(1).strip()
    match = re.search(r"answer\s*:\s*(.+)", cleaned, flags=re.IGNORECASE)
    if match:
        return match.group(1).splitlines()[0].strip()
    return text


def parse_model_confidence(prediction):
    text = str(prediction).strip()
    cleaned = text
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            for key in ["confidence", "calibrated_confidence", "answer_confidence"]:
                if key in obj:
                    value = float(obj[key])
                    return max(0.0, min(1.0, value))
    except Exception:
        pass
    match = re.search(
        r"(?:confidence|calibrated_confidence|answer_confidence)\s*[:=]\s*([01](?:\.\d+)?)",
        cleaned, flags=re.IGNORECASE,
    )
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return None


def is_refusal(answer_text):
    normalized = normalize_text(answer_text)
    markers = ["insufficient_evidence", "insufficient evidence", "not enough evidence",
               "cannot determine", "not provided", "no evidence"]
    return any(m in normalized for m in markers)


def score_prediction(prediction, gold):
    answer_text = parse_model_answer(prediction)
    confidence = parse_model_confidence(prediction)
    weak_match = weak_answer_match(answer_text, gold)
    numeric_match = numeric_close(answer_text, gold)
    refusal = is_refusal(answer_text)
    correctness = 1.0 if weak_match else 0.0
    brier = None if confidence is None else (confidence - correctness) ** 2
    return {
        "parsed_answer": answer_text,
        "weak_match_raw": weak_answer_match(prediction, gold),
        "weak_match_answer": weak_match,
        "numeric_match_answer": numeric_match,
        "refusal": refusal,
        "confidence": confidence,
        "brier": brier,
        "overconfident_wrong": bool(
            confidence is not None and confidence >= 0.8 and not weak_match
        ),
    }


def expected_calibration_error(rows, n_bins=5):
    """ECE over rows with a confidence, using weak_match_answer as correctness."""
    scored = [r for r in rows if r.get("confidence") is not None]
    if not scored:
        return None
    total = len(scored)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if i == n_bins - 1:
            bucket = [r for r in scored if lo <= r["confidence"] <= hi]
        else:
            bucket = [r for r in scored if lo <= r["confidence"] < hi]
        if not bucket:
            continue
        acc = sum(float(r["weak_match_answer"]) for r in bucket) / len(bucket)
        conf = sum(r["confidence"] for r in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(acc - conf)
    return ece


# ---------------------------------------------------------------------------
# Synthetic FinanceBench-like sample (the five examples discussed in
# results_log.md). Kept tiny and deterministic; no dataset download.
# ---------------------------------------------------------------------------

SAMPLE = [
    {
        "id": "walmart_ebitda_margin",
        "question": "What was Walmart's FY2019 EBITDA margin?",
        "answer": "6.1%",
        "evidence_text": "[doc=WMT_2019_10K page=35] Total revenues were $514,405 million. "
                         "Operating income was $21,957 million. Depreciation and amortization "
                         "was $10,678 million.",
        "hard": True,   # requires computing EBITDA/revenue — small model fails
    },
    {
        "id": "acme_retention_ratio",
        "question": "What was the company's dividend retention ratio for FY2022?",
        "answer": "0.79",
        "evidence_text": "[doc=ACME_2022_10K page=52] Net income was $3,200 million. "
                         "Dividends declared were $672 million.",
        "hard": True,
    },
    {
        "id": "amex_operating_margin",
        "question": "Is American Express's FY2022 operating margin above 20%?",
        "answer": "No, American Express's operating margin was approximately 18% in FY2022, below 20%.",
        "evidence_text": "[doc=AXP_2022_10K page=40] Total revenues net of interest expense were "
                         "$52,862 million. Pretax income was $9,570 million.",
        "hard": True,   # sentence-level gold answer; scorer is strict
    },
    {
        "id": "boeing_effective_tax_rate",
        "question": "What was Boeing's FY2018 effective tax rate?",
        "answer": "The effective tax rate was 87.2%.",
        "evidence_text": "[doc=BA_2018_10K page=60] The effective income tax rate for 2018 was 87.2%.",
        "hard": False,  # answer is stated directly -> model gets it
    },
    {
        "id": "mgm_ebitdar_segment",
        "question": "Which MGM segment contributed the most Adjusted Property EBITDAR in FY2019?",
        "answer": "The Las Vegas Strip Resorts segment contributed the most Adjusted Property EBITDAR.",
        "evidence_text": "[doc=MGM_2019_10K page=71] Las Vegas Strip Resorts Adjusted Property "
                         "EBITDAR was $2,509 million; Regional Operations was $968 million; "
                         "MGM China was $704 million.",
        "hard": True,
    },
]

SYSTEM_INSTRUCTION = (
    "You are a careful financial QA assistant. Answer only from the provided "
    "evidence when evidence is provided. If the answer is not supported, say "
    "INSUFFICIENT_EVIDENCE. Return a concise answer and a confidence from 0 to 1."
)


def _numeric_examples():
    return [e for e in SAMPLE if extract_numbers(e["answer"])]


def _counterfactual(answer, factor=1.37):
    nums = extract_numbers(answer)
    if not nums:
        return None
    changed = nums[0] * factor
    text = str(answer)
    rendered = f"{changed:.1f}%" if "%" in text else f"{changed:,.2f}"
    return rendered


# ---------------------------------------------------------------------------
# MockModel — stands in for the HF generator. Returns the same JSON shape a
# small instruct model would, tuned to reproduce results_log.md behavior.
# Swap this for pilot.load_hf_generator(...) to run for real.
# ---------------------------------------------------------------------------

class MockModel:
    model_id = "mock/tiny-instruct"

    def _emit(self, answer, confidence):
        return json.dumps({
            "answer": answer, "confidence": round(confidence, 2),
            "evidence_support": "mock", "short_rationale": "mock stubbed output",
        })

    def baseline(self, ex, condition):
        """condition in {question_only, with_gold_evidence}."""
        if not ex["hard"]:
            # Directly-stated fact: model answers correctly in both settings.
            return self._emit(ex["answer"], 0.90 if condition == "question_only" else 0.95)
        if condition == "question_only":
            # Hard: often refuses or guesses; overconfident.
            if ex["id"] == "walmart_ebitda_margin":
                return self._emit("INSUFFICIENT_EVIDENCE", 0.30)   # 1/5 refusal
            return self._emit("about 12%", 0.75)
        # with_gold_evidence: stops refusing, still wrong, MORE confident (miscalibrated).
        return self._emit("approximately 15%", 0.94)

    def probe(self, ex, condition, target):
        """Grounding probe conditions."""
        if condition in ("evidence_compressed", "direct_grounded_evidence"):
            return self._emit(ex["answer"], 0.98)                 # copies short evidence
        if condition == "counterfactual_direct_evidence":
            return self._emit(target, 0.98)                       # follows counterfactual
        if condition == "missing_evidence":
            if ex["id"] == "acme_retention_ratio":
                return self._emit("roughly 0.5", 0.99)            # 1/3 hallucinates
            return self._emit("INSUFFICIENT_EVIDENCE", 0.99)      # 2/3 abstain
        # gold_evidence: fails to extract from raw evidence, overconfident.
        return self._emit("approximately 15%", 0.94)


# ---------------------------------------------------------------------------
# Aggregation (pure-python version of summarize_results).
# ---------------------------------------------------------------------------

def summarize(rows, group_key, metric_keys):
    groups = {}
    for r in rows:
        groups.setdefault(r[group_key], []).append(r)
    out = []
    for name, grp in sorted(groups.items()):
        entry = {group_key: name}
        for m in metric_keys:
            vals = [float(r[m]) for r in grp if r.get(m) is not None]
            entry[m] = round(sum(vals) / len(vals), 4) if vals else None
        entry["ece"] = round(expected_calibration_error(grp), 4) if expected_calibration_error(grp) is not None else None
        entry["n"] = len(grp)
        out.append(entry)
    return out


def _print_table(title, rows):
    print(f"\n=== {title} ===")
    if not rows:
        print("(no rows)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Stages.
# ---------------------------------------------------------------------------

def run_baseline(model, outdir):
    rows = []
    for ex in SAMPLE:
        for condition in ("question_only", "with_gold_evidence"):
            pred = model.baseline(ex, condition)
            s = score_prediction(pred, ex["answer"])
            rows.append({"condition": condition, "id": ex["id"],
                         "gold_answer": ex["answer"], **s})
    metrics = ["weak_match_raw", "weak_match_answer", "numeric_match_answer",
               "refusal", "confidence", "brier", "overconfident_wrong"]
    summary = summarize(rows, "condition", metrics)
    _print_table("BASELINE SUMMARY (mock)", summary)
    if outdir:
        _write_csv(os.path.join(outdir, "mock_hf_summary.csv"), summary)
        _write_csv(os.path.join(outdir, "mock_hf_results.csv"), rows)
    return summary


def run_probe(model, outdir):
    rows = []
    for ex in _numeric_examples()[:3]:
        target_cf = _counterfactual(ex["answer"])
        conditions = [
            ("gold_evidence", ex["answer"], "answer"),
            ("evidence_compressed", ex["answer"], "answer"),
            ("missing_evidence", "INSUFFICIENT_EVIDENCE", "abstain"),
            ("direct_grounded_evidence", ex["answer"], "answer"),
        ]
        if target_cf:
            conditions.append(("counterfactual_direct_evidence", target_cf, "answer"))
        for condition, target, behavior in conditions:
            pred = model.probe(ex, condition, target)
            s = score_prediction(pred, target)
            success = s["refusal"] if behavior == "abstain" else s["weak_match_answer"]
            rows.append({"condition": condition, "id": ex["id"],
                         "expected_behavior": behavior, "target_answer": target,
                         "probe_success": bool(success), **s})
    metrics = ["probe_success", "weak_match_answer", "numeric_match_answer",
               "refusal", "confidence", "brier", "overconfident_wrong"]
    summary = summarize(rows, "condition", metrics)
    _print_table("GROUNDING PROBE SUMMARY (mock)", summary)
    if outdir:
        _write_csv(os.path.join(outdir, "mock_probe_summary.csv"), summary)
        _write_csv(os.path.join(outdir, "mock_probe_results.csv"), rows)
    return summary


# ---------------------------------------------------------------------------
# Selective accuracy — THE HEADLINE MECHANISM.
#
# Answer only the highest-support cases, abstain on the rest. If the support
# probability tracks correctness, accuracy on the answered subset rises well
# above full-coverage accuracy. This mock uses a synthetic scored set where a
# calibrated support signal correlates with correctness (what the 7B extractor's
# support_probability provides in the real cascade), to SHOW how acc@30/acc@50
# clears 75% even when acc@100 is modest. Illustrative, not real numbers.
# ---------------------------------------------------------------------------

def risk_coverage(scored):
    """scored: list of (support, correct). Returns coverage->selective accuracy."""
    order = sorted(scored, key=lambda x: x[0], reverse=True)
    rows, running = [], 0.0
    n = len(order)
    for k, (_s, correct) in enumerate(order, start=1):
        running += 1.0 if correct else 0.0
        rows.append({"k": k, "coverage": round(k / n, 3),
                     "selective_accuracy": round(running / k, 3)})
    return rows


def accuracy_at_coverage(scored, coverages=(0.3, 0.5, 0.7, 1.0)):
    curve = risk_coverage(scored)
    n = len(curve)
    out = {}
    for c in coverages:
        k = max(1, min(n, round(c * n)))
        out[f"acc@{int(c*100)}"] = next(r["selective_accuracy"] for r in curve if r["k"] == k)
    out["aurc"] = round(sum(1 - r["selective_accuracy"] for r in curve) / n, 3)
    out["full_accuracy"] = curve[-1]["selective_accuracy"]
    return out


def run_selective(outdir):
    """Synthetic calibrated set: ~40% correct overall, support tracks correctness."""
    # 20 items: (support_probability, correct). High support -> more likely correct.
    scored = [
        (0.97, True), (0.95, True), (0.93, True), (0.92, True), (0.90, True),
        (0.88, True), (0.86, True), (0.83, False), (0.80, True), (0.78, False),
        (0.55, False), (0.52, True), (0.50, False), (0.48, False), (0.45, False),
        (0.40, False), (0.35, False), (0.30, False), (0.25, False), (0.20, False),
    ]
    metrics = accuracy_at_coverage(scored)
    row = [{"model": "cascade+selective (mock)", **{k: metrics[k] for k in
            ["acc@30", "acc@50", "acc@70", "acc@100", "aurc"]}, "n": len(scored)}]
    _print_table("SELECTIVE ACCURACY (mock headline mechanism)", row)
    n_correct = sum(1 for _s, c in scored if c)
    print(f"  full accuracy (acc@100) = {n_correct}/{len(scored)} = {n_correct/len(scored):.2f}")
    print("  -> answering only the top-support subset lifts accuracy past 0.75.")
    print("     REAL run: support_probability comes from the 7B extractor (no gold answer).")
    if outdir:
        _write_csv(os.path.join(outdir, "mock_selective.csv"), row)
    return row


def run_tool(outdir):
    """Demo of the v2 method: tool computes the answer; grounding gives confidence;
    selective prediction on that confidence clears 0.80 at moderate coverage."""
    import method_tool as mt

    print("\n=== TOOL-AUGMENTED COMPUTATION (mock demo of the v2 method) ===")
    ex = SAMPLE[0]  # Walmart EBITDA margin
    variables = {"operating_income": 21957, "dep_amort": 10678, "total_revenue": 514405}
    formula = "(operating_income + dep_amort) / total_revenue * 100"
    val = mt.execute_program(variables, formula)
    conf = mt.grounding_confidence(list(variables.values()), ex["evidence_text"], val is not None)
    print(f"  Q: {ex['question']}")
    print(f"    extracted vars = {variables}")
    print(f"    formula = {formula}")
    print(f"    tool result = {mt.format_value(val, 'percent')}  (gold {ex['answer']})")
    print(f"    grounding confidence = {conf} (all inputs found verbatim in evidence)")

    # Representative scored set for tool+grounded answers (illustrative, not real).
    # Tool computation lifts full accuracy from ~0.2 (raw) to ~0.70; grounding
    # confidence tracks correctness, so selective accuracy clears 0.80.
    scored = [(0.95, True), (0.92, True), (0.90, False), (0.88, True), (0.85, True),
              (0.82, True), (0.80, True), (0.78, False), (0.75, True), (0.72, True),
              (0.68, True), (0.65, True), (0.60, True), (0.55, True), (0.50, True),
              (0.45, True), (0.40, False), (0.35, False), (0.30, False), (0.20, False)]
    metrics = mt.accuracy_at_coverage(scored, coverages=(0.3, 0.5, 0.7, 1.0))
    row = [{"method": "tool+grounded+selective (mock)",
            **{k: metrics[k] for k in ["acc@30", "acc@50", "acc@70", "acc@100", "aurc"]},
            "n": len(scored)}]
    _print_table("TOOL METHOD — SELECTIVE ACCURACY (mock headline)", row)
    print("  raw-evidence baseline (from the real run) ≈ 0.20 full accuracy.")
    print("  -> tool computation lifts full accuracy; grounding-calibrated selection clears 0.80.")
    print("     REAL run: method_tool.run_tool_cascade (extractor -> executor -> grounding).")
    if outdir:
        _write_csv(os.path.join(outdir, "mock_tool_selective.csv"), row)
    return row


def main():
    ap = argparse.ArgumentParser(description="Offline mimic of the GEP pilot")
    ap.add_argument("--stages", default="baseline,probe,selective,tool",
                    help="comma list of: baseline,probe,selective,tool")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    model = MockModel()
    print(f"Model: {model.model_id} (MOCK — no HF/network) | sample: {len(SAMPLE)} examples")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if "baseline" in stages:
        run_baseline(model, args.outdir)
    if "probe" in stages:
        run_probe(model, args.outdir)
    if "selective" in stages:
        run_selective(args.outdir)
    if "tool" in stages:
        run_tool(args.outdir)
    print(f"\nDone. CSVs in {args.outdir}/ (mock_*.csv)")
    print("Swap MockModel -> pilot.load_hf_generator(...) and run run_pilot.py for real numbers.")


if __name__ == "__main__":
    main()
