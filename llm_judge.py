"""
llm_judge.py — reproducible LLM-as-judge scoring for GEP results.

Reads a results CSV (needs columns: question, gold_answer, and a prediction column),
asks a judge model whether each prediction is correct (unit- and semantics-aware),
and writes a judged CSV + prints accuracy (full-coverage and conditional-on-answer).

Judge should be a DIFFERENT, stronger model than the one under test (avoid self-bias),
e.g. GPT-4-class or Claude via an OpenAI-compatible endpoint. Configure with env:
    JUDGE_BASE_URL  (e.g. https://api.openai.com/v1  or a Groq/Anthropic-compatible URL)
    JUDGE_API_KEY
    JUDGE_MODEL     (e.g. gpt-4o, claude-...-via-proxy, etc.)

Usage:
    python llm_judge.py results/tool_results.csv --pred-col computed
    python llm_judge.py results/hf_results.csv   --pred-col parsed_answer

Needs network + the judge key (run on your Mac, not the sandbox). For a top-tier paper,
also hand-label a stratified subset and report judge-vs-human agreement (Cohen's kappa).
"""

import argparse
import csv
import json
import os
import sys

JUDGE_SYSTEM = (
    "You are a meticulous financial-QA grader. Decide whether the MODEL ANSWER is "
    "correct relative to the GOLD ANSWER for the QUESTION. Judge on financial meaning, "
    "not string overlap. Rules:\n"
    "- Accept unit/scale equivalence ($400M == $400,000,000; 13,200 million == $13.2B).\n"
    "- Accept rounding within ~1% and equivalent phrasings.\n"
    "- For yes/no questions, the key conclusion (and any decisive number) must match.\n"
    "- A number that answers a different quantity than asked is INCORRECT.\n"
    "- If the model abstained / gave no answer, mark incorrect (but note abstained=true).\n"
    'Return strict JSON: {"correct": true|false, "abstained": true|false, "reason": "<short>"}'
)


def judge_one(client, model, question, gold, pred):
    prompt = (f"QUESTION:\n{question}\n\nGOLD ANSWER:\n{gold}\n\n"
              f"MODEL ANSWER:\n{pred if str(pred).strip() else '(no answer / abstained)'}\n\n"
              "Grade it.")
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content
    try:
        obj = json.loads(text[text.find("{"):text.rfind("}") + 1])
        return bool(obj.get("correct")), bool(obj.get("abstained")), str(obj.get("reason", ""))
    except Exception:
        return ("true" in text.lower()[:40]), False, text[:120]


def main():
    ap = argparse.ArgumentParser(description="LLM-as-judge scoring")
    ap.add_argument("csv")
    ap.add_argument("--pred-col", default="computed")
    ap.add_argument("--gold-col", default="gold_answer")
    ap.add_argument("--q-col", default="question")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base_url = os.getenv("JUDGE_BASE_URL")
    api_key = os.getenv("JUDGE_API_KEY")
    model = os.getenv("JUDGE_MODEL")
    if not (api_key and model):
        sys.exit("Set JUDGE_API_KEY and JUDGE_MODEL (and JUDGE_BASE_URL for non-OpenAI).")
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    judged, correct, answered, correct_answered = [], 0, 0, 0
    for r in rows:
        pred = r.get(args.pred_col, "")
        is_correct, abstained, reason = judge_one(
            client, model, r[args.q_col], r[args.gold_col], pred)
        r["judge_correct"], r["judge_abstained"], r["judge_reason"] = is_correct, abstained, reason
        judged.append(r)
        correct += int(is_correct)
        if not abstained and str(pred).strip():
            answered += 1
            correct_answered += int(is_correct)
        mark = "OK " if is_correct else "XX "
        print(f"{mark} gold={str(r[args.gold_col])[:40]!r} pred={str(pred)[:24]!r} :: {reason[:70]}")

    n = len(rows)
    print("\n=== LLM-JUDGE RESULT ===")
    print(f"full-coverage accuracy : {correct}/{n} = {correct / n:.3f}")
    if answered:
        print(f"accuracy when answered : {correct_answered}/{answered} = {correct_answered / answered:.3f} "
              f"(coverage {answered / n:.3f})")

    out = args.out or args.csv.replace(".csv", "_judged.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(judged[0].keys()))
        w.writeheader()
        w.writerows(judged)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
