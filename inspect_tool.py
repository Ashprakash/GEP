#!/usr/bin/env python3
"""Inspect a results CSV: executed rate, failure reasons, and per-example
prediction vs gold. Works on tool-method CSVs (results/by_model/<label>.csv,
results/tool_results.csv) and baseline CSVs (<label>_baseline.csv).

Usage:
  python3 inspect_tool.py                                   # results/tool_results.csv
  python3 inspect_tool.py results/by_model/Phi-3.5-mini-instruct.csv
  python3 inspect_tool.py results/by_model/deepseek-llm-7b-chat_baseline.csv
"""
import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "results/tool_results.csv"
d = pd.read_csv(path)
cols = set(d.columns)

acc_col = "match_units" if "match_units" in cols else (
    "weak_match_answer" if "weak_match_answer" in cols else None)
pred_col = "computed" if "computed" in cols else (
    "parsed_answer" if "parsed_answer" in cols else "prediction")

exec_str = f"{d['executed'].astype(float).mean():.3f}" if "executed" in cols else "n/a"
acc_str = f"{d[acc_col].astype(float).mean():.3f}" if acc_col else "n/a"
print(f"rows: {len(d)} | executed: {exec_str} | accuracy({acc_col}): {acc_str}")
if "fail_reason" in cols:
    print("fail reasons:", d["fail_reason"].value_counts(dropna=False).to_dict())

for _, r in d.iterrows():
    print("=" * 80)
    if "condition" in cols:
        print("cond:", r["condition"])
    print("Q:", str(r.get("question", ""))[:110])
    print(f"gold: {str(r.get('gold_answer',''))[:60]} | pred: {r.get(pred_col,'')} "
          f"| correct: {r.get(acc_col, '')}")
    if "formula" in cols:
        print(f"n_vars: {r.get('n_vars','')} | formula: {r.get('formula','')!r} | fail: {r.get('fail_reason','')}")
    raw = r.get("extractor_raw", r.get("prediction", ""))
    print("RAW:", str(raw)[:450])
