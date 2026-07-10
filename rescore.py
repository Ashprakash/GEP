#!/usr/bin/env python3
"""
rescore.py — re-score existing runs with the CURRENT parser/executor, WITHOUT
re-calling any model. Reads the saved `extractor_raw` from each by_model CSV,
re-parses + re-executes, and recomputes accuracy. Lets you estimate the effect of
executor improvements (name reconciliation, max/min, assignment-strip) for free.

Caveat: `extractor_raw` was truncated to 600 chars in earlier runs, so verbose
outputs (whose formula sits after a long variable list) can't be fully recovered
here — the re-scored number is therefore a LOWER BOUND for those. Rows at the 600
cap are counted and reported as `truncated`.

Usage:  python3 rescore.py                 # all results/by_model/*.csv
        python3 rescore.py results/by_model/Nemotron-Mini-4B-Instruct.csv
"""
import csv
import glob
import os
import sys

import method_tool as m


def _b(s):
    return str(s).strip().lower() == "true"


def rescore(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    old_exec = old_full = 0
    new_exec = new_full = commit_n = commit_correct = trunc = 0
    for r in rows:
        old_exec += _b(r.get("executed", ""))
        old_full += _b(r.get("match_units", ""))
        raw = r.get("extractor_raw", "") or ""
        if len(raw) >= 600:
            trunc += 1
        variables, formula, unit = m.parse_program(raw)
        val = m.compute_answer(variables, formula)
        ok = m.units_tolerant_correct(val, r.get("gold_answer", ""))
        new_exec += (val is not None)
        new_full += ok
        if val is not None:
            commit_n += 1
            commit_correct += ok
    return {
        "model": os.path.basename(path)[:-4], "n": n,
        "old_exec": old_exec / n, "old_full": old_full / n,
        "new_exec": new_exec / n, "new_full": new_full / n,
        "new_commit": (commit_correct / commit_n) if commit_n else 0.0,
        "truncated": trunc,
    }


def main():
    paths = sys.argv[1:] or sorted(glob.glob("results/by_model/*.csv"))
    if not paths:
        raise SystemExit("No by_model CSVs found.")
    print(f"{'model':32} {'n':>3} {'exec':>12} {'full-acc':>14} {'@commit':>8} {'trunc':>6}")
    print("-" * 82)
    for p in paths:
        s = rescore(p)
        print(f"{s['model']:32} {s['n']:>3} "
              f"{s['old_exec']:.2f}->{s['new_exec']:.2f}   "
              f"{s['old_full']:.2f}->{s['new_full']:.2f}    "
              f"{s['new_commit']:>6.2f} {s['truncated']:>6}")
    print("\n(old->new; 'trunc' = rows whose saved raw was truncated at 600 chars, "
          "so new is a LOWER BOUND for those.)")


if __name__ == "__main__":
    main()
