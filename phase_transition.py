#!/usr/bin/env python3
"""
phase_transition.py — the finding, computed directly.

Reads a within-family size sweep (by_model CSVs), and for each model size reports
the selective-abstention AURC of grounding vs verbalized vs their ensemble. Orders
by model size so the calibration crossover (if any) is visible: the size at which
verbalized confidence overtakes deterministic grounding.

Usage:
  python3 phase_transition.py results/qwen_sweep/by_model/*.csv
  python3 phase_transition.py                      # defaults to results/by_model/*.csv
"""
import glob
import os
import re
import sys

from compare_selective import analyze

_SIZE = re.compile(r"(\d+\.?\d*)\s*[bB]\b")


def _size(name):
    m = _SIZE.search(name)
    return float(m.group(1)) if m else None


def _fmt(x):
    return "  --  " if x is None else f"{x:.3f}"


def main():
    paths = [p for p in (sys.argv[1:] or sorted(glob.glob("results/by_model/*.csv")))
             if not p.endswith("_baseline.csv")]
    rows = []
    for p in paths:
        name = os.path.basename(p)[:-4]
        res = analyze(p)
        g = (res.get("support_probability") or {}).get("aurc")
        v = (res.get("verbalized_confidence") or {}).get("aurc")
        c = (res.get("combo") or {}).get("aurc")
        rows.append((_size(name), name, g, v, c))
    rows.sort(key=lambda r: (r[0] is None, r[0] or 0))

    print(f"{'size(B)':>7}  {'model':30} {'ground':>7} {'verbal':>7} {'combo':>7}  winner")
    print("-" * 74)
    crossover = None
    for sz, name, g, v, c in rows:
        cands = {k: x for k, x in (("ground", g), ("verbal", v), ("combo", c)) if x is not None}
        win = min(cands, key=cands.get) if cands else "-"
        print(f"{str(sz):>7}  {name:30} {_fmt(g):>7} {_fmt(v):>7} {_fmt(c):>7}  {win}")
        if crossover is None and g is not None and v is not None and v < g:
            crossover = sz
    print("-" * 74)
    if crossover is not None:
        print(f"FINDING: verbalized confidence overtakes deterministic grounding at ~{crossover}B "
              f"(calibration crossover). Below it, grounding is the better selector; above it, "
              f"self-report wins. Ensemble is the robust default.")
    else:
        print("FINDING: no crossover observed — deterministic grounding not surpassed by "
              "verbalized confidence across the swept sizes (report as-is).")


if __name__ == "__main__":
    main()
