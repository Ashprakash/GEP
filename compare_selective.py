#!/usr/bin/env python3
"""
compare_selective.py — THE decisive DRC experiment.

For each model's committed answers, compare three confidence signals as selective
predictors: grounding (deterministic), verbalized (self-report), and log-prob
(parametric). Reports AURC (lower=better), ECE (lower=better), and accuracy at
coverage. The DRC claim is EARNED only if `support_probability` (grounding) has
the lowest AURC and ECE.

Pure stdlib; runs on the by_model CSVs (no models needed).
Usage: python3 compare_selective.py            # all results/by_model/*.csv
       python3 compare_selective.py results/by_model/Qwen2.5-7B-Instruct.csv
"""
import csv
import glob
import os
import sys

SIGNALS = [("support_probability", "grounding (DRC)"),
           ("verbalized_confidence", "verbalized"),
           ("logprob_confidence", "log-prob"),
           ("combo", "grounding x verbal")]
COVS = (0.2, 0.5, 1.0)


def _b(s):
    return str(s).strip().lower() == "true"


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _correct(r):
    return _b(r.get("match_units", "")) or _b(r.get("weak_match_answer", ""))


def _risk_coverage(pairs):
    order = sorted(pairs, key=lambda x: x[0], reverse=True)
    n, cum, curve = len(order), 0, []
    for i, (_c, ok) in enumerate(order, 1):
        cum += 1 if ok else 0
        curve.append(cum / i)
    aurc = sum(1 - a for a in curve) / n
    at = {c: curve[max(1, min(n, round(c * n))) - 1] for c in COVS}
    return aurc, at


def _ece(pairs, bins=5):
    tot = len(pairs)
    if not tot:
        return None
    e = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        bkt = [p for p in pairs if (lo <= p[0] < hi) or (i == bins - 1 and p[0] >= hi - 1e-9)]
        if not bkt:
            continue
        acc = sum(1 for _c, ok in bkt if ok) / len(bkt)
        conf = sum(c for c, _ok in bkt) / len(bkt)
        e += len(bkt) / tot * abs(acc - conf)
    return e


def analyze(path):
    committed = [r for r in csv.DictReader(open(path)) if _b(r.get("executed", ""))]
    out = {}
    for key, _label in SIGNALS:
        pairs = [(_f(r.get(key)), _correct(r)) for r in committed if _f(r.get(key)) is not None]
        if len(pairs) < 3:
            out[key] = None
            continue
        aurc, at = _risk_coverage(pairs)
        out[key] = {"n": len(pairs), "aurc": round(aurc, 3), "ece": round(_ece(pairs), 3),
                    "acc@20": round(at[0.2], 3), "acc@50": round(at[0.5], 3),
                    "acc@100": round(at[1.0], 3)}
    # ensemble: grounding x verbalized (tests if neither alone suffices)
    combo = [(_f(r.get("support_probability")) * _f(r.get("verbalized_confidence")), _correct(r))
             for r in committed
             if _f(r.get("support_probability")) is not None and _f(r.get("verbalized_confidence")) is not None]
    if len(combo) >= 3:
        aurc, at = _risk_coverage(combo)
        out["combo"] = {"n": len(combo), "aurc": round(aurc, 3), "ece": round(_ece(combo), 3),
                        "acc@20": round(at[0.2], 3), "acc@50": round(at[0.5], 3),
                        "acc@100": round(at[1.0], 3)}
    else:
        out["combo"] = None
    return out


def main():
    paths = sys.argv[1:] or [p for p in sorted(glob.glob("results/by_model/*.csv"))
                             if not p.endswith("_baseline.csv")]
    wins = {k: 0 for k, _ in SIGNALS}
    for path in paths:
        model = os.path.basename(path)[:-4]
        res = analyze(path)
        print(f"\n### {model}")
        print(f"{'signal':22} {'n':>3} {'AURC↓':>7} {'ECE↓':>7} {'acc@20':>7} {'acc@50':>7} {'acc@100':>8}")
        best_aurc, best_key = 9.9, None
        for key, label in SIGNALS:
            s = res.get(key)
            if not s:
                print(f"{label:22} {'--- not available (rerun to populate) ---'}")
                continue
            print(f"{label:22} {s['n']:>3} {s['aurc']:>7.3f} {s['ece']:>7.3f} "
                  f"{s['acc@20']:>7.2f} {s['acc@50']:>7.2f} {s['acc@100']:>8.2f}")
            if s["aurc"] < best_aurc:
                best_aurc, best_key = s["aurc"], key
        if best_key:
            wins[best_key] += 1
            print(f"  -> lowest AURC: {dict(SIGNALS)[best_key]}")
    print("\n=== VERDICT (models where each signal had the best AURC) ===")
    for key, label in SIGNALS:
        print(f"  {label:22} {wins[key]}")
    if wins["support_probability"] > max(wins["verbalized_confidence"], wins["logprob_confidence"]):
        print("\n  DRC SUPPORTED: grounding wins on more models.")
    else:
        print("\n  DRC NOT YET SUPPORTED: grounding does not dominate — reframe or improve the signal.")


if __name__ == "__main__":
    main()
