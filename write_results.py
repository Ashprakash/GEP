#!/usr/bin/env python3
"""
write_results.py — turn run CSVs into paper-ready tables (Markdown + LaTeX bodies).

Pure stdlib (no pandas), so it runs anywhere. Reads a tool-method results CSV (and,
if present, the LLM-judged version and a baseline CSV), computes the reported metrics,
and prints/writes:
  - Markdown tables (for RESULTS.md)
  - LaTeX \begin{tabular} row bodies matching paper/gep.tex (Tables 1-4)

Correctness column preference (per row): judge_correct > match_units > weak_match_answer.

Usage:
  python3 write_results.py                                  # defaults under results/
  python3 write_results.py --tool results/tool_results.csv \
      --baseline results/hf_results.csv --model-label Qwen2.5-7B
Outputs: results/tables.md and results/tables.tex
"""

import argparse
import csv
import os

COVERAGES = (0.2, 0.3, 0.5, 0.7, 1.0)
TASK_ORDER = ["guidance_delta", "line_item_lookup", "ratio_calculation",
              "period_comparison", "cash_flow_category_selection", "generic_financial_qa"]


def _truthy(s):
    return str(s).strip().lower() in ("true", "1", "yes")


def _float(s, default=0.0):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def _correct_fn(rows):
    """Pick the best available correctness column, return (fn, name)."""
    cols = rows[0].keys() if rows else []
    if "judge_correct" in cols:
        return (lambda r: _truthy(r["judge_correct"])), "LLM-judge"
    if "match_units" in cols:
        return (lambda r: _truthy(r["match_units"])), "units-tolerant"
    return (lambda r: _truthy(r.get("weak_match_answer", ""))), "strict"


def _committed(r):
    return _truthy(r.get("executed", "")) or bool(str(r.get("computed", "")).strip())


def _acc_at_coverage(rows, correct, conf_key):
    ranked = sorted(rows, key=lambda r: _float(r.get(conf_key, 0.0)), reverse=True)
    n = len(ranked)
    if n == 0:
        return {c: None for c in COVERAGES}
    cum, curve = 0, []
    for i, r in enumerate(ranked, 1):
        cum += 1 if correct(r) else 0
        curve.append(cum / i)
    return {c: curve[max(1, min(n, round(c * n))) - 1] for c in COVERAGES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default="results/tool_results.csv")
    ap.add_argument("--judged", default="results/tool_results_judged.csv")
    ap.add_argument("--baseline", default="results/hf_results.csv")
    ap.add_argument("--conf", default="verbalized_confidence",
                    help="confidence column for selective ranking")
    ap.add_argument("--model-label", default="Qwen2.5-7B")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    tool = _load(args.judged) or _load(args.tool)
    if not tool:
        raise SystemExit(f"No tool results at {args.judged} or {args.tool}")
    correct, scorer_name = _correct_fn(tool)
    n = len(tool)
    committed = [r for r in tool if _committed(r)]
    exec_rate = len(committed) / n
    full_acc = sum(correct(r) for r in tool) / n
    commit_acc = (sum(correct(r) for r in committed) / len(committed)) if committed else 0.0
    conf_key = args.conf if (tool and args.conf in tool[0]) else "support_probability"
    sel = _acc_at_coverage(tool, correct, conf_key)

    # baseline (raw gold-evidence) full-coverage accuracy, if available
    base = _load(args.baseline)
    base_acc = None
    if base:
        bc, _ = _correct_fn(base)
        bg = [r for r in base if r.get("condition") == "with_gold_evidence"] or base
        base_acc = sum(bc(r) for r in bg) / len(bg)

    # by task type
    tt = {}
    for r in tool:
        t = r.get("task_type", "?")
        tt.setdefault(t, []).append(r)

    def f(x):
        return "---" if x is None else f"{x:.2f}"

    # ---- Markdown ----
    md = []
    md.append(f"## Main ({args.model_label}, n={n}, scorer={scorer_name})\n")
    md.append("| Method | Cov. | Acc@full | Acc@commit | Exec. |")
    md.append("|---|:--:|:--:|:--:|:--:|")
    md.append(f"| Raw gold-evidence (baseline) | 1.00 | {f(base_acc)} | {f(base_acc)} | --- |")
    md.append(f"| GEP (ours) | {exec_rate:.2f} | {full_acc:.2f} | **{commit_acc:.2f}** | {exec_rate:.2f} |\n")
    md.append(f"## Selective (ranked by {conf_key})\n")
    md.append("| Coverage | 20% | 30% | 50% | 70% | 100% |")
    md.append("|---|:--:|:--:|:--:|:--:|:--:|")
    md.append(f"| Accuracy | {f(sel[0.2])} | {f(sel[0.3])} | {f(sel[0.5])} | {f(sel[0.7])} | {f(sel[1.0])} |\n")
    md.append("## By task type\n| Task type | Acc | Exec | n |\n|---|:--:|:--:|:--:|")
    for t in TASK_ORDER + [k for k in tt if k not in TASK_ORDER]:
        if t not in tt:
            continue
        g = tt[t]
        a = sum(correct(r) for r in g) / len(g)
        e = sum(_committed(r) for r in g) / len(g)
        md.append(f"| {t} | {a:.2f} | {e:.2f} | {len(g)} |")
    md_text = "\n".join(md) + "\n"

    # ---- LaTeX table bodies (paste into gep.tex) ----
    tex = []
    tex.append("% Table 1 (main) body")
    tex.append(f"Raw gold-evidence (baseline) & 1.00 & {f(base_acc)} & {f(base_acc)} & --- \\\\")
    tex.append(f"\\textbf{{\\textsc{{GEP}} (ours)}} & {exec_rate:.2f} & {full_acc:.2f} & \\textbf{{{commit_acc:.2f}}} & {exec_rate:.2f} \\\\")
    tex.append("% Table selective body")
    tex.append(f"Accuracy & {f(sel[0.2])} & {f(sel[0.3])} & {f(sel[0.5])} & {f(sel[1.0])} \\\\")
    tex.append("% Table task-type body")
    for t in TASK_ORDER + [k for k in tt if k not in TASK_ORDER]:
        if t not in tt:
            continue
        g = tt[t]
        a = sum(correct(r) for r in g) / len(g)
        e = sum(_committed(r) for r in g) / len(g)
        tex.append(f"{t.replace('_', chr(92)+'_')} & {a:.2f} & {e:.2f} & {len(g)} \\\\")
    tex.append("% Table 5 (multi-model) row for this run")
    tex.append(f"{args.model_label} & {f(base_acc)} & \\textbf{{{commit_acc:.2f}}} \\\\")
    tex_text = "\n".join(tex) + "\n"

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "tables.md"), "w") as fp:
        fp.write(md_text)
    with open(os.path.join(args.outdir, "tables.tex"), "w") as fp:
        fp.write(tex_text)

    print(md_text)
    print("=" * 60)
    print(tex_text)
    print(f"wrote {args.outdir}/tables.md and {args.outdir}/tables.tex")


if __name__ == "__main__":
    main()
