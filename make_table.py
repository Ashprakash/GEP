#!/usr/bin/env python3
"""
make_table.py — the master results table: 4 models x 3 datasets (12 rows),
baseline vs GEP (method) side by side. Reads the per-benchmark by_model
dirs. Missing cells show TBD (not fabricated). Writes results/master_table.{md,tex}.

Usage: python3 make_table.py
"""
import os

from write_results import _load, _correct_fn, _committed

DATASETS = [("FinanceBench", "results/by_model"),
            ("FinQA", "results/finqa/by_model"),
            ("DROP", "results/drop/by_model")]
MODELS = [("Qwen2.5-7B", "Qwen2.5-7B-Instruct"),
          ("DeepSeek-7B", "deepseek-llm-7b-chat"),
          ("Nemotron-4B", "Nemotron-Mini-4B-Instruct"),
          ("Phi-3.5-mini", "Phi-3.5-mini-instruct")]


def tool_metrics(path):
    rows = _load(path)
    if not rows:
        return None
    correct, _ = _correct_fn(rows)
    n = len(rows)
    comm = [r for r in rows if _committed(r)]
    cov = len(comm) / n
    commit = sum(correct(r) for r in comm) / len(comm) if comm else 0.0
    return {"n": n, "cov": round(cov, 2), "commit": round(commit, 2)}


def baseline_acc(path):
    rows = _load(path)
    if not rows:
        return None
    bg = [r for r in rows if r.get("condition") == "with_gold_evidence"] or rows
    c, _ = _correct_fn(bg)
    return round(sum(c(r) for r in bg) / len(bg), 2)


def fmt(x):
    return "TBD" if x is None else f"{x:.2f}"


def main():
    rows = []
    for ds, d in DATASETS:
        for disp, fname in MODELS:
            tm = tool_metrics(os.path.join(d, f"{fname}.csv"))
            ba = baseline_acc(os.path.join(d, f"{fname}_baseline.csv"))
            commit = tm["commit"] if tm else None
            cov = tm["cov"] if tm else None
            yld = round(commit * cov, 2) if (commit is not None and cov is not None) else None
            delta = round(yld - ba, 2) if (yld is not None and ba is not None) else None
            rows.append({"dataset": ds, "model": disp, "baseline": ba,
                         "gf_commit": commit, "gf_cov": cov, "gf_yield": yld,
                         "delta": delta, "n": tm["n"] if tm else 0})

    md = ["| Dataset | Model | Baseline | GEP@commit | Cov | GEP yield (acc×cov) | Δyield vs base | n |",
          "|---|---|:--:|:--:|:--:|:--:|:--:|:--:|"]
    for r in rows:
        md.append(f"| {r['dataset']} | {r['model']} | {fmt(r['baseline'])} | "
                  f"**{fmt(r['gf_commit'])}** | {fmt(r['gf_cov'])} | {fmt(r['gf_yield'])} | "
                  f"{fmt(r['delta'])} | {r['n']} |")
    md_text = "\n".join(md) + "\n"

    tex = ["% dataset & model & baseline & GEP@commit & cov & GEP-yield & Δyield"]
    for r in rows:
        tex.append(f"{r['dataset']} & {r['model']} & {fmt(r['baseline'])} & "
                   f"\\textbf{{{fmt(r['gf_commit'])}}} & {fmt(r['gf_cov'])} & "
                   f"{fmt(r['gf_yield'])} & {fmt(r['delta'])} \\\\")
    tex_text = "\n".join(tex) + "\n"

    os.makedirs("results", exist_ok=True)
    open("results/master_table.md", "w").write(md_text)
    open("results/master_table.tex", "w").write(tex_text)
    print(md_text)
    print("wrote results/master_table.md and results/master_table.tex")


if __name__ == "__main__":
    main()
