#!/usr/bin/env python3
"""
run_all.py — run the GEP tool method across a set of models, resiliently.

Resilience (so nothing is lost if it crashes mid-sweep):
  * per-EXAMPLE checkpoint: results/by_model/<label>.jsonl (resume within a model)
  * per-MODEL checkpoint:    results/by_model/<label>.csv  (skip finished models)
  * the combined multi-model table is regenerated after EVERY model
  * a failure in one model is logged to the manifest and the sweep continues

Usage (on your Mac):
  python run_all.py                                   # default 4 open families, n=50
  python run_all.py --n 30 --models "Qwen/Qwen2.5-7B-Instruct,microsoft/Phi-3.5-mini-instruct"
  python run_all.py --force                           # recompute even finished models
Resume: just re-run the same command — done models/examples are skipped automatically.
"""

import argparse
import csv
import json
import os
import sys
import traceback

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Pin the HF cache to the repo's .hf_cache so models aren't re-downloaded when the
# shell wasn't `source activate.sh`-ed. setdefault respects an already-set HF_HOME.
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache"))

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-ai/deepseek-llm-7b-chat",
    "nvidia/Nemotron-Mini-4B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]

# stdlib metric helpers, reused so aggregation needs no pandas
from write_results import _load, _correct_fn, _committed  # noqa: E402


def model_summary(csv_path, label):
    rows = _load(csv_path)
    if not rows:
        return None
    correct, scorer = _correct_fn(rows)
    n = len(rows)
    committed = [r for r in rows if _committed(r)]
    cov = len(committed) / n
    full = sum(correct(r) for r in rows) / n
    commit = (sum(correct(r) for r in committed) / len(committed)) if committed else 0.0
    return {"model": label, "family": label.split("-")[0], "n": n,
            "coverage": round(cov, 3), "acc_full": round(full, 3),
            "acc_commit": round(commit, 3), "scorer": scorer}


def baseline_acc(csv_path):
    """Full-coverage accuracy of raw gold-evidence prompting (strict weak-match)."""
    rows = _load(csv_path)
    if not rows:
        return None
    bg = [r for r in rows if r.get("condition") == "with_gold_evidence"] or rows
    correct, _ = _correct_fn(bg)
    return round(sum(correct(r) for r in bg) / len(bg), 3)


def write_combined(bymodel_dir, outdir):
    """Regenerate the combined multi-model table from whatever per-model CSVs exist."""
    summaries = []
    for fn in sorted(os.listdir(bymodel_dir)):
        if not fn.endswith(".csv") or fn.endswith("_baseline.csv"):
            continue
        label = fn[:-4]
        s = model_summary(os.path.join(bymodel_dir, fn), label)
        if not s:
            continue
        base_path = os.path.join(bymodel_dir, f"{label}_baseline.csv")
        s["baseline"] = baseline_acc(base_path) if os.path.exists(base_path) else None
        summaries.append(s)
    if not summaries:
        return

    def bf(x):
        return "TBD" if x is None else f"{x:.2f}"

    with open(os.path.join(outdir, "multimodel.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    md = ["| Model | n | Baseline | Coverage | Acc@full | Acc@commit | Scorer |",
          "|---|:--:|:--:|:--:|:--:|:--:|:--:|"]
    for s in summaries:
        md.append(f"| {s['model']} | {s['n']} | {bf(s.get('baseline'))} | {s['coverage']:.2f} | "
                  f"{s['acc_full']:.2f} | **{s['acc_commit']:.2f}** | {s['scorer']} |")
    with open(os.path.join(outdir, "multimodel.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    tex = []
    for s in summaries:
        tex.append(f"{s['model'].replace('_', chr(92)+'_')} & {bf(s.get('baseline'))} "
                   f"& \\textbf{{{s['acc_commit']:.2f}}} & {s['coverage']:.2f} \\\\")
    with open(os.path.join(outdir, "multimodel.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    print("  [combined] " + " | ".join(
        f"{s['model']}:{bf(s.get('baseline'))}->{s['acc_commit']:.2f}@{s['coverage']:.2f}"
        for s in summaries))


def append_manifest(path, entry):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "status", "detail"])
        if not exists:
            w.writeheader()
        w.writerow(entry)


def main():
    ap = argparse.ArgumentParser(description="Resilient multi-model method sweep")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-evidence-chars", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--benchmark", default="financebench",
                    help="financebench | finqa | drop (use a distinct --outdir per benchmark)")
    ap.add_argument("--with-baseline", action="store_true",
                    help="also run+capture the raw gold-evidence baseline per model")
    ap.add_argument("--force", action="store_true", help="recompute finished models")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bymodel = os.path.join(args.outdir, "by_model")
    os.makedirs(bymodel, exist_ok=True)
    manifest = os.path.join(args.outdir, "run_all_manifest.csv")

    # Heavy imports here so --help stays light and errors are actionable.
    try:
        from run_pilot import build_mac_loader
        from benchmark import financebench_pilot as pilot
        import method_tool
    except ModuleNotFoundError as e:
        sys.exit(f"Missing dependency: {e}. Run: pip install -r requirements-colab.txt")

    if args.device != "cuda":
        pilot.load_hf_generator = build_mac_loader(args.device)

    import benchmarks
    print(f"Loading benchmark '{args.benchmark}'… (models: {len(models)})")
    df = benchmarks.get_benchmark(args.benchmark)

    for model in models:
        label = model.split("/")[-1]
        per_csv = os.path.join(bymodel, f"{label}.csv")
        per_jsonl = os.path.join(bymodel, f"{label}.jsonl")

        # --- tool method (skip if already done) ---
        if os.path.exists(per_csv) and not args.force:
            print(f"[skip] {label} tool (already done)")
        else:
            print(f"\n[run] {label} tool — n={args.n} …")
            try:
                results = method_tool.run_tool_cascade(
                    df, extractor_model_id=model, n_examples=args.n,
                    random_state=args.seed, max_evidence_chars=args.max_evidence_chars,
                    max_new_tokens=args.max_new_tokens, checkpoint_path=per_jsonl,
                )
                tmp = per_csv + ".tmp"
                results.to_csv(tmp, index=False)
                os.replace(tmp, per_csv)
                s = model_summary(per_csv, label)
                append_manifest(manifest, {"model": model, "status": "OK",
                                           "detail": f"commit={s['acc_commit']} cov={s['coverage']} n={s['n']}"})
                print(f"[done] {label} tool: acc@commit={s['acc_commit']:.2f} @ cov {s['coverage']:.2f}")
            except KeyboardInterrupt:
                print(f"\n[interrupted] partial progress saved in {per_jsonl}; re-run to resume.")
                raise
            except Exception as e:
                traceback.print_exc()
                append_manifest(manifest, {"model": model, "status": "FAILED", "detail": str(e)[:180]})
                print(f"[FAIL] {label} tool: {e} — continuing")

        # --- raw gold-evidence baseline (optional; skip if already done) ---
        if args.with_baseline:
            base_csv = os.path.join(bymodel, f"{label}_baseline.csv")
            if os.path.exists(base_csv) and not args.force:
                print(f"[skip] {label} baseline (already done)")
            else:
                print(f"[run] {label} baseline — n={args.n} …")
                try:
                    bres, _ = pilot.run_hf_baseline(
                        df, n_examples=args.n, model_id=model, random_state=args.seed,
                        max_new_tokens=args.max_new_tokens, max_evidence_chars=args.max_evidence_chars,
                    )
                    tmp = base_csv + ".tmp"
                    bres.to_csv(tmp, index=False)
                    os.replace(tmp, base_csv)
                    ba = baseline_acc(base_csv)
                    append_manifest(manifest, {"model": model, "status": "OK-baseline",
                                               "detail": f"baseline={ba}"})
                    print(f"[done] {label} baseline: {ba}")
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    traceback.print_exc()
                    append_manifest(manifest, {"model": model, "status": "FAILED-baseline", "detail": str(e)[:180]})
                    print(f"[FAIL] {label} baseline: {e} — continuing")

        write_combined(bymodel, args.outdir)           # refresh after each model

    write_combined(bymodel, args.outdir)
    print(f"\nSweep complete. Combined tables: {args.outdir}/multimodel.{{csv,md,tex}}")
    print(f"Per-model results: {bymodel}/  | manifest: {manifest}")


if __name__ == "__main__":
    main()
