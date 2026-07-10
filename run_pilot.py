#!/usr/bin/env python3
"""
run_pilot.py — terminal entry point for the GEP pilot (no Colab needed).

Mirrors the runnable cells of benchmark/groundfin_colab_runner.ipynb so you can
get the numbers from a plain shell:

    baseline  -> question_only vs with_gold_evidence   (Stage 1)
    probe     -> gold/compressed/missing/direct/counterfactual evidence (Stages 2-4)
    template  -> raw vs summary vs trace vs template reliability

Everything routes through the tested logic in benchmark/financebench_pilot.py.

Requires network (HuggingFace) + the deps in requirements-colab.txt, and works
best on a GPU. CPU works for the tiny default sample, just slower.

Examples:
    python run_pilot.py                              # all stages, Qwen 0.5B, tiny n
    python run_pilot.py --stages baseline --n 20
    python run_pilot.py --model Qwen/Qwen2.5-1.5B-Instruct --n 50 --outdir results/
"""

import argparse
import os
import sys

# Let unsupported fp16 ops fall back to CPU instead of crashing on Apple MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Pin the HF cache to the repo's .hf_cache so models aren't re-downloaded when the
# shell wasn't `source activate.sh`-ed. setdefault respects an already-set HF_HOME.
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache"))

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _resolve_device(choice):
    """auto -> cuda if present, else Apple MPS, else CPU."""
    import torch
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_mac_loader(device_choice):
    """A HF generator loader that targets Apple MPS / CPU with NO bitsandbytes.

    Drop-in for pilot.load_hf_generator (same signature). We ignore load_in_4bit
    because 4-bit needs CUDA bitsandbytes; on Mac we run fp16 (MPS) or fp32 (CPU)
    dense weights instead. Fine for <=3B instruct models generating pilot numbers.
    """
    def load(model_id, max_new_tokens=192, load_in_4bit=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        device = _resolve_device(device_choice)
        dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        model.to(device)
        print(f"[mac-loader] {model_id} on {device} ({dtype})"
              + (" [ignored load_in_4bit: no CUDA bitsandbytes]" if load_in_4bit else ""))
        return pipeline(
            "text-generation", model=model, tokenizer=tokenizer, device=device,
            max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False,
            # NOTE: no repetition_penalty / no_repeat_ngram_size — they corrupt JSON
            # (force the model to misspell repeated tokens like '", "'), which broke
            # structured extraction. Pure greedy is the right decode for JSON output.
        )
    return load


def _write_and_show(name, summary, results, outdir):
    """Print a summary table and persist both summary and per-row results."""
    summary_reset = summary.reset_index()
    print(f"\n=== {name.upper()} SUMMARY ===")
    print(summary_reset.to_string(index=False))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        sp = os.path.join(outdir, f"{name}_summary.csv")
        rp = os.path.join(outdir, f"{name}_results.csv")
        summary_reset.to_csv(sp, index=False)
        results.to_csv(rp, index=False)
        print(f"wrote {sp} and {rp}")


def main():
    ap = argparse.ArgumentParser(description="GEP pilot runner")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF instruct model id")
    ap.add_argument("--stages", default="all",
                    help="comma list of: baseline,probe,template  (or 'all')")
    ap.add_argument("--n", type=int, default=5,
                    help="baseline sample size (FinanceBench examples)")
    ap.add_argument("--probe-n", type=int, default=3,
                    help="probe base numeric examples (each expands to 5 conditions)")
    ap.add_argument("--template-n", type=int, default=3,
                    help="template base examples (each expands to 6 conditions)")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--max-evidence-chars", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--extractor", default="Qwen/Qwen2.5-7B-Instruct",
                    help="cascade extractor model (turns raw evidence -> compact bundle + support prob)")
    ap.add_argument("--readers", default="",
                    help="cascade: comma list of reader models (default: --model). "
                         "e.g. Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"],
                    help="compute device; 'auto' picks MPS on Apple Silicon (no CUDA needed)")
    ap.add_argument("--outdir", default="results",
                    help="directory for *_summary.csv / *_results.csv (empty to skip)")
    args = ap.parse_args()

    # Import here so --help works without the heavy deps installed.
    try:
        from benchmark import financebench_pilot as pilot
    except ModuleNotFoundError as e:
        sys.exit(f"Missing dependency: {e}. Run: pip install -r requirements-colab.txt")

    # On Mac/CPU, swap the repo's CUDA-oriented loader for a device-aware one
    # (fp16 on MPS, fp32 on CPU, no bitsandbytes). CUDA keeps the original loader.
    if args.device != "cuda":
        pilot.load_hf_generator = build_mac_loader(args.device)

    stages = ["baseline", "probe", "template"] if args.stages == "all" \
        else [s.strip() for s in args.stages.split(",") if s.strip()]

    print(f"Loading FinanceBench (PatronusAI/financebench)…")
    df = pilot.load_financebench()
    print(f"Loaded {len(df)} examples, {df['company'].nunique()} companies.")
    print(f"Model: {args.model} | stages: {stages}")

    if "baseline" in stages:
        results, summary = pilot.run_hf_baseline(
            df, n_examples=args.n, model_id=args.model,
            random_state=args.seed, max_new_tokens=args.max_new_tokens,
            max_evidence_chars=args.max_evidence_chars,
        )
        _write_and_show("hf", summary, results, args.outdir)

    if "probe" in stages:
        results, summary = pilot.run_hf_grounding_probe(
            df, n_examples=args.probe_n, model_id=args.model,
            random_state=args.seed, max_new_tokens=args.max_new_tokens,
            max_evidence_chars=args.max_evidence_chars,
        )
        _write_and_show("probe", summary, results, args.outdir)

    if "template" in stages:
        results, summary = pilot.run_hf_template_comparison(
            df, n_examples=args.template_n, model_id=args.model,
            random_state=args.seed, max_new_tokens=args.max_new_tokens,
            max_evidence_chars=args.max_evidence_chars,
        )
        _write_and_show("template", summary, results, args.outdir)

    if "cascade" in stages:
        # THE HEADLINE PATH: 7B extractor -> compact bundle + support prob (no gold
        # answer); reader answers from the bundle; report selective accuracy@coverage
        # ranked by support probability. This is the repo's route to a 75%+ number.
        readers = tuple(r.strip() for r in args.readers.split(",") if r.strip()) \
            or (args.model,)
        results, summary = pilot.run_extraction_cascade(
            df, extractor_model_id=args.extractor, reader_model_ids=readers,
            n_examples=args.n, random_state=args.seed,
            max_evidence_chars=args.max_evidence_chars, max_new_tokens=args.max_new_tokens,
        )
        print("\n=== EXTRACTION CASCADE SUMMARY (accuracy + calibration) ===")
        print(summary.to_string(index=False))

        bundle = results[results["condition"] == "cascade_bundle"]
        sel = pilot.selective_summary(bundle, ["reader_id"],
                                      confidence_col="support_probability")
        print("\n=== SELECTIVE ACCURACY — bundle, ranked by support_probability ===")
        print("(acc@30 / acc@50 are the headline; abstain on the rest)")
        print(sel.to_string(index=False))

        if args.outdir:
            os.makedirs(args.outdir, exist_ok=True)
            summary.to_csv(os.path.join(args.outdir, "cascade_summary.csv"), index=False)
            results.to_csv(os.path.join(args.outdir, "cascade_results.csv"), index=False)
            sel.to_csv(os.path.join(args.outdir, "cascade_selective.csv"), index=False)
            print(f"\nwrote cascade_summary.csv, cascade_results.csv, cascade_selective.csv to {args.outdir}/")

    if "tool" in stages:
        # v2 METHOD: extractor -> typed variables + formula -> deterministic executor
        # -> grounding-calibrated confidence -> selective accuracy. Fixes the numeric
        # bottleneck (offloads arithmetic) and gives a real confidence for selection.
        import method_tool
        results = method_tool.run_tool_cascade(
            df, extractor_model_id=args.extractor, n_examples=args.n,
            random_state=args.seed, max_evidence_chars=args.max_evidence_chars,
            max_new_tokens=args.max_new_tokens,
        )
        acc = float(results["weak_match_answer"].astype(float).mean())
        acc_u = float(results["match_units"].astype(float).mean())
        exec_rate = float(results["executed"].astype(float).mean())
        print(f"\n=== TOOL METHOD (extractor={args.extractor}) ===")
        print(f"full-coverage accuracy: {acc:.3f} (units-tolerant: {acc_u:.3f}) "
              f"| executed: {exec_rate:.3f} | n={len(results)}")

        # Scope to the computable subset — the question types a scalar formula can fit.
        COMPUTABLE = {"ratio_calculation", "cash_flow_line_item", "guidance_delta",
                      "line_item_lookup", "period_comparison"}
        sub = results[results["task_type"].isin(COMPUTABLE)]
        if len(sub):
            print(f"computable subset: acc {sub['weak_match_answer'].astype(float).mean():.3f} "
                  f"(units-tolerant {sub['match_units'].astype(float).mean():.3f}) "
                  f"| executed {sub['executed'].astype(float).mean():.3f} | n={len(sub)}")

        # Which question types does the scalar tool method actually fit?
        bt = results.groupby("task_type").agg(
            accuracy=("weak_match_answer", "mean"),
            units_acc=("match_units", "mean"),
            executed=("executed", "mean"),
            n=("weak_match_answer", "size"),
        ).round(3)
        print("\n=== BY TASK TYPE ===")
        print(bt.to_string())
        print("\nfail reasons:", results["fail_reason"].value_counts(dropna=False).to_dict())

        # Report selective accuracy by BOTH confidence signals + their product, so one
        # (slow) run settles which selector wins. combo needs no extra model calls.
        results["combo"] = (results["support_probability"].astype(float)
                            * results["verbalized_confidence"].astype(float))
        sels = {}
        for sig in ["support_probability", "verbalized_confidence", "combo"]:
            sel = pilot.selective_summary(results, ["extractor_id"], confidence_col=sig)
            sels[sig] = sel
            print(f"\n=== SELECTIVE ACCURACY — ranked by {sig} ===")
            print(sel.to_string(index=False))

        if args.outdir:
            os.makedirs(args.outdir, exist_ok=True)
            results.to_csv(os.path.join(args.outdir, "tool_results.csv"), index=False)
            for sig, sel in sels.items():
                sel.to_csv(os.path.join(args.outdir, f"tool_selective_{sig}.csv"), index=False)
            print(f"\nwrote tool_results.csv + tool_selective_*.csv to {args.outdir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
