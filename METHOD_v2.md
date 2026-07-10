# GEP v2 — Tool-Verified Grounded Decisions (the path to 80%)

This is the method tweak driven by the pilot diagnostics. Read
[`HOW_TO_RUN.md`](HOW_TO_RUN.md) to run it.

## Why v1 stalled
The pilot showed the bottleneck is **computation, not evidence representation**:
- `numeric_accuracy ≈ 0.2` **even with gold evidence** — the numbers are present, the model
  can't do the arithmetic.
- `direct_grounded_evidence` (answer stated, no math) scored ~1.0 — so grounding works; *computing*
  doesn't.
- Compact bundles didn't beat raw and pushed refusal to 0.6 — prose compression drops the numbers.
- The token-overlap "support probability" had `support_corr ≈ 0` — no signal for selective prediction.

Compressing evidence can't fix an arithmetic gap, and a decorative probability can't power
selective accuracy. So v2 changes the mechanism.

## The v2 pipeline
```
raw evidence + question
   → EXTRACT typed variables + a formula        (LLM, never sees the gold answer)
   → COMPUTE with a deterministic executor       (tool / program-of-thought — the arithmetic)
   → CONFIDENCE = grounding-completeness × execution-success   (a REAL calibrated signal)
   → SELECTIVE: answer if confident, else abstain → report accuracy@coverage
```
Implemented in [`method_tool.py`](method_tool.py); run via `run_pilot.py --stages tool`.

Two fixes, mapped to the two failures:
1. **Executor fixes computation.** The model only extracts `(variables, formula)`; a sandboxed
   evaluator does the math. Small models *can* extract and *can't* compute — so we only ask them
   for what they're good at.
2. **Grounding confidence fixes calibration.** A program that executed *and* drew every input
   verbatim from the evidence is usually right; a hallucinated/failed one is usually wrong. That
   correlation (unlike token overlap) is what makes `acc@50`/`acc@70` climb past 80%.

## What "80%" means here (so it survives review)
- **Reported as selective accuracy at a stated coverage** on grounded-answerable questions —
  e.g. "82% at 60% coverage" — **not** full-coverage accuracy of a small model on raw evidence.
- **The method must beat the baselines at the same coverage**: answer-only, rationale-only,
  raw-evidence, and length-matched compact evidence. An 80% that isn't compared at matched
  coverage is not a result.
- Report the **abstention rate** and **AURC** alongside, and a **teacher-gap-recovery** number.
- Oracle/direct-answer conditions stay as an upper-bound table only — never the headline.

## Why this is the strongest *novel* framing
Individually, tool use (PoT), self-consistency, selective prediction, and calibration are known.
The defensible contribution is the **combination, specialized to finance and made verifiable**:

> **Verifiable grounded financial decisions for small LMs** — extract typed decision variables +
> formula from raw filings (no gold answer), compute deterministically, and abstain unless the
> decision is fully grounded and executable — yielding calibrated, counterfactual-robust answers
> that a small student can be distilled to reproduce.

The novelty rests on: (a) **grounding-completeness × execution as a calibrated confidence** for
financial QA, (b) the **finance-typed variable/formula schema** (generic summaries provably don't
help — your data), and (c) distilling this *verifiable* behavior into a small student, evaluated
on counterfactual / missing / stale splits. The 80% is the *evidence* the method works, not the
claim itself.

## The safety net
If the executor path still underperforms at n≥50, the **benchmark paper** stands on its own: a
rigorous demonstration that capable models fail to compute from real filings, miscalibrate, and
cling to memorized values under counterfactual evidence. Publishable regardless of the method.

## Run it
```bash
# offline mechanism demo (no deps):
python3 mock_pilot.py --stages tool

# real, on your Mac (7B extractor, MPS):
python run_pilot.py --stages tool --n 30 \
  --extractor Qwen/Qwen2.5-7B-Instruct --device auto
```
`tool_selective_*.csv` hold the `acc@coverage` headline for each confidence signal
(grounding, verbalized self-verification, and their product); `tool_results.csv` has the
per-example program, computed value, and both confidences for auditing and any post-hoc selector.
