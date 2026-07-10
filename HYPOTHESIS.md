# GEP — Hypothesis & Measurement Protocol

The goal is a *measured research result*, not a leaderboard score. This file fixes the
hypothesis angle and exactly what to report, so any run's numbers are publishable as-is.

## The claim (revised to what 4 models × 3 benchmarks actually support)

**GEP — extract typed variables + formula → deterministic execution → abstain when no
verifiable program can be built — trades coverage for precision, improving accuracy on answered
questions over raw-evidence prompting across all 4 models × 3 benchmarks (up to +0.42), a
reliability gain for high-stakes numeric reasoning. Separately, we show that *which* confidence
signal to abstain on is unsolved: deterministic grounding, verbalized self-report, and
log-probability each win on different benchmarks, none transfers, and their ensemble is the most
robustly calibrated.**

### What is NOT claimed (falsified by our own ablations)
- ✗ "Deterministic grounding universally beats verbalized confidence" — false; the best signal is
  benchmark- and model-dependent (verbalized wins on FinanceBench-Qwen, grounding on FinQA-Qwen,
  ensemble/log-prob on DROP).
- ✗ A clean capability/scale "phase transition" — confounded (family ≠ size) and not stable across
  benchmarks. Would require a within-family size sweep with seeds to even test.

### Honest metrics (report both)
- `acc@commit` (accuracy on answered questions) — GEP wins **12/12 cells**.
- `yield = acc@commit × coverage` (coverage-fair vs full-coverage baseline) — GEP wins
  **7/12**, losing where it abstains too much (low coverage, e.g. Phi/FinanceBench @0.20).
- The defensible headline uses acc@commit **with** yield reported and a matched-coverage
  baseline-with-abstention comparison (pending).

See `RESULTS.md` for the full 12-cell table and the confidence sub-study.

### Why this is the right, strong claim
- It is **relative** (method vs. raw at identical model/scorer), so it is immune to "your model
  is small" and "your scorer is strict" objections — both sides carry the same handicap.
- It **relocates the bottleneck**: the field assumes evidence access (retrieval/long context) is
  the problem; we show extraction-and-computation is, which reframes how small financial models
  should be built (extract → tool → verify → abstain), not just fed more context.
- The **failure taxonomy is part of the result**: the gap to full coverage is generation/format
  reliability (an engineering axis), *not* reasoning — which is itself a publishable finding and
  points directly at the distillation follow-up.

**Sub-claims, each already measurable with the current harness:**
1. **Raw evidence barely helps small models.** 0.5B ≈ 0.20 with gold evidence; 7B raw 0.22. *(measured)*
2. **The gap is computation, not access.** Numeric accuracy stays ~0.2 even with gold evidence;
   `direct_grounded_evidence` (answer stated) ≈ 1.0. *(measured)*
3. **Tool decomposition computes correctly when the program is clean.** Every well-formed
   `(variables, formula)` produced a right answer (quick ratio, working capital, ratio margin,
   deltas, lookups). *(measured)*
4. **Grounding/verification gives a usable selective signal.** `acc@low-coverage > acc@full`. *(measured, modest)*
5. **Applicability is bounded by task type** — the scalar-formula method fits computational
   questions but not selection / list / yes-no-explanation questions. *(measured — a finding about
   the benchmark itself)*

Any of these landing — including #5 as a *negative* bound — is a result.

## What the pilot measured (honest, current)
- Full-set accuracy is low (~0.20) because FinanceBench is heterogeneous.
- Failure decomposition on n=30: ~27% **generation instability** (7B fp16 on MPS emits garbage),
  ~40% **task-type mismatch** (non-computational questions), plus fixable parser/unit issues.
- On the **computational subset**, clean programs compute correct answers; units-tolerant scoring
  recovers right-value/wrong-scale cases (e.g. `400 ↔ $400,000,000`).

## Measurement protocol (what to run and report)
Report on the **computational subset** (`ratio_calculation`, `cash_flow_line_item`,
`guidance_delta`, `line_item_lookup`, `period_comparison`) AND on the full set, always separating
the two. For each, report:

| Metric | Why |
|--------|-----|
| accuracy (strict) + **units-tolerant** accuracy | headline; units-tolerant is the fair one for finance |
| executed rate | how often a runnable program was produced |
| selective `acc@30/50/70/100` (by grounding / verbalized / combo) | reliability payoff |
| by-task-type table | supports sub-claim #5 (applicability bound) |
| fail-reason breakdown | the honest failure taxonomy (infra vs structural vs bug) |

Baselines to run at matched settings (the harness already has them):
`question_only`, `with_gold_evidence` (raw), and the tool method — so the claim is *relative*
(method vs raw), not absolute.

Run:
```bash
python run_pilot.py --stages baseline           --n 50 --model Qwen/Qwen2.5-7B-Instruct --device auto
python run_pilot.py --stages tool --max-new-tokens 256 --n 50 --extractor Qwen/Qwen2.5-7B-Instruct --device auto
python run_pilot.py --stages tool --max-new-tokens 256 --n 50 --extractor Qwen/Qwen2.5-3B-Instruct --device auto  # stability check
python3 inspect_tool.py
```

## Reporting rules (so it survives review)
- Always show **strict AND units-tolerant** accuracy; never hide the scorer.
- Always **split computational vs full set**; never quote a subset number as the whole.
- Report the **failure taxonomy** — generation instability is infra, task mismatch is structural,
  and saying so is a strength, not a weakness.
- Selective numbers are "accuracy at N% coverage", with abstention rate shown.
- No oracle/answer-embedded condition in the headline (upper-bound table only).

## Positioning
The measured story is either framing:
- **Method-scoped:** "Tool-verified grounded reasoning improves computational financial QA and
  calibration for small models where raw-evidence prompting fails" — with the task-type bound as
  an honest scope statement.
- **Benchmark/diagnostic:** "Why grounded financial QA is hard for small models" — decomposing
  the failure into computation, task heterogeneity, extraction, and calibration.

Both are supported by the *same* measured tables. Pick after the n=50 subset numbers land.
