# Experiment Plan — ≥10 Defensible Results for a Top-Tier Submission

Target: a strong (7–8) submission. Honest bar: that needs **the distillation result**, not just
the prompt-time pilot. The pilot is the *motivation*; the trained student is the *contribution*.
Every result below is LLM-judge scored (see `llm_judge.py`), with a human-agreement check.

## Scoring protocol (applies to every result)
- **Primary metric: LLM-as-judge** (a stronger, different-family judge than the model under test),
  reporting full-coverage accuracy AND accuracy@coverage (selective).
- Report weak-match and units-tolerant alongside for transparency.
- **Judge validation (Result 10):** hand-label a stratified 40–60 subset; report judge-vs-human
  Cohen's κ (target > 0.7). Without this, reviewers discount the judge.
- Always paired with baselines at **matched model + coverage**.

## The results matrix (≥10)

| # | Result | What it establishes | Status |
|---|--------|---------------------|--------|
| 1 | **Baseline sweep**: question-only vs raw gold-evidence, across Qwen 0.5/1.5/3/7B + Gemma-2 2b/9b | Small models fail on financial QA; scale trend | Qwen-7B baseline done (~0.22); per-family TBD |
| 2 | **Tool method vs raw baseline**, per model | Core lift | **DONE: 12 cells (4 models × 3 benchmarks). GEP@commit > baseline in all 12; coverage-fair yield 7/12. See RESULTS.md** |
| 3 | **Selective risk–coverage curves**, per model | Reliability payoff (acc@20/30/50) | pilot done |
| 4 | **Ablation ladder**: extract-only → +compute → +verify → +abstain | Which component drives the gain | to run |
| 5 | **Calibration**: ECE/Brier for verbalized vs grounding vs combo confidence | The confidence signal is real | pilot partial |
| 6 | **By task-type** (computational vs selection/comparison/textual) | Scope + the "task-form, not reasoning" finding | pilot done |
| 7 | **Counterfactual-evidence robustness** | Method follows changed evidence, not memory | harness exists |
| 8 | **Missing-evidence abstention** (does it correctly abstain) | Safety / selective correctness | harness exists |
| 9 | **Scorer triangulation**: LLM-judge vs weak-match vs units-tolerant agreement | Methodological rigor | pilot: judge==units (8/13) |
| 10 | **Judge-vs-human agreement** (κ on a labeled subset) | Validates the judge | to do |
| 11 | **DISTILLATION** (the contribution): student SFT on answer-only vs rationale vs tool-trace+abstention; held-out, company-disjoint | Small student internalizes the method, beats baselines | to run (needs GPU) |
| 12 | **Teacher-gap recovery** + selective acc of the distilled student | The headline method result | to run (needs GPU) |
| 13 | **Full-set n=150** stable numbers with CIs | Statistical solidity | to run |

Results 1–10 are the *diagnostic/measurement* half (mostly runnable on the Mac). Results 11–12
are the *method* half and need CUDA (LoRA SFT / GRPO — Colab/GPU box, per `train_groundfin.py`).

## Multi-model requirement
For generality: **≥3 families** — Qwen2.5 (0.5/3/7B), DeepSeek (deepseek-llm-7b-chat), NVIDIA
Nemotron (Nemotron-Mini-4B), Phi-3.5-mini (all open), + Gemma-2/Mistral/Llama (gated) as extras —
each through results 1–3 and 5–6. This kills the "Qwen-7B artifact" objection. All Mac-runnable
in fp16 (no bitsandbytes) via `run_pilot.py --device auto`. Use plain instruct/chat models, not
reasoning models (R1-distills emit `<think>` and break JSON extraction).

## Reproducibility (reviewers check this)
- Fixed seeds; report mean±std across ≥3 seeds for headline tables.
- Release: schema, prompts, `run_pilot.py`, `run_all.py` (resilient sweep), `method_tool.py`,
  `llm_judge.py`, `rescore.py` (re-score saved outputs after executor changes, no re-run), and
  the judge prompt.
- Pure-greedy decode (no repetition penalties — they corrupt JSON extraction; verified).
- Full `extractor_raw` saved per row → any executor change is re-scored offline via `rescore.py`.

## Honest gating
- Prompt-time pilot alone → workshop/short paper.
- Pilot + distillation showing a small student beats answer-only/rationale distillation on
  accuracy AND calibration AND counterfactual robustness → the 7–8 top-tier submission.
- If distillation doesn't beat baselines → pivot to the **benchmark/diagnostic paper**
  (FinGKD-Bench): still publishable, built from results 1–10.

## Run order
1. Mac: results 1–3, 5–6, 9 across the model set (LLM-judge scored). ← establishes motivation
2. Mac: results 7–8 (counterfactual / missing-evidence).
3. Label subset → result 10 (judge validation).
4. GPU: results 11–12 (distillation) — the contribution.
5. Full-set n=150 + seeds → result 13.
