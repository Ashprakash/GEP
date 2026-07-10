# Results — GEP across 4 models × 3 benchmarks

Primary results from the full run (Colab). Scorer: units-tolerant. `acc@commit` = accuracy on
answered questions; `coverage` = fraction answered (the method abstains when it cannot build a
verifiable program). `yield = acc@commit × coverage` = fraction of ALL questions answered
correctly (the coverage-fair comparison vs the full-coverage baseline). Regenerate the table with
`python3 make_table.py`.

## Master table (12 cells)

| Dataset | Model | Baseline | GEP@commit | Cov | GEP yield (acc×cov) | Δyield vs base |
|---|---|:--:|:--:|:--:|:--:|:--:|
| FinanceBench | Qwen2.5-7B   | 0.24 | 0.66 | 0.62 | 0.41 | **+0.17** |
| FinanceBench | DeepSeek-7B  | 0.18 | 0.52 | 0.55 | 0.29 | **+0.11** |
| FinanceBench | Nemotron-4B  | 0.14 | 0.35 | 0.38 | 0.13 | −0.01 |
| FinanceBench | Phi-3.5-mini | 0.36 | 0.50 | 0.20 | 0.10 | −0.26 |
| FinQA | Qwen2.5-7B   | 0.25 | 0.57 | 0.95 | 0.54 | **+0.29** |
| FinQA | DeepSeek-7B  | 0.16 | 0.18 | 0.55 | 0.10 | −0.06 |
| FinQA | Nemotron-4B  | 0.04 | 0.41 | 0.70 | 0.29 | **+0.25** |
| FinQA | Phi-3.5-mini | 0.18 | 0.45 | 0.87 | 0.39 | **+0.21** |
| DROP | Qwen2.5-7B   | 0.22 | 0.42 | 0.80 | 0.34 | **+0.12** |
| DROP | DeepSeek-7B  | 0.17 | 0.28 | 0.45 | 0.13 | −0.04 |
| DROP | Nemotron-4B  | 0.06 | 0.21 | 0.65 | 0.14 | **+0.08** |
| DROP | Phi-3.5-mini | 0.40 | 0.48 | 0.53 | 0.25 | −0.15 |

## What the data supports (honest)
1. **GEP@commit beats baseline in all 12 cells** — structured extract-compute raises
   accuracy-on-answered-questions everywhere (up to +0.42 absolute on Qwen/FinanceBench).
2. **On the coverage-fair yield metric it's 7 wins / 5 losses.** GEP wins total-yield when
   **coverage ≳ 0.6**; it loses when it abstains too much (Phi/FinanceBench @0.20 → −0.26;
   DeepSeek @0.45–0.55). So the "12/12" is partly abstention flattering acc@commit — report yield too.
3. **Coverage tracks benchmark type**: FinanceBench (heterogeneous) 0.20–0.62 < DROP (numeric RC)
   0.45–0.80 < FinQA (pure numeric) 0.55–0.95. The method applies where questions are computational.
4. **Accuracy tracks model capability**, not size alone (Qwen-7B ≫ DeepSeek-7B at equal size).

## Confidence-signal sub-study (compare_selective.py)
Which signal to abstain on — grounding (deterministic) vs verbalized (self-report) vs log-prob
(parametric) vs their ensemble — is **benchmark- and model-dependent; no signal transfers**:

| Benchmark | Best AURC signal |
|---|---|
| FinanceBench | verbalized (Qwen) / grounding (small models) |
| FinQA | grounding (2/3) |
| DROP | log-prob (2/3) |

Same model (Qwen) flips best-signal across benchmarks (verbalized → grounding → ensemble). All
AURCs are high (0.42–0.75; weak selectors). **One consistent thread: the grounding×verbalized
ensemble has the lowest ECE** on most cells — best-calibrated, though not best-ranking.
→ Honest finding: *no single confidence signal solves selective abstention for LLM numeric
reasoning; the ensemble is the most robustly calibrated.*

## Framing for the paper
- **Headline (supported):** GEP trades coverage for precision across 4 models × 3
  benchmarks — a reliability improvement for high-stakes numeric reasoning, where abstention
  ("needs review") is preferable to a confident wrong number.
- **Sub-finding (supported):** the abstention *signal* is an open problem — none transfers;
  ensemble best-calibrated.
- **NOT claimed** (falsified by ablation): "deterministic grounding universally beats self-report,"
  and a clean capability/size phase transition (confounded by family; not stable across benchmarks).

## Must-do before submission
- Report **yield** alongside acc@commit (done above); disclose low-coverage losses (Phi).
- Give the **baseline its own abstention** and compare at **matched coverage** (add to
  compare_selective) — otherwise the acc@commit win is contested.
- Standardize **n** across cells; add **≥3 seeds + CIs**; validate the LLM-judge vs human (κ).

## Pending
- Matched-coverage baseline-vs-GEP selective comparison. **TBD**
- Seeds/CIs, uniform n, judge-vs-human κ. **TBD**
- Distillation (student learns extract-compute-abstain) — the learning contribution. **TBD (GPU)**
