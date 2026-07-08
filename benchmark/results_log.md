# Results Log

This file records notable pilot results and interpretation. Numbers are early diagnostics, not paper-ready results.

## 2026-07-08: Stage 0 / Stage 1 Pilot, Qwen 0.5B

Model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Sample:

```text
n = 5 FinanceBench examples
conditions = question_only, with_gold_evidence
```

Colab summary:

| condition | weak_accuracy_raw | weak_accuracy_answer | numeric_accuracy_answer | refusal_rate | n |
|---|---:|---:|---:|---:|---:|
| question_only | 0.4 | 0.2 | 0.2 | 0.2 | 5 |
| with_gold_evidence | 0.4 | 0.4 | 0.2 | 0.0 | 5 |

Initial interpretation:

- Parsed-answer accuracy improved from `0.2` to `0.4` when gold evidence was provided.
- Numeric accuracy remained flat at `0.2`, suggesting the 0.5B model still struggles with financial arithmetic/evidence extraction.
- Refusal dropped from `0.2` to `0.0` with evidence.
- The result weakly supports the grounding diagnostic, but `n=5` is too small for conclusions.

Manual inspection notes:

- Walmart EBITDA margin: model missed both question-only and evidence settings.
- Retention ratio: model hallucinated large numeric values in both settings.
- AMEX operating margin: current scorer over-counted a short `No` answer as correct against a long gold answer. This motivated stricter substring matching.
- Boeing effective tax rate: model answered correctly in both settings.
- MGM EBITDAR: evidence condition moved from wrong region (`China`) to a more relevant answer (`Las Vegas Strip Resorts`), but the weak scorer marked it wrong because the gold answer is phrased as a sentence.

Action:

- Tighten weak string matching to avoid short substring false positives.
- Rerun `n=5` once after scorer fix.
- If stable, run `n=20` on Qwen 0.5B.
- Then test Qwen 1.5B with the same random seed and sample size.

## 2026-07-08: Stage 1 Rerun With Stricter Scoring, Qwen 0.5B

Model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Sample:

```text
n = 5 FinanceBench examples
conditions = question_only, with_gold_evidence
```

Colab summary:

| condition | weak_accuracy_raw | weak_accuracy_answer | numeric_accuracy_answer | refusal_rate | n |
|---|---:|---:|---:|---:|---:|
| question_only | 0.4 | 0.2 | 0.2 | 0.2 | 5 |
| with_gold_evidence | 0.4 | 0.2 | 0.2 | 0.0 | 5 |

Interpretation:

- After stricter parsed-answer scoring, gold evidence did **not** improve answer accuracy for Qwen 0.5B on this tiny sample.
- Gold evidence reduced abstention/refusal from `0.2` to `0.0`, but did not improve correctness.
- This suggests that merely adding long FinanceBench evidence is not enough for a very small model.

Manual inspection:

- The model often states high confidence even when the numeric answer is wrong.
- The model sometimes uses evidence-like language without actually extracting the correct value.
- The MGM example improved semantically from `China` to `Las Vegas Strip Resorts`, but the current scorer still marks it incorrect because the gold answer is sentence-level.

Action:

- Move to the quick grounding probe to test whether the model can follow short, explicit grounded evidence.
- If direct evidence works but real gold evidence fails, the next method component should be evidence compression / evidence-bundle distillation.

## 2026-07-08: Stage 3 Quick Grounding Probe, Qwen 0.5B

Model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Sample:

```text
n = 3 numeric FinanceBench examples
conditions = gold_evidence, missing_evidence, direct_grounded_evidence, counterfactual_direct_evidence
```

Colab summary:

| condition | success_rate | answer_weak_accuracy | answer_numeric_accuracy | refusal_rate | n |
|---|---:|---:|---:|---:|---:|
| counterfactual_direct_evidence | 1.000 | 1.000 | 1.000 | 0.000 | 3 |
| direct_grounded_evidence | 1.000 | 1.000 | 0.667 | 0.000 | 3 |
| gold_evidence | 0.000 | 0.000 | 0.000 | 0.000 | 3 |
| missing_evidence | 0.667 | 0.000 | 0.000 | 0.667 | 3 |

Interpretation:

- The model can follow short direct evidence perfectly on this small probe.
- The model can follow direct counterfactual evidence perfectly, which means it is not hopelessly stuck to memorized answers when evidence is concise and explicit.
- The model fails real FinanceBench gold evidence completely on this sample.
- The model abstains on missing evidence in 2 of 3 examples, but still hallucinates in 1 of 3.

Research signal:

> The bottleneck is not simply that the small model cannot obey grounded evidence. The bottleneck is that real financial evidence is too noisy, long, and structurally complex for the small model to reliably extract the decision-relevant fact.

This supports a sharper GROUNDFIN method hypothesis:

> Grounded probabilistic distillation should teach a student model not only the answer, but also a compact evidence representation, evidence support label, and abstention behavior.

Next action:

- Add an `evidence_compressed` condition that gives the model a short teacher/rule-derived evidence bundle.
- Compare `gold_evidence` vs `evidence_compressed` vs `direct_grounded_evidence`.
- If compressed evidence closes much of the gap, build the method around evidence-bundle distillation.

Follow-up implementation:

- Added `evidence_compressed` to the grounding probe using FinanceBench `justification` as a compact evidence proxy where available.
