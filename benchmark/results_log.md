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
