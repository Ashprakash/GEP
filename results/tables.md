## Main (Qwen2.5-7B, n=30, scorer=units-tolerant)

| Method | Cov. | Acc@full | Acc@commit | Exec. |
|---|:--:|:--:|:--:|:--:|
| Raw gold-evidence (baseline) | 1.00 | --- | --- | --- |
| GEP (ours) | 0.43 | 0.27 | **0.62** | 0.43 |

## Selective (ranked by verbalized_confidence)

| Coverage | 20% | 30% | 50% | 70% | 100% |
|---|:--:|:--:|:--:|:--:|:--:|
| Accuracy | 1.00 | 0.67 | 0.40 | 0.33 | 0.27 |

## By task type
| Task type | Acc | Exec | n |
|---|:--:|:--:|:--:|
| guidance_delta | 1.00 | 1.00 | 1 |
| line_item_lookup | 0.60 | 0.60 | 5 |
| ratio_calculation | 0.27 | 0.64 | 11 |
| period_comparison | 0.17 | 0.33 | 6 |
| cash_flow_category_selection | 0.00 | 0.00 | 1 |
| generic_financial_qa | 0.00 | 0.00 | 6 |
