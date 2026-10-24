# ContextRoast report

Dataset: curated-faq-retrieval · Provider: python · Model: examples.retrieval_agent:predict

**Holdout validation: FAIL.** Deployment-ready gate: blocked.

**Python callable execution: local code can incur real API charges. Serialized token proxies do not measure internal model prompts or billing.**

Kept fields: billing_faq, access_faq, shipping_faq

Removed fields: legacy_faq, marketing_faq

Allowed absolute accuracy drop: 0.00% (always compared with the original baseline).

| Metric (entire evaluation set) | Baseline | Final |
| --- | ---: | ---: |
| Accuracy | 100.00% (8/8) | 100.00% (8/8) |
| Estimated input tokens (serialized proxy) | 3,864 | 2,738 |
| Estimated output tokens (serialized proxy) | 214 | 214 |
| Provider-reported calls | 0 / 8 | 0 / 8 |
| Reported input tokens (known calls only) | 0 | 0 |
| Reported output tokens (known calls only) | 0 | 0 |
| Costing source | estimated | estimated |
| Modeled inference cost | unknown | unknown |

Accuracy change: +0.00 percentage points. Serialized input-proxy reduction: 29.14%.

## Evaluation and savings

- Monetary information: unavailable; measured billing: unknown.
- USD prices per million tokens: input unknown, output unknown.
- Selection search: 13 unique subsets, 104 python_callable_invocations; includes baseline and rejected trials.
- Holdout invocations: 4; total evaluation invocations: 108.
- Total evaluation usage: 37,731 estimated input / 2,528 estimated output tokens; 0 calls with provider-reported usage.
- Modeled evaluation token cost: unknown.
- Evaluation execution cost: unknown (unavailable).
- Projected requests: 10,000.
- Estimated gross inference savings: unknown.
- Estimated net savings after execution cost: unknown.
- Estimated net savings after modeled paid evaluation: unknown.
- Modeled paid-evaluation break-even requests: unavailable (unknown usage/prices or no positive savings).
- Execution-cost break-even requests: unavailable (unknown prices or no positive savings) (basis: unavailable).

Future requests have the same case mix and token usage as this dataset. Only accepted-subset inference is projected; the search and any holdout check run once.

## Removal trials

| Round | Field removed | Accuracy | Within tolerance | Chosen |
| ---: | --- | ---: | --- | --- |
| 1 | billing_faq | 62.50% | no | no |
| 1 | access_faq | 62.50% | no | no |
| 1 | shipping_faq | 75.00% | no | no |
| 1 | legacy_faq | 100.00% | yes | no |
| 1 | marketing_faq | 100.00% | yes | yes |
| 2 | billing_faq | 62.50% | no | no |
| 2 | access_faq | 62.50% | no | no |
| 2 | shipping_faq | 75.00% | no | no |
| 2 | legacy_faq | 100.00% | yes | yes |
| 3 | billing_faq | 62.50% | no | no |
| 3 | access_faq | 62.50% | no | no |
| 3 | shipping_faq | 75.00% | no | no |

## Holdout validation

Status: **FAIL**.

Dataset: curated-faq-holdout. Field selection used only the optimization dataset.

Holdout accuracy: 100.00% (2/2) -> 50.00% (1/2); change -50.00 percentage points; allowed drop 0.00 percentage points.

Holdout calls: 4 across 2 subset(s); modeled token cost: unknown, included in total evaluation cost.

Cases correct with full context and incorrect with selected context:

| Case | Input | Expected | Full-context prediction | Selected-context prediction | Error |
| --- | --- | --- | --- | --- | --- |
| logo | Where can I download the company brand logo? | {'article_id': 'logo', 'answer': 'Download the approved logo files from the brand assets page.'} | {'answer': 'Download the approved logo files from the brand assets page.', 'article_id': 'logo'} | {'answer': 'Open Billing, select the month, and choose Download invoice.', 'article_id': 'invoice'} | exact-match failure |

Small holdouts cannot establish generalization or statistical certainty. PASS only means the configured accuracy-drop gate passed on these cases; it is not a production-safety guarantee. Do not retune fields on this holdout.

## Limits and assumptions

- Greedy backward elimination is order-dependent and is not globally optimal.
- Accuracy is measured on this evaluation set; it does not establish statistical certainty or performance on unseen data. Validate on a separate holdout set before deployment.
- Estimated tokens use ceil(UTF-8 bytes / 4) plus approximate message framing, not a model tokenizer. Estimates can be inaccurate, especially for non-English text.
- Costs use flat user-supplied USD prices. Discounts, cached-input rates, tools, taxes, and other billing adjustments are not modeled. Output length can change after removal.
- Small holdouts cannot establish generalization or statistical certainty. PASS only means the configured accuracy-drop gate passed on these cases; it is not a production-safety guarantee. Do not retune fields on this holdout.
- Holdout validation FAIL: exported context is not deployment-ready. Savings projections do not override the observed accuracy regression.
- Python code may execute multiple paid calls per invocation, including at import. Only returned aggregate token usage is observed; import-time charges, unreported calls, and non-token costs are unavailable. Execution must never be assumed free. All monetary amounts are pricing estimates, not measured billing.
- For Python, estimated tokens are serialized input/output proxies, not measured agent prompts. Monetary projections require complete reported usage and both prices.
- Input and output prices are both required for monetary estimates. Missing pricing is unknown, never zero. Token changes remain available.
