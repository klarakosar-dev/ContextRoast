# Example results

The examples use curated data. No production agent or real model endpoint was benchmarked.

| Example | Full-context accuracy | Selected accuracy | Input-size reduction |
| --- | --- | --- | --- |
| Support mock | 8/8 | 8/8 | 52.05% estimated tokens |
| Shipping mock | 6/6 | 6/6 | 31.24% estimated tokens |
| Python FAQ retriever | 8/8 | 8/8 | 29.14% serialized input proxy |
| FAQ holdout | 2/2 | **1/2** | — |

The mock outcomes follow their built-in rules. The Python retriever searches all supplied collections using word-count cosine similarity; it has no hardcoded collection names or access to expected answers.

## Reproduce the FAQ result

From an installed checkout:

```bash
contextroast run examples/retrieval.json \
  --provider python --entrypoint examples.retrieval_agent:predict \
  --allow-paid --holdout examples/retrieval-holdout.json \
  --export-context --output-dir reports/faq
```

The command exits 1 after writing reports. Selection keeps `billing_faq`, `access_faq` and `shipping_faq`, and removes `legacy_faq` and `marketing_faq`. It evaluates 13 subsets across eight cases: 104 callable invocations, plus four holdout invocations.

The logo question needs the removed marketing collection. On holdout, the reduced context answers it incorrectly; the exported schema records `FAIL` and `deployment_ready: false`. See [the report](holdout-example-report.md).

The estimated serialized input falls from 3,864 to 2,738. The function reports no internal model usage, so monetary savings remain unknown. Two holdout cases cannot establish generalisation.

Run the mocks with `contextroast run examples/support.json` or `contextroast run examples/shipping.yaml`, choosing a fresh `--output-dir` for each run. Prices supplied to the mocks describe hypothetical paid usage; actual mock execution is free.
