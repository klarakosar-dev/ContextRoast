# Holdout checks

Holdout validation compares full and selected context on separate cases after field selection is complete.

## Run the check

From an installed checkout:

```bash
contextroast run examples/retrieval.json \
  --provider python --entrypoint examples.retrieval_agent:predict --allow-paid \
  --holdout examples/retrieval-holdout.json --holdout-tolerance 0 \
  --export-context --output-dir reports/holdout-check
```

Python entrypoints require review and explicit consent: imports and callable execution are unsandboxed and can incur real API charges. This inspected example makes no network calls. The command intentionally exits **1** because the supplied holdout reveals a regression. Reports and diagnostic exports are written before that exit.

## Selection and validation stay separate

1. Load and check both datasets before any provider call or Python entrypoint import.
2. Run the greedy optimizer on the optimization dataset only.
3. Keep that selection fixed. Evaluate all original fields and the selected fields on holdout cases using the same provider.
4. Compare selected holdout accuracy with full-context holdout accuracy minus `--holdout-tolerance`.
5. Write the status, case-level regressions, invocation/cost totals, and export readiness metadata. Never choose another subset or retune using holdout scores.

`--tolerance` controls selection against the original optimization baseline. `--holdout-tolerance` independently controls the validation gate against the holdout baseline. Both are absolute fractions: `0.05` means 5 percentage points. Holdout tolerance defaults to zero. Without `--holdout`, no validation calls occur, the command keeps its previous stdout/exit behavior, and reports say `NOT_VALIDATED`. A holdout tolerance without a holdout does not validate anything.

The loader requires the same context field names and exact task instruction. Field declaration order may differ. Mock mode also requires matching `mock_task`. Identical optimization input/context requests are rejected even if their case IDs or labels were changed. This detects obvious overlap, not semantic duplicates, related entities, or all forms of data leakage. Holdout case IDs may overlap optimization IDs if the actual input/context request differs. Prefer genuinely independent, representative cases.

Invalid/missing/incompatible holdouts and invalid tolerances stop before loading user code or making model calls. The same provider instance is used after selection; stateful/stochastic code can affect measurements, as in existing evaluations. If no fields were removed, the identical holdout subset is evaluated once and reused, avoiding duplicate fees.

## Status and exit codes

| Result | Meaning | CLI exit | Export flag |
| --- | --- | ---: | --- |
| `PASS` | Selected holdout accuracy is within the configured drop from the holdout baseline | 0 | `deployment_ready` may be true, within the limited scope below |
| `FAIL` | Holdout accuracy drops beyond tolerance | 1, after artifacts are written | false |
| `NOT_VALIDATED` | No holdout was supplied | 0 | false |
| Configuration/provider failure | Validation could not complete reliably | 2; no complete report | no new completed export |

`deployment_ready` is a **limited holdout gate flag**, not an assertion that a system is safe for production. It is true only for `PASS` with some correct predictions in both the selected optimization and holdout evaluations. Zero correctness never earns that label even if relative accuracy did not drop. Application-specific minimum quality, representative coverage, and independent review remain the developer's responsibility.

Small holdouts cannot establish generalization or statistical certainty. Passing two examples proves only that these two comparisons met your tolerance. Do not repeatedly use a holdout to choose fields or increase tolerance after observing failures; use a new independent check if you retune the system. ContextRoast cannot enforce an operator's experimental process.

## Case-level explanations and exports

`report.json` contains `holdout.status`, full-context and selected-context scores/predictions, tolerance, accuracy change, regressed cases, and improved case IDs. A regression means a case was correct with full context and incorrect with selected context; cases already incorrect are not mislabeled as regressions. Markdown shows the regressed case ID, input, expected answer, both predictions, and any final prediction error. This list is retained even when improvements elsewhere or a nonzero tolerance allow an aggregate `PASS`.

Holdout regression reports include inputs and expected answers and may contain private data. Keep generated reports out of public commits. The optional field-name schema and removal trace carry status/readiness metadata without copying those case values. Schema metadata is under `x-contextroast`; trace metadata is under `validation`. `FAIL` and `NOT_VALIDATED` schemas explicitly say **not deployment-ready** in their titles, with a false readiness flag. Fields remain available as diagnostic recommendations for review.

No fields are automatically reintroduced after failure. Doing so based on holdout results would turn this holdout into selection data.

## Invocation and cost accounting

For `F` optimization fields, `N` optimization cases, and `H` holdout cases, the conservative bound is:

```text
N × (1 + F(F+1)/2) + 2H provider invocations
```

Actual holdout calls can be `H` if full/final fields are identical. One Python invocation can make any number of internal API calls; this remains no spend cap.

`evaluation.selection_invocations` and `evaluation.holdout_invocations` add up to `evaluation.invocations`. Total usage, modeled evaluation cost, execution-cost estimates, net savings, and break-even include both phases. `evaluation.unique_subsets` retains the optimization-subset count; `holdout_unique_subsets` records validation subsets separately. Baseline/final inference projections still use only the optimization case mix; holdout validation is a one-time cost, not extra future inference traffic.

Existing pricing rules apply: Python serialized token proxies cannot reveal internal model usage; missing/partial reported usage or pricing keeps the relevant USD estimates unavailable. Mock zero execution cost remains separate from modeled paid-evaluation cost. Failure does not erase the cost of the validation calls. API transport failures still stop without automatic retries or a complete report; previous calls may have incurred charges.

## FAQ example

Selection remains 8/8 with 2/5 collections removed, across 104 invocations. The integrated holdout adds 4 invocations for **108 total**, observes full-context 2/2 versus selected-context 1/2, and returns `FAIL` at zero tolerance. Case `logo` needs the removed marketing collection. Its full-context answer is correct; with selected context the retriever returns the invoice article instead. The selected fields are not changed in response, and the exported gate flag is false.

See [the actual report](holdout-example-report.md). These are curated integration examples, not real-world LLM benchmarks or measured API savings. The existing synthetic mock examples remain selection-only and therefore `NOT_VALIDATED` unless an independent holdout is supplied.
