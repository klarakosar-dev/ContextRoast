# Reference

Command-line options, selection rules, scoring and cost accounting. See [the README](../README.md) for installation and examples.

## Dataset format

JSON and YAML share one schema. Every case supplies exactly the declared context fields; values may be any finite JSON value. `expected` is the desired structured answer, without a wrapper.

```yaml
version: 1
name: shipping-example
instruction: Calculate the shipping amount in cents from shipping_rates and currency.
mock_task: shipping-quote
context_fields: [shipping_rates, currency, marketing_copy]
cases:
  - id: gb-2kg
    input: Ship 2 kg to GB.
    context:
      shipping_rates:
        GB: {base_cents: 100, per_kg_cents: 50}
      currency: GBP
      marketing_copy: Our customers love quick deliveries.
    expected: {amount_cents: 200, currency: GBP}
```

`name`, `instruction`, `version`, `context_fields`, and a non-empty `cases` list are required. Each case requires unique `id`, string `input`, `context`, and `expected`. `mock_task` is needed only for mock runs. Unknown keys, duplicate JSON/YAML keys, undeclared/missing fields, non-finite numbers, and non-JSON YAML values are rejected. Quote YAML dates and ambiguous strings. See [the dataset guide](datasets.md).

## How removal works

1. Evaluate every case with all context fields.
2. Evaluate all single-field removals from the current subset.
3. Among candidates whose accuracy is at least **original baseline accuracy minus tolerance**, choose the highest accuracy, then the fewest estimated input tokens. Remaining ties follow the declared field order.
4. Accept that removal and repeat until no candidate qualifies or no fields remain.

`--tolerance 0.05` allows a drop of **5 percentage points**, not 5% relative. The tolerance always refers to the original baseline, preventing cumulative quality drift. A run-local cache evaluates each unique subset once; the chosen candidate becomes the final result without a redundant rerun. For `F` fields and `N` cases, the upper bound is `N × (1 + F(F+1)/2)` provider invocations. A Python invocation can make any number of internal API calls; the bound does not limit its spend.

Greedy elimination is order-dependent and **not globally optimal**. Interactions can prevent useful removals or make multiple subsets equally valid. This is a search on the supplied cases, not proof that a field is universally unnecessary. A small evaluation does not establish statistical certainty. Review failed cases and validate the selected fields on a separate, representative holdout set before deployment. A zero-accuracy baseline can technically lose every field while remaining useless; the report flags this.

## Scoring

Built-in providers use a JSON object with exactly one key: `{"answer": <structured value>}`. Python callables return the prediction directly and the runner handles wrapping. Expected answers stay local to the scorer and are never included in requests. Object key order and insignificant whitespace do not matter. Array order, string case/whitespace, extra keys, and JSON value types do matter; `true`, `1`, and `1.0` are distinct. There is no fuzzy matching or LLM judge. Invalid outputs score as incorrect and retain valid reported usage. Python exceptions count as failed invocations without automatic retries; HTTP/transport failures in the built-in API stop the run.

## Tokens, costs, and break-even

Reports keep these separate:

- **Estimated tokens:** `ceil(UTF-8 bytes / 4)` plus approximate framing. For Python these are serialized input/output proxies, not the internal model prompts. Input counts include instructions, field names, values, and user input.
- **Provider-reported tokens:** the API's `prompt_tokens` / `completion_tokens`, or the Python `AgentResult` aggregate pair. API usage falls back to estimates if either count is missing/invalid. Invalid Python usage is a failed case; omitted usage remains unavailable for monetary calculations.
- **Costing tokens:** the mock/API use reported usage where available, otherwise estimates. Python monetary estimates require reported usage for every invocation in the relevant subset/search, because serialized inputs cannot reveal internal calls.

Both `--input-price` and `--output-price` are required for money estimates. A missing price means **unknown**, never free. Explicit `0` is supported. No built-in model prices go stale.

```text
cost = (input_tokens × input_price + output_tokens × output_price) / 1,000,000
savings_per_request = (baseline_case_set_cost − final_case_set_cost) / number_of_cases
gross_savings = savings_per_request × projected_requests
net_savings = gross_savings − total_evaluation_execution_cost
break_even_requests = ceil(total_evaluation_execution_cost / savings_per_request)
modeled_paid_break_even = ceil(modeled_evaluation_token_cost / savings_per_request)
```

Evaluation totals include the baseline and **all evaluated candidates, including rejected ones**. Baseline and final inference costs describe one pass over the cases. Projections assume future requests resemble that case mix. Output tokens can increase, and net savings can be negative. Break-even is unavailable if prices are missing or per-request savings are not positive.

With `--holdout`, totals also include full-context and selected-context holdout evaluations. If the subsets are identical, that holdout subset is evaluated once and reused. Python evaluation/net costs remain unknown if any selection or validation invocation has unreported usage. A failed gate does not turn modeled savings into a deployment recommendation.

Mock execution costs exactly $0. The report shows **modeled paid-evaluation break-even separately** from execution-cost break-even. At illustrative input/output prices of USD 2 / USD 8 per million tokens, the support simulation has a modeled search token cost of USD 0.067098 and a modeled paid break-even of **189 requests**, versus 0 for free mock execution. Its modeled net savings after paid evaluation are USD 3.492902 across 10,000 projected requests. These are hypothetical, not a real LLM benchmark.

Python execution is never labelled free by default. Reported token counts are observations supplied by the application; USD amounts calculated from them and prices are estimates, not measured billing. Missing/partial usage leaves evaluation/net costs unavailable. Import-time charges, unreported calls, cached discounts, tool/hosting/CPU costs, heterogeneous model rates, and provider-specific billing are outside the model. See `cost_information`, `evaluation.invocations`, and `evaluation.underlying_api_calls` in the JSON report.

## Optional API calls

The adapter uses the OpenAI-compatible **Chat Completions** endpoint. Install the optional dependency and set a key outside the dataset:

```bash
python -m pip install -e '.[api]'
export OPENAI_API_KEY='your-key'
contextroast run your-dataset.yaml --provider api --allow-paid \
  --model YOUR_MODEL_ID \
  --input-price YOUR_INPUT_PRICE --output-price YOUR_OUTPUT_PRICE \
  --max-output-tokens 512 --output-dir reports/api
```

Replace all uppercase placeholders before running. This command can incur charges. Both `--provider api` and `--allow-paid` are required, even when a key is present. Merely installing the extra, importing the package, validating data, or running the mock does not call an API. The adapter also requires explicit opt-in when used as a Python class.

Use `--base-url https://your-provider.example/v1` and `--api-key-env YOUR_KEY_VARIABLE` for another compatible endpoint. HTTPS is required except for localhost HTTP. The request sends `response_format: {"type":"json_object"}` and `max_completion_tokens`; your endpoint/model must support those parameters. JSON mode is used for compatibility; it does not guarantee schema adherence. See [official OpenAI documentation](https://developers.openai.com/api/docs/guides/structured-outputs#json-mode). There are no silent fallbacks, implicit model choices, or automatic retries. Timeout defaults to 30 seconds per request.

Before the first API request, the CLI prints the upper bound on request count. This is **not a dollar budget cap**: input sizes, usage, and pricing vary. On HTTP/transport failure the command stops with exit code 2; previous calls may have incurred charges and no complete report is written. Completed successful runs account for all their evaluation calls.

API runs send the instruction, case input, and selected context to your chosen endpoint. They never send expected answers or case IDs. JSON reports include predictions and case IDs, which may be sensitive. Keys are read from the environment and excluded from reports and error messages. Keep real data, keys, and generated reports out of commits.

## CLI reference

```text
contextroast --version
contextroast validate DATASET
contextroast run DATASET [OPTIONS]

--provider [mock|api|python] default: mock
--entrypoint MODULE:FN      trusted Python callable; required for python
--export-context           optional recommended schema and standalone removal trace
--tolerance FLOAT           absolute accuracy drop, 0..1; default: 0
--holdout PATH              separate untouched JSON/YAML dataset; optional
--holdout-tolerance FLOAT   absolute drop on holdout, 0..1; default: 0
--input-price DECIMAL       USD / million input tokens; default: unknown
--output-price DECIMAL      USD / million output tokens; default: unknown
--projected-requests INT    default: 10000
--output-dir PATH           default: reports
--allow-paid                consent for API calls or trusted Python imports/execution
--model TEXT                required for API
--base-url URL              default: https://api.openai.com/v1
--api-key-env NAME          default: OPENAI_API_KEY
--timeout FLOAT             seconds per API request; default: 30
--max-output-tokens INT     max_completion_tokens; default: 512
```

Run `contextroast run --help` for full help. Success or no holdout exits 0; a failed holdout gate exits 1 after writing reports; configuration, data, and provider failures exit 2.

## Development and release

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest --cov=contextroast --cov-branch --cov-fail-under=95 --cov-report=term-missing
python -m build
python -m twine check dist/*.whl dist/*.tar.gz
```

Tests are entirely offline, including fake-transport API tests, a mock-backed Python integration, and tests proving holdout isolation. CI is configured to test Python 3.11–3.14, enforce 95% combined statement/branch coverage, run examples and assert the expected holdout failure, build/check packages, and install the wheel in a clean environment. See [CONTRIBUTING.md](../CONTRIBUTING.md), [release notes](../CHANGELOG.md), and the [release checklist](releasing.md). The manual Draft release workflow creates prerelease drafts only; publication is a separate GitHub action. No PyPI upload is configured.

## License

[MIT](../LICENSE).
