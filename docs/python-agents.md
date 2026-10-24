# Python agents

ContextRoast can evaluate a trusted Python callable through the same dataset loader, exact-match scorer, greedy backward elimination, and reports as the mock/API providers.

Add `--holdout` to validate the selected fields on separate cases. The quickstart below performs selection only; its report says `NOT_VALIDATED`. See [holdout checks](holdout.md).

## Quickstart

Run from a checkout or extracted source distribution with Python 3.11+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
contextroast validate examples/retrieval.json
contextroast run examples/retrieval.json \
  --provider python --entrypoint examples.retrieval_agent:predict \
  --allow-paid --export-context --output-dir reports/retrieval
```

**Read the entrypoint before consenting.** `--allow-paid` authorizes trusted Python code execution, including module imports, which may perform network calls, incur real API charges, access files, print data, or make other changes. ContextRoast does not sandbox user code. The bundled retriever has been inspected and makes no network calls; the Python provider never assumes arbitrary local code is free.

The CLI prints the warning before import. Without explicit consent it does not import the entrypoint. Only use code and dependencies you trust. ContextRoast cannot enforce a budget inside arbitrary Python code; the invocation bound is not a bound on internal API calls or spend. `--timeout` and `--max-output-tokens` configure the built-in API adapter only; your function must implement its own timeouts, output limits, and internal retry policy.

Actual stdout (the execution/charges warning is also printed to stderr):

```text
Valid: curated-faq-retrieval — 8 cases, 5 context fields
Accuracy: 100.00% -> 100.00%; removed 2/5 fields; serialized input-proxy reduction: 29.14%
Reports: reports/retrieval/report.json and reports/retrieval/report.md
```

Choose a fresh output directory for every run. The extra export files are `recommended-context.schema.json` and `removal-trace.json`.

## Function contract

An importable module contains a synchronous callable that accepts two positional arguments:

```python
from contextroast.python_provider import AgentCase


def predict(case: AgentCase, context: dict):
    # Invoke your application's existing pipeline here.
    return {"result": context.get("setting")}
```

Use `--entrypoint your_module:predict`. The current working directory is added to the import search path, so checkout-local modules work from the console script. Module paths such as `agents.customer:predict` work; filesystem paths, expressions, and nested attribute expressions are not accepted. Install any application dependencies in the same environment as ContextRoast. Loading/import/signature errors stop the command with exit code 2 and no complete report; import side effects may already have happened after consent.

`AgentCase` has immutable `id`, `input`, and `instruction` attributes. It deliberately has no expected answer or full original context. Unlike the remote API adapter, the Python function receives the case ID for application diagnostics; do not use answer-revealing IDs. `context` is a deep copy of the current candidate fields and their values. Mutating it does not alter later trials or the original dataset. Expected answers stay in the local scorer.

Return the **prediction itself** as a finite JSON-compatible value: object, array, string, integer, float, boolean, or null. Do not add ContextRoast's API `answer` wrapper. A dictionary containing an `answer` key is treated as a normal prediction dictionary. Parse JSON text into a Python object if the expected answer is an object. Object order and JSON formatting do not affect scoring; array order, string case/whitespace, extra keys and types do.

Async functions are unsupported; wrap your existing async pipeline in a synchronous function if necessary. Exceptions derived from `Exception` produce an incorrect case with `python_exception:TYPE`; exception messages are omitted to reduce accidental secret exposure. Invalid predictions or usage are explicit incorrect cases. The search continues, counts each attempted invocation once, and makes no automatic retries. User code's own printed logs are not redacted. `KeyboardInterrupt` and `SystemExit` propagate. A zero-accuracy baseline is flagged and does not establish useful removability.

Functions should produce repeatable results and avoid persistent state between calls. ContextRoast imports a module once and calls its function for each case/subset. It does not restart the application between invocations. Stochasticity, internal caches, and external state can affect comparisons; use a separate holdout and repeated evaluations as appropriate.

## Optional token usage

Return an `AgentResult` to attach usage:

```python
from contextroast.python_provider import AgentResult


def predict(case, context):
    result = run_your_agent(case.input, context)  # your application code
    return AgentResult(
        prediction=result.prediction,
        input_tokens=result.total_input_tokens,
        output_tokens=result.total_output_tokens,
    )
```

Both usage counts must be non-negative Python integers, or both must be `None`. Booleans, partial pairs, and negative counts are invalid. Counts must aggregate **all model calls within that invocation**, including application retries and failed model outputs whose usage is known. ContextRoast cannot independently verify your instrumentation. If a function raises after a paid call and cannot return usage, that invocation's spend is unknown.

If a result's prediction is invalid but its usage pair is valid, that usage still enters evaluation totals. If usage is invalid, the case fails and usage remains unavailable.

## Measurements, proxies, and money

- Correct/total cases, selected fields, trace decisions, and callable invocation counts are observations from the actual run.
- `reported_*_tokens` are the aggregate counts returned by the application. ContextRoast does not measure an agent's internal prompts directly.
- Python `estimated_*_tokens` describe serialized case/context and prediction sizes using the existing heuristic. They are **input/output proxies**, not actual prompt/output usage. They are used for token-size tie-breaking, not to invent dollar costs for hidden application calls.
- Monetary estimates for a Python subset require a valid reported usage pair for every invocation in that subset and both USD token prices. The entire search cost additionally requires coverage of every evaluated invocation, including rejected trials. Partial coverage leaves evaluation/net costs unavailable even when baseline/final projections can be estimated.
- Prices turn reported usage into estimates, not billing receipts. Import-time charges, unreported calls, tool fees, hosting/CPU costs, and heterogeneous model prices are not observed. Aggregate counters with one price pair are appropriate only when the same flat rates apply across calls. Measured billing is unavailable for Python; local execution is never labelled free by default.

`evaluation.invocations` counts callable attempts. `evaluation.underlying_api_calls` is `null` for Python because one attempt can make any number of API calls. `cost_information` records the distinction explicitly. Existing dataset schema version 1 and report keys remain compatible; the new report fields are additive.

For priced scenarios, add `--input-price` and `--output-price`. Missing usage or prices remain unknown. The built-in mock's known zero execution charge is shown separately from **modeled paid-evaluation break-even**, which includes the modeled cost of the entire search. The old `projection.break_even_requests` key remains an execution-cost-based compatibility alias; use the explicit `execution_break_even_requests` or `modeled_paid_evaluation_break_even_requests` keys in new tools.

## Recommended fields and trace

`--export-context` writes a JSON Schema requiring exactly the selected field names. Values are deliberately unconstrained; the schema does not infer application-specific value types, validate your agent's semantics, or establish that excluded fields are unnecessary on unseen requests. The standalone removal trace includes candidate fields, score changes against the original baseline, tolerance checks, and chosen removals. It excludes context values and predictions. Full per-case predictions remain in `report.json` and may be sensitive.

## What this demonstration establishes

The example function is a generic FAQ retriever using word-count cosine similarity. It searches every supplied collection and never branches on collection names or reads gold labels. The actual greedy algorithm invokes it on 13 subsets and 8 labelled cases (104 invocations), keeps 8/8 correct, and removes `legacy_faq` and `marketing_faq`. Renaming all collections leaves behavior intact; adding a logo question makes the previously removable marketing collection necessary. Tests verify both facts.

The labels and FAQ corpus are curated example data. On a separate two-case holdout, the full context scores 2/2 but the selected fields score **1/2** because the logo question requires the removed marketing collection. This is a measured integration result on a small curated dataset, not a real-world LLM benchmark or evidence of API savings. See [the benchmark record](benchmarks.md) for both successful and regressed cases.
