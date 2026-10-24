# Dataset guide

Version 1 datasets are UTF-8 JSON (`.json`) or YAML (`.yaml` / `.yml`). See [support.json](../examples/support.json) and [shipping.yaml](../examples/shipping.yaml) for complete examples.

Python mode uses the same schema, including `expected`, with no `mock_task` needed. See [retrieval.json](../examples/retrieval.json) and the [Python integration guide](python-agents.md). The callable receives a public case view without `expected` and a deep copy of the selected context fields.

| Key | Required | Meaning |
| --- | --- | --- |
| `version` | yes | Integer `1` |
| `name` | yes | Non-empty dataset name |
| `instruction` | yes | Task instruction used in every model request |
| `context_fields` | yes | Ordered list of unique, non-empty names; empty is allowed |
| `cases` | yes | Non-empty list of cases |
| `mock_task` | mock only | `support-routing` or `shipping-quote` |

Every case has exactly `id`, `input`, `context`, and `expected`. IDs are unique non-empty strings, and inputs are non-empty strings. Context must contain exactly the declared fields, even when a value is `null`. Missing a field is different from evaluating its removal: ContextRoast controls removal itself. If there is no context, declare `context_fields: []` and use `context: {}`.

Context values and expected answers accept JSON objects, arrays, strings, integers, finite floating-point numbers, booleans, and null. Object keys must be strings. YAML timestamps, sets, binary values, non-finite numbers, and recursive values are not supported. Avoid YAML merge keys/anchors; write explicit values. Duplicate keys are errors. Unknown top-level and case keys are errors to catch misspellings.

Each model request contains the instruction (plus the JSON-wrapper instruction), case input, and selected fields. It excludes the expected answer and case ID. The scorer expects `{"answer": <value>}`, while dataset `expected` stores just `<value>`. To return an object that itself has a key called `answer`, wrap it again at the provider boundary.

The deterministic mock is a simulation, not a general language model. It ignores natural-language instructions and implements only the following rules; arbitrary tasks need the opt-in API adapter.

## Support routing

Set `mock_task: support-routing`. Required context:

```yaml
routing_policy:
  routes:
    - queue: billing
      keywords: [invoice, refund]
    - queue: technical
      keywords: [login, error]
  default_queue: general
  priority_tiers: [enterprise]
account: {tier: enterprise}
```

The first route with any case-insensitive keyword substring in the input wins. If none matches, the default queue wins. Priority is `high` for a listed account tier, otherwise `normal`. The answer is `{"queue": "billing", "priority": "high"}`, for example. Removing either required field returns a synthetic missing-context error and fails these example cases. Other context fields are ignored.

## Shipping quote

Set `mock_task: shipping-quote`. Input must match `Ship INTEGER kg to CC.` exactly, with a positive integer weight and a two-letter uppercase country code. Required context:

```yaml
shipping_rates:
  GB: {base_cents: 100, per_kg_cents: 50}
currency: GBP
```

The amount is `base_cents + weight × per_kg_cents`, returned as `{"amount_cents": 200, "currency": "GBP"}` for 2 kg to GB. Other context fields are ignored.

## Designing useful real evaluations

Choose representative cases with ground-truth structured answers, including rare cases where a field matters. Remove direct answer leakage from inputs and context. A field that appears unnecessary on six cases may matter on the next case. Reserve separate holdout cases before tuning context, and evaluate the final selected subset on them through your application or the Python `evaluate` function. Use `--holdout` to check a separately prepared dataset after selection. ContextRoast does not create the split, run significance tests or change a deployment.
