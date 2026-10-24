"""Auditable reports with explicit unknown prices and token provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import ROUND_CEILING, Decimal, DecimalException, InvalidOperation
from typing import Any

from contextroast import __version__
from contextroast.engine import Evaluation, Run, TokenTotals, max_evaluations
from contextroast.holdout import HOLDOUT_WARNING, HoldoutValidation


@dataclass(frozen=True)
class Pricing:
    input_per_million: Decimal | None = None
    output_per_million: Decimal | None = None

    def __post_init__(self) -> None:
        for value in (self.input_per_million, self.output_per_million):
            if value is not None and (not value.is_finite() or value < 0):
                raise ValueError("token prices must be finite and non-negative")
            if value is not None:
                try:
                    # Reject finite values outside the Decimal arithmetic range before a run.
                    _ = +value
                except DecimalException as exc:
                    raise ValueError("token price is outside the supported decimal range") from exc

    @classmethod
    def parse(cls, input_price: str | None, output_price: str | None) -> Pricing:
        try:
            return cls(
                Decimal(input_price) if input_price is not None else None,
                Decimal(output_price) if output_price is not None else None,
            )
        except InvalidOperation as exc:
            raise ValueError("token prices must be finite and non-negative numbers") from exc

    @property
    def complete(self) -> bool:
        return self.input_per_million is not None and self.output_per_million is not None

    def cost(self, usage: TokenTotals) -> Decimal | None:
        if not self.complete:
            return None
        try:
            return (
                Decimal(usage.costing_input_tokens) * self.input_per_million
                + Decimal(usage.costing_output_tokens) * self.output_per_million
            ) / Decimal(1_000_000)
        except DecimalException as exc:
            raise ValueError("cost calculation is outside the supported decimal range") from exc


def _money(value: Decimal | None) -> str | None:
    if value is None:
        return None
    # Preserve decimal strings without expanding extreme exponents into huge reports.
    return str(value) if abs(value.as_tuple().exponent) > 100 else format(value, "f")


def _usage(usage: TokenTotals) -> dict[str, Any]:
    return {**asdict(usage), "costing_source": usage.costing_source}


def _cost(usage: TokenTotals, pricing: Pricing, provider: str) -> Decimal | None:
    # Serialized case/context size cannot reveal a Python function's internal model prompts.
    if provider == "python" and usage.reported_calls != usage.calls:
        return None
    return pricing.cost(usage)


def _evaluation(evaluation: Evaluation, pricing: Pricing, provider: str) -> dict[str, Any]:
    return {
        "id": evaluation.id,
        "fields": list(evaluation.fields),
        "correct": evaluation.correct,
        "cases": len(evaluation.outcomes),
        "accuracy": evaluation.accuracy,
        "usage": _usage(evaluation.usage),
        "inference_cost_usd": _money(_cost(evaluation.usage, pricing, provider)),
        "outcomes": [asdict(outcome) for outcome in evaluation.outcomes],
    }


def _holdout_report(
    validation: HoldoutValidation | None, pricing: Pricing, provider: str
) -> dict[str, Any]:
    result = {
        "status": "NOT_VALIDATED",
        "dataset": None,
        "tolerance": None,
        "baseline": None,
        "final": None,
        "accuracy_delta": None,
        "minimum_accuracy": None,
        "regressed_cases": [],
        "improved_case_ids": [],
        "usage": _usage(TokenTotals()),
        "unique_subsets": 0,
        "token_cost_usd": None,
        "warning": HOLDOUT_WARNING,
    }
    if validation is None:
        return result
    baseline, final = validation.baseline, validation.final
    regressions, improvements = [], []
    for case, before, after in zip(
        validation.dataset.cases, baseline.outcomes, final.outcomes, strict=True
    ):
        if before.correct and not after.correct:
            regressions.append(
                {
                    "case_id": case.id,
                    "input": case.input,
                    "expected": case.expected,
                    "baseline_prediction": before.predicted,
                    "final_prediction": after.predicted,
                    "baseline_error": before.error,
                    "final_error": after.error,
                }
            )
        elif not before.correct and after.correct:
            improvements.append(case.id)
    result.update(
        {
            "status": validation.status,
            "dataset": validation.dataset.name,
            "tolerance": validation.tolerance,
            "baseline": _evaluation(baseline, pricing, provider),
            "final": _evaluation(final, pricing, provider),
            "accuracy_delta": final.accuracy - baseline.accuracy,
            "minimum_accuracy": max(0, baseline.accuracy - validation.tolerance),
            "regressed_cases": regressions,
            "improved_case_ids": improvements,
            "usage": _usage(validation.usage),
            "unique_subsets": 1 if baseline is final else 2,
            "token_cost_usd": _money(_cost(validation.usage, pricing, provider)),
        }
    )
    return result


def make_report(
    run: Run,
    pricing: Pricing,
    projected_requests: int = 10_000,
    *,
    holdout: HoldoutValidation | None = None,
) -> dict[str, Any]:
    if type(projected_requests) is not int or projected_requests < 0:
        raise ValueError("projected requests must be a non-negative integer")
    baseline_cost = _cost(run.baseline.usage, pricing, run.provider)
    final_cost = _cost(run.final.usage, pricing, run.provider)
    total_usage = TokenTotals()
    total_usage.add(run.usage)
    if holdout is not None:
        total_usage.add(holdout.usage)
    held = _holdout_report(holdout, pricing, run.provider)
    deployment_ready = (
        holdout is not None
        and held["status"] == "PASS"
        and run.final.correct > 0
        and holdout.final.correct > 0
    )
    token_cost = _cost(total_usage, pricing, run.provider)
    execution_cost = Decimal(0) if run.provider == "mock" else token_cost
    per_request = (
        (baseline_cost - final_cost) / len(run.dataset.cases)
        if baseline_cost is not None and final_cost is not None
        else None
    )
    gross = per_request * projected_requests if per_request is not None else None
    net = gross - execution_cost if gross is not None and execution_cost is not None else None
    break_even = None
    if per_request is not None and per_request > 0 and execution_cost is not None:
        break_even = int((execution_cost / per_request).to_integral_value(rounding=ROUND_CEILING))
    modeled_break_even = None
    if per_request is not None and per_request > 0 and token_cost is not None:
        modeled_break_even = int(
            (token_cost / per_request).to_integral_value(rounding=ROUND_CEILING)
        )
    modeled_net = gross - token_cost if gross is not None and token_cost is not None else None
    baseline_input = run.baseline.usage.estimated_input_tokens
    input_saved = baseline_input - run.final.usage.estimated_input_tokens
    warnings = [
        "Greedy backward elimination is order-dependent and is not globally optimal.",
        "Accuracy is measured on this evaluation set; it does not establish statistical certainty "
        "or performance on unseen data. Validate on a separate holdout set before deployment.",
        "Estimated tokens use ceil(UTF-8 bytes / 4) plus approximate message framing, not a "
        "model tokenizer. Estimates can be inaccurate, especially for non-English text.",
        "Costs use flat user-supplied USD prices. Discounts, cached-input rates, tools, taxes, "
        "and other billing adjustments are not modeled. Output length can change after removal.",
    ]
    warnings.append(HOLDOUT_WARNING)
    if held["status"] == "NOT_VALIDATED":
        warnings.append(
            "No holdout supplied: NOT_VALIDATED. Exported context is not deployment-ready."
        )
    elif held["status"] == "FAIL":
        warnings.append(
            "Holdout validation FAIL: exported context is not deployment-ready. "
            "Savings projections do not override the observed accuracy regression."
        )
    elif not deployment_ready:
        warnings.append(
            "Holdout gate passed but observed zero correctness blocks deployment-ready labeling."
        )
    if run.provider == "mock":
        warnings.append(
            "Offline synthetic simulation: execution is free. Priced inference and evaluation "
            "token costs are hypothetical; these results do not measure a real LLM's savings."
        )
    if run.provider == "python":
        warnings.append(
            "Python code may execute multiple paid calls per invocation, including at import. "
            "Only returned aggregate token usage is observed; import-time charges, unreported "
            "calls, and non-token costs are unavailable. Execution must never be assumed free. "
            "All monetary amounts are pricing estimates, not measured billing."
        )
        warnings.append(
            "For Python, estimated tokens are serialized input/output proxies, not measured "
            "agent prompts. Monetary projections require complete reported usage and both prices."
        )
    if not pricing.complete:
        warnings.append(
            "Input and output prices are both required for monetary estimates. Missing pricing "
            "is unknown, never zero. Token changes remain available."
        )
    if run.baseline.accuracy == 0:
        warnings.append(
            "Baseline accuracy is zero; accepted removals do not demonstrate useful quality."
        )
    if total_usage.costing_source == "mixed":
        warnings.append(
            "Python usage coverage is partial; evaluation and net cost estimates are unavailable."
            if run.provider == "python"
            else "Cost calculations mix provider-reported usage and estimated usage."
        )
    return {
        "schema_version": 1,
        "contextroast_version": __version__,
        "dataset": run.dataset.name,
        "provider": run.provider,
        "model": run.model,
        "holdout": held,
        "deployment_ready": deployment_ready,
        "readiness_scope": "configured holdout gate with nonzero observed correctness; "
        "not a generalization or production-safety guarantee",
        "algorithm": "greedy_backward_elimination",
        "accuracy_tolerance": run.tolerance,
        "minimum_accuracy": max(0, run.baseline.accuracy - run.tolerance),
        "kept_fields": list(run.final.fields),
        "removed_fields": [
            name for name in run.dataset.context_fields if name not in run.final.fields
        ],
        "token_estimate_basis": "serialized_case_context_proxy"
        if run.provider == "python"
        else "approximate_chat_messages",
        "cost_information": {
            "usage_source": total_usage.costing_source,
            "monetary_source": "estimated_from_reported_usage"
            if token_cost is not None and total_usage.costing_source == "provider_reported"
            else "estimated_from_tokens"
            if token_cost is not None
            else "unavailable",
            "measured_billing_cost_usd": "0" if run.provider == "mock" else None,
            "scope": "returned token usage only; excludes import-time and unreported charges"
            if run.provider == "python"
            else "flat token pricing model",
        },
        "baseline": _evaluation(run.baseline, pricing, run.provider),
        "final": _evaluation(run.final, pricing, run.provider),
        "changes": {
            "accuracy_delta": run.final.accuracy - run.baseline.accuracy,
            "estimated_input_tokens_saved": input_saved,
            "estimated_input_reduction_fraction": input_saved / baseline_input
            if baseline_input
            else 0,
            "estimated_output_tokens_saved": (
                run.baseline.usage.estimated_output_tokens - run.final.usage.estimated_output_tokens
            ),
            "costing_input_tokens_saved": (
                run.baseline.usage.costing_input_tokens - run.final.usage.costing_input_tokens
            ),
            "costing_output_tokens_saved": (
                run.baseline.usage.costing_output_tokens - run.final.usage.costing_output_tokens
            ),
        },
        "evaluation": {
            "unique_subsets": len(run.evaluations),
            "holdout_unique_subsets": held["unique_subsets"],
            "max_possible_subsets_evaluated": max_evaluations(len(run.dataset.context_fields))
            + (2 if holdout is not None else 0),
            "usage": _usage(total_usage),
            "invocations": total_usage.calls,
            "selection_invocations": run.usage.calls,
            "holdout_invocations": holdout.usage.calls if holdout is not None else 0,
            "invocation_unit": "python_callable_invocations"
            if run.provider == "python"
            else "provider_requests",
            "underlying_api_calls": None
            if run.provider == "python"
            else total_usage.calls
            if run.provider == "api"
            else 0,
            "token_cost_usd": _money(token_cost),
            "execution_cost_usd": _money(execution_cost),
            "execution_cost_basis": "offline_free"
            if run.provider == "mock"
            else "unavailable"
            if execution_cost is None
            else total_usage.costing_source,
            "includes": "optimization baseline and all trials, including rejected removals, "
            "plus any holdout baseline/final validation calls",
        },
        "pricing": {
            "currency": "USD",
            "input_per_million_tokens": _money(pricing.input_per_million),
            "output_per_million_tokens": _money(pricing.output_per_million),
            "complete": pricing.complete,
        },
        "projection": {
            "requests": projected_requests,
            "savings_per_request_usd": _money(per_request),
            "gross_savings_usd": _money(gross),
            "net_savings_after_evaluation_usd": _money(net),
            "break_even_requests": break_even,
            "execution_break_even_requests": break_even,
            "modeled_paid_evaluation_break_even_requests": modeled_break_even,
            "net_savings_after_modeled_paid_evaluation_usd": _money(modeled_net),
            "assumption": "Future requests have the same case mix and token usage as this dataset. "
            "Only accepted-subset inference is projected; "
            "the search and any holdout check run once.",
        },
        "trials": [asdict(trial) for trial in run.trials],
        "evaluations": [
            _evaluation(evaluation, pricing, run.provider) for evaluation in run.evaluations
        ],
        "warnings": warnings,
    }


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _usd(value: str | None) -> str:
    return "unknown" if value is None else f"${Decimal(value):,.8f}"


def render_markdown(report: dict[str, Any]) -> str:
    baseline, final = report["baseline"], report["final"]
    old, new = baseline["usage"], final["usage"]
    evaluation, projection = report["evaluation"], report["projection"]
    held = report["holdout"]
    break_even = projection["break_even_requests"]
    break_even_text = (
        str(break_even)
        if break_even is not None
        else "unavailable (unknown prices or no positive savings)"
    )
    modeled_break_even = projection["modeled_paid_evaluation_break_even_requests"]
    modeled_break_even_text = (
        str(modeled_break_even)
        if modeled_break_even is not None
        else "unavailable (unknown usage/prices or no positive savings)"
    )
    input_label = (
        "Estimated input tokens (serialized proxy)"
        if report["provider"] == "python"
        else "Estimated input tokens"
    )
    output_label = (
        "Estimated output tokens (serialized proxy)"
        if report["provider"] == "python"
        else "Estimated output tokens"
    )
    reduction_label = (
        "Serialized input-proxy reduction"
        if report["provider"] == "python"
        else "Estimated input-token reduction"
    )
    lines = [
        "# ContextRoast report",
        "",
        f"Dataset: {_cell(report['dataset'])} · Provider: {_cell(report['provider'])} "
        f"· Model: {_cell(report['model'])}",
        "",
        *(
            ["**Offline synthetic simulation. Priced inference savings are hypothetical.**", ""]
            if report["provider"] == "mock"
            else []
        ),
        f"**Holdout validation: {held['status']}.** "
        "Deployment-ready gate: "
        f"{'passed (limited scope)' if report['deployment_ready'] else 'blocked'}.",
        "",
        *(
            [
                "**Python callable execution: local code can incur real API charges. Serialized "
                "token proxies do not measure internal model prompts or billing.**",
                "",
            ]
            if report["provider"] == "python"
            else []
        ),
        "Kept fields: " + (", ".join(_cell(name) for name in report["kept_fields"]) or "(none)"),
        "",
        "Removed fields: "
        + (", ".join(_cell(name) for name in report["removed_fields"]) or "(none)"),
        "",
        f"Allowed absolute accuracy drop: {report['accuracy_tolerance']:.2%} "
        "(always compared with the original baseline).",
        "",
        "| Metric (entire evaluation set) | Baseline | Final |",
        "| --- | ---: | ---: |",
        f"| Accuracy | {baseline['accuracy']:.2%} ({baseline['correct']}/{baseline['cases']}) | "
        f"{final['accuracy']:.2%} ({final['correct']}/{final['cases']}) |",
        f"| {input_label} | {old['estimated_input_tokens']:,} | "
        f"{new['estimated_input_tokens']:,} |",
        f"| {output_label} | {old['estimated_output_tokens']:,} | "
        f"{new['estimated_output_tokens']:,} |",
        f"| Provider-reported calls | {old['reported_calls']} / {old['calls']} | "
        f"{new['reported_calls']} / {new['calls']} |",
        f"| Reported input tokens (known calls only) | {old['reported_input_tokens']:,} | "
        f"{new['reported_input_tokens']:,} |",
        f"| Reported output tokens (known calls only) | {old['reported_output_tokens']:,} | "
        f"{new['reported_output_tokens']:,} |",
        f"| Costing source | {old['costing_source']} | {new['costing_source']} |",
        f"| Modeled inference cost | {_usd(baseline['inference_cost_usd'])} | "
        f"{_usd(final['inference_cost_usd'])} |",
        "",
        f"Accuracy change: {report['changes']['accuracy_delta'] * 100:+.2f} percentage points. "
        f"{reduction_label}: "
        f"{report['changes']['estimated_input_reduction_fraction']:.2%}.",
        "",
        "## Evaluation and savings",
        "",
        f"- Monetary information: {report['cost_information']['monetary_source']}; "
        f"measured billing: {_usd(report['cost_information']['measured_billing_cost_usd'])}.",
        "- USD prices per million tokens: "
        f"input {_usd(report['pricing']['input_per_million_tokens'])}, "
        f"output {_usd(report['pricing']['output_per_million_tokens'])}.",
        f"- Selection search: {evaluation['unique_subsets']} unique subsets, "
        f"{evaluation['selection_invocations']} {evaluation['invocation_unit']}; "
        "includes baseline and rejected trials.",
        f"- Holdout invocations: {evaluation['holdout_invocations']}; "
        f"total evaluation invocations: {evaluation['invocations']}.",
        f"- Total evaluation usage: {evaluation['usage']['estimated_input_tokens']:,} "
        "estimated input / "
        f"{evaluation['usage']['estimated_output_tokens']:,} estimated output tokens; "
        f"{evaluation['usage']['reported_calls']} calls with provider-reported usage.",
        f"- Modeled evaluation token cost: {_usd(evaluation['token_cost_usd'])}.",
        f"- Evaluation execution cost: {_usd(evaluation['execution_cost_usd'])} "
        f"({evaluation['execution_cost_basis']}).",
        f"- Projected requests: {projection['requests']:,}.",
        f"- Estimated gross inference savings: {_usd(projection['gross_savings_usd'])}.",
        "- Estimated net savings after execution cost: "
        f"{_usd(projection['net_savings_after_evaluation_usd'])}.",
        "- Estimated net savings after modeled paid evaluation: "
        f"{_usd(projection['net_savings_after_modeled_paid_evaluation_usd'])}.",
        f"- Modeled paid-evaluation break-even requests: {modeled_break_even_text}.",
        f"- Execution-cost break-even requests: {break_even_text} "
        f"(basis: {evaluation['execution_cost_basis']}).",
        "",
        projection["assumption"],
        "",
        "## Removal trials",
        "",
        "| Round | Field removed | Accuracy | Within tolerance | Chosen |",
        "| ---: | --- | ---: | --- | --- |",
    ]
    by_id = {item["id"]: item for item in report["evaluations"]}
    for trial in report["trials"]:
        accuracy = by_id[trial["evaluation_id"]]["accuracy"]
        lines.append(
            f"| {trial['round']} | {_cell(trial['removed_field'])} | {accuracy:.2%} | "
            f"{'yes' if trial['within_tolerance'] else 'no'} | "
            f"{'yes' if trial['accepted'] else 'no'} |"
        )
    lines.extend(["", "## Holdout validation", "", f"Status: **{held['status']}**.", ""])
    if held["status"] == "NOT_VALIDATED":
        lines.append("No holdout was supplied. Exported context is not deployment-ready.")
    else:
        before, after = held["baseline"], held["final"]
        lines.extend(
            [
                f"Dataset: {_cell(held['dataset'])}. "
                "Field selection used only the optimization dataset.",
                "",
                f"Holdout accuracy: {before['accuracy']:.2%} "
                f"({before['correct']}/{before['cases']}) "
                f"-> {after['accuracy']:.2%} ({after['correct']}/{after['cases']}); "
                f"change {held['accuracy_delta'] * 100:+.2f} percentage points; "
                f"allowed drop {held['tolerance'] * 100:.2f} percentage points.",
                "",
                f"Holdout calls: {held['usage']['calls']} "
                f"across {held['unique_subsets']} subset(s); "
                f"modeled token cost: {_usd(held['token_cost_usd'])}, "
                "included in total evaluation cost.",
                "",
            ]
        )
        if held["regressed_cases"]:
            lines.extend(
                [
                    "Cases correct with full context and incorrect with selected context:",
                    "",
                    "| Case | Input | Expected | Full-context prediction | "
                    "Selected-context prediction | Error |",
                    "| --- | --- | --- | --- | --- | --- |",
                ]
            )
            for case in held["regressed_cases"]:
                lines.append(
                    f"| {_cell(case['case_id'])} | {_cell(case['input'])} | "
                    f"{_cell(case['expected'])} | {_cell(case['baseline_prediction'])} | "
                    f"{_cell(case['final_prediction'])} | "
                    f"{_cell(case['final_error'] or 'exact-match failure')} |"
                )
        else:
            lines.append("No holdout cases changed from correct to incorrect.")
        if held["improved_case_ids"]:
            lines.extend(
                [
                    "",
                    "Improved holdout cases: "
                    + ", ".join(_cell(case) for case in held["improved_case_ids"]),
                ]
            )
    lines.extend(["", held["warning"]])
    lines.extend(["", "## Limits and assumptions", ""])
    lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"
