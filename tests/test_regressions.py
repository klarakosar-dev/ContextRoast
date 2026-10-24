"""Regression tests for selection, scoring, and cost accounting."""

import copy
import json
from dataclasses import replace
from decimal import Decimal

import pytest
from typer.testing import CliRunner

from contextroast.cli import app
from contextroast.data import dataset_from_dict, load_dataset
from contextroast.engine import evaluate, optimize
from contextroast.errors import ProviderError
from contextroast.providers import MockProvider, OpenAICompatibleProvider, Request, Response
from contextroast.reports import Pricing, make_report, render_markdown

from .conftest import ROOT
from .test_reports import MeteredProvider


def test_demo_ablation_measures_accuracy_not_only_prompt_size():
    data = load_dataset(ROOT / "examples/support.json")
    provider = MockProvider(data.mock_task)
    full = evaluate(data, provider, data.context_fields, 0)
    assert full.correct == 8
    for name in data.context_fields:
        fields = tuple(field for field in data.context_fields if field != name)
        candidate = evaluate(data, provider, fields, 1)
        assert candidate.usage.estimated_input_tokens < full.usage.estimated_input_tokens
        assert candidate.correct == (0 if name in {"account", "routing_policy"} else 8)
    report = make_report(optimize(data, provider), Pricing())
    assert report["baseline"]["usage"]["estimated_input_tokens"] == 2736
    assert report["final"]["usage"]["estimated_input_tokens"] == 1312
    assert report["changes"]["estimated_input_reduction_fraction"] == pytest.approx(1424 / 2736)
    assert report["baseline"]["usage"]["reported_calls"] == 0
    assert report["projection"]["gross_savings_usd"] is None
    assert "Estimated input-token reduction: 52.05%" in render_markdown(report)


@pytest.mark.parametrize("change", ["account", "routing_policy", "input"])
def test_demo_predictions_depend_on_values_not_only_field_presence(change):
    data = load_dataset(ROOT / "examples/support.json")
    case = data.cases[0]  # invoice + enterprise -> billing/high
    provider = MockProvider(data.mock_task)
    context = copy.deepcopy(case.context)
    message = case.input
    if change == "account":
        context["account"]["tier"] = "standard"
    elif change == "routing_policy":
        context["routing_policy"]["routes"][0]["queue"] = "changed-queue"
    else:
        message = "I cannot login."
    original = provider.generate(Request(data.instruction, case.input, case.context)).text
    modified = provider.generate(Request(data.instruction, message, context)).text
    assert modified != original


def test_wrong_expected_answers_do_not_change_mock_predictions():
    data = load_dataset(ROOT / "examples/support.json")
    wrong = replace(data, cases=tuple(replace(case, expected="wrong-label") for case in data.cases))
    provider = MockProvider(data.mock_task)
    original = evaluate(data, provider, data.context_fields, 0)
    modified = evaluate(wrong, provider, wrong.context_fields, 0)
    assert [item.predicted for item in modified.outcomes] == [
        item.predicted for item in original.outcomes
    ]
    assert original.correct == 8
    assert modified.correct == 0


def correlated_dataset():
    return dataset_from_dict(
        {
            "version": 1,
            "name": "correlated-values",
            "instruction": "Read primary.value, or backup.value if primary is absent.",
            "context_fields": ["primary", "backup", "noise"],
            "cases": [
                {
                    "id": str(value),
                    "input": "Read value",
                    "expected": value,
                    "context": {
                        "primary": {"value": value},
                        "backup": {"value": value},
                        "noise": "x" * 100,
                    },
                }
                for value in [2, 5, 11, 19]
            ],
        }
    )


class CorrelatedProvider:
    name, model = "test", "correlated-reader"

    def generate(self, request):
        selected = request.context.get("primary", request.context.get("backup", {}))
        return Response(json.dumps({"answer": selected.get("value")}))


def test_correlated_fields_cannot_both_be_removed():
    data, provider = correlated_dataset(), CorrelatedProvider()
    assert evaluate(data, provider, ("primary",), 0).accuracy == 1
    assert evaluate(data, provider, ("backup",), 1).accuracy == 1
    assert evaluate(data, provider, ("noise",), 2).accuracy == 0
    result = optimize(data, provider)
    assert len(result.final.fields) == 1
    assert set(result.final.fields) <= {"primary", "backup"}
    assert result.final.accuracy == 1


def test_joint_removal_can_beat_greedy_search(raw_dataset):
    raw_dataset["context_fields"] = ["signal", "left", "right"]
    raw_dataset["cases"][0]["context"] = {"signal": 1, "left": "x" * 100, "right": "x" * 100}

    class InteractionProvider:
        name, model = "test", "paired-interaction"

        def generate(self, request):
            fields = request.context
            complete_pair = ("left" in fields) == ("right" in fields)
            value = fields.get("signal") if complete_pair else None
            return Response(json.dumps({"answer": value}))

    data, provider = dataset_from_dict(raw_dataset), InteractionProvider()
    greedy = optimize(data, provider)
    alternative = evaluate(data, provider, ("signal",), 99)
    assert greedy.final.fields == ("signal", "left", "right")
    assert alternative.accuracy == greedy.final.accuracy == 1
    assert alternative.usage.estimated_input_tokens < greedy.final.usage.estimated_input_tokens


def test_larger_outputs_can_erase_input_savings(dataset):
    class ExpandingOutput(MeteredProvider):
        def generate(self, request):
            result = super().generate(request)
            return replace(result, output_tokens=20 if "noise" in request.context else 1000)

    report = make_report(optimize(dataset, ExpandingOutput()), Pricing.parse("2", "8"), 10000)
    assert report["changes"]["estimated_input_tokens_saved"] > 0
    assert Decimal(report["projection"]["gross_savings_usd"]) < 0
    assert Decimal(report["projection"]["net_savings_after_evaluation_usd"]) < 0
    assert report["projection"]["break_even_requests"] is None


def test_invalid_model_output_still_counts_paid_usage(dataset):
    class InvalidOutput(MeteredProvider):
        def generate(self, request):
            return Response("not JSON", 120, 10)

    report = make_report(optimize(dataset, InvalidOutput()), Pricing.parse("2", "8"))
    assert report["baseline"]["accuracy"] == 0
    assert report["evaluation"]["usage"]["reported_calls"] == 4
    assert Decimal(report["evaluation"]["execution_cost_usd"]) == Decimal("0.00128")
    assert all(outcome["error"] == "invalid_json" for outcome in report["baseline"]["outcomes"])


@pytest.mark.parametrize("consent", [False, None, 0, 1, "false", "true"])
def test_python_adapter_requires_boolean_true_consent(consent, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No client may be constructed without boolean True consent")

    monkeypatch.setattr("httpx.Client", forbidden)
    with pytest.raises(ProviderError, match="allow-paid"):
        OpenAICompatibleProvider(model="fake", api_key="fake", allow_paid=consent)


@pytest.mark.parametrize("limit", [True, 1.5, float("nan"), float("inf"), 0, -1])
def test_python_adapter_rejects_invalid_completion_limits(limit):
    with pytest.raises(ProviderError, match="positive integer"):
        OpenAICompatibleProvider(
            model="fake", api_key="fake", allow_paid=True, max_output_tokens=limit
        )


def test_extreme_price_fails_before_arithmetic_overflow():
    with pytest.raises(ValueError, match="decimal range"):
        Pricing.parse("1e1000000", "1")


@pytest.mark.parametrize("price", ["1e1000000", "1e999999"])
def test_extreme_prices_return_cli_errors_instead_of_tracebacks(tmp_path, price):
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(ROOT / "examples/support.json"),
            "--output-dir",
            str(tmp_path / "out"),
            "--input-price",
            price,
            "--output-price",
            "1",
        ],
    )
    assert result.exit_code == 2
    assert "decimal range" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "options,message",
    [
        (["--provider", "api"], "--allow-paid"),
        (["--provider", "api", "--allow-paid"], "--model"),
        (["--provider", "api", "--allow-paid", "--model", "fake"], "API key"),
    ],
)
def test_incomplete_api_cli_configuration_never_constructs_client(
    tmp_path, options, message, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid configuration may not construct an HTTP client")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("httpx.Client", forbidden)
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(ROOT / "examples/support.json"),
            "--output-dir",
            str(tmp_path / "out"),
            *options,
        ],
    )
    assert result.exit_code == 2
    assert message in result.output


def test_allow_paid_flag_alone_still_uses_offline_mock(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("--allow-paid does not choose the API provider")

    monkeypatch.setattr("httpx.Client", forbidden)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(ROOT / "examples/support.json"),
            "--allow-paid",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["provider"] == "mock"
