import json
from decimal import Decimal

import pytest

from contextroast.data import dataset_from_dict, load_dataset
from contextroast.engine import optimize
from contextroast.providers import MockProvider, Response
from contextroast.reports import Pricing, make_report, render_markdown

from .conftest import ROOT
from .test_engine import RequiredProvider


class MeteredProvider(RequiredProvider):
    name = "api"

    def generate(self, request):
        answer = 1 if "required" in request.context else 0
        tokens = 300 if "noise" in request.context else 100
        return Response(json.dumps({"answer": answer}), tokens, 20)


def test_complete_evaluation_cost_and_break_even(dataset):
    run = optimize(dataset, MeteredProvider())
    report = make_report(run, Pricing.parse("2", "8"), projected_requests=10)
    assert run.usage.calls == 4
    assert report["baseline"]["inference_cost_usd"] == "0.00076"
    assert report["final"]["inference_cost_usd"] == "0.00036"
    assert report["evaluation"]["execution_cost_usd"] == "0.00224"
    assert report["evaluation"]["usage"]["costing_source"] == "provider_reported"
    assert report["projection"]["savings_per_request_usd"] == "0.00040"
    assert Decimal(report["projection"]["gross_savings_usd"]) == Decimal("0.004")
    assert Decimal(report["projection"]["net_savings_after_evaluation_usd"]) == Decimal("0.00176")
    assert report["projection"]["break_even_requests"] == 6
    assert report["evaluation"]["usage"]["reported_input_tokens"] == 800


def test_negative_net_savings(dataset):
    report = make_report(optimize(dataset, MeteredProvider()), Pricing.parse("2", "8"), 1)
    assert Decimal(report["projection"]["net_savings_after_evaluation_usd"]) < 0


@pytest.mark.parametrize("prices", [(None, None), ("2", None), (None, "8")])
def test_missing_pricing_is_unknown(dataset, prices):
    report = make_report(optimize(dataset, MeteredProvider()), Pricing.parse(*prices))
    assert report["baseline"]["inference_cost_usd"] is None
    assert report["evaluation"]["execution_cost_usd"] is None
    assert report["projection"]["gross_savings_usd"] is None
    assert report["projection"]["break_even_requests"] is None
    assert any("never zero" in warning for warning in report["warnings"])


def test_explicit_zero_prices_are_valid(dataset):
    report = make_report(optimize(dataset, MeteredProvider()), Pricing.parse("0", "0"))
    assert report["pricing"]["complete"]
    assert Decimal(report["evaluation"]["execution_cost_usd"]) == 0
    assert report["projection"]["break_even_requests"] is None


def test_mock_free_execution_separate_from_hypothetical_cost(dataset):
    provider = RequiredProvider()
    provider.name = "mock"
    report = make_report(optimize(dataset, provider), Pricing.parse("2", "8"))
    assert report["evaluation"]["execution_cost_usd"] == "0"
    assert Decimal(report["evaluation"]["token_cost_usd"]) > 0
    assert report["projection"]["break_even_requests"] == 0
    unknown = make_report(optimize(dataset, provider), Pricing())
    assert unknown["evaluation"]["execution_cost_usd"] == "0"
    assert unknown["evaluation"]["token_cost_usd"] is None
    assert unknown["projection"]["gross_savings_usd"] is None


def test_mixed_token_usage_explicit(dataset):
    class MixedProvider(MeteredProvider):
        def generate(self, request):
            response = super().generate(request)
            return response if "noise" in request.context else Response(response.text)

    report = make_report(optimize(dataset, MixedProvider()), Pricing.parse("2", "8"))
    usage = report["evaluation"]["usage"]
    assert usage["calls"] == 4
    assert usage["reported_calls"] == 2
    assert usage["costing_source"] == "mixed"
    assert usage["estimated_input_tokens"] > 0
    assert usage["reported_input_tokens"] == 600
    assert any("mix" in warning for warning in report["warnings"])


@pytest.mark.parametrize("value", ["-1", "NaN", "Infinity", "not-a-number"])
def test_invalid_prices(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        Pricing.parse(value, "1")


def test_no_positive_savings(dataset):
    class FixedTokens(MeteredProvider):
        def generate(self, request):
            response = super().generate(request)
            return Response(response.text, 100, 20)

    report = make_report(optimize(dataset, FixedTokens()), Pricing.parse("2", "8"))
    assert Decimal(report["projection"]["savings_per_request_usd"]) == 0
    assert report["projection"]["break_even_requests"] is None


def test_reports_serialize_and_escape_markdown(raw_dataset):
    raw_dataset["name"] = "dataset|with\nnewlines"
    report = make_report(optimize(dataset_from_dict(raw_dataset), RequiredProvider()), Pricing())
    serialized = json.loads(json.dumps(report, allow_nan=False))
    assert serialized["schema_version"] == 1
    markdown = render_markdown(report)
    assert "dataset\\|with newlines" in markdown
    assert "unknown" in markdown
    assert "not globally optimal" in markdown
    assert "statistical certainty" in markdown
    assert "Provider-reported" in markdown


def test_zero_baseline_warning_and_invalid_projection(raw_dataset):
    raw_dataset["cases"][0]["expected"] = "impossible"
    run = optimize(dataset_from_dict(raw_dataset), RequiredProvider())
    report = make_report(run, Pricing())
    assert report["baseline"]["accuracy"] == 0
    assert any("Baseline accuracy is zero" in warning for warning in report["warnings"])
    with pytest.raises(ValueError):
        make_report(run, Pricing(), -1)


def test_paid_evaluation_break_even_is_separate_from_free_mock_execution():
    data = load_dataset(ROOT / "examples/support.json")
    report = make_report(optimize(data, MockProvider(data.mock_task)), Pricing.parse("2", "8"))
    projection = report["projection"]
    assert projection["execution_break_even_requests"] == projection["break_even_requests"] == 0
    assert projection["modeled_paid_evaluation_break_even_requests"] == 189
    assert Decimal(projection["net_savings_after_modeled_paid_evaluation_usd"]) == Decimal(
        "3.492902"
    )
    assert "Modeled paid-evaluation break-even requests: 189" in render_markdown(report)
    assert "Execution-cost break-even requests: 0" in render_markdown(report)
