import copy
import json
from dataclasses import asdict, replace
from decimal import Decimal

import pytest
from typer.testing import CliRunner

from contextroast.cli import app
from contextroast.data import dataset_from_dict, load_dataset
from contextroast.engine import optimize
from contextroast.errors import DatasetError
from contextroast.exports import recommended_schema, removal_trace
from contextroast.holdout import check_holdout, validate_holdout
from contextroast.providers import Response
from contextroast.python_provider import PythonProvider
from contextroast.reports import Pricing, make_report, render_markdown

from .conftest import ROOT
from .test_engine import RequiredProvider
from .test_reports import MeteredProvider


def holdout_for(data, count=2):
    return replace(
        data,
        name="untouched-holdout",
        cases=tuple(
            replace(data.cases[0], id=f"holdout-{index}", input=f"holdout-{index}")
            for index in range(count)
        ),
    )


def test_no_holdout_preserves_selection_calls_costs_and_stdout(dataset, tmp_path):
    provider = MeteredProvider()
    run = optimize(dataset, provider)
    report = make_report(run, Pricing.parse("2", "8"))
    assert report["holdout"]["status"] == "NOT_VALIDATED"
    assert report["deployment_ready"] is False
    assert report["evaluation"]["invocations"] == report["evaluation"]["selection_invocations"] == 4
    assert report["evaluation"]["holdout_invocations"] == 0
    assert report["evaluation"]["execution_cost_usd"] == "0.00224"
    assert recommended_schema(report)["x-contextroast"]["deployment_ready"] is False
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(ROOT / "examples/support.json"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0
    assert "Holdout:" not in result.output
    assert len(result.output.splitlines()) == 2
    assert "NOT_VALIDATED" in (tmp_path / "out/report.md").read_text()


def test_holdout_cannot_affect_selection_or_trial_order(raw_dataset):
    raw_dataset["context_fields"] = ["signal", "fallback", "noise"]
    raw_dataset["cases"][0]["context"] = {"signal": 1, "fallback": 1, "noise": "x" * 100}
    data = dataset_from_dict(raw_dataset)

    class RecordingProvider:
        name, model = "test", "different-dependencies"

        def __init__(self):
            self.calls = []

        def generate(self, request):
            self.calls.append((request.input, tuple(request.context)))
            field = "fallback" if request.input.startswith("holdout") else "signal"
            return Response(json.dumps({"answer": request.context.get(field, 0)}))

    provider = RecordingProvider()
    selection = optimize(data, provider)
    selection_state = copy.deepcopy(asdict(selection))
    optimization_calls = list(provider.calls)
    assert selection.final.fields == ("signal",)
    validation = validate_holdout(selection, holdout_for(data), provider)
    assert validation.status == "FAIL"  # a holdout-driven search would keep fallback instead
    assert asdict(selection) == selection_state
    assert provider.calls[: len(optimization_calls)] == optimization_calls
    assert all(
        value.startswith("holdout") for value, _ in provider.calls[len(optimization_calls) :]
    )
    changed = replace(
        holdout_for(data),
        cases=tuple(replace(case, expected=0) for case in holdout_for(data).cases),
    )
    assert validate_holdout(selection, changed, provider).status == "PASS"
    assert asdict(selection) == selection_state
    repeated = RecordingProvider()
    assert asdict(optimize(data, repeated)) == selection_state
    assert repeated.calls == optimization_calls


@pytest.mark.parametrize(
    "tolerance,status", [(0, "FAIL"), (0.4999, "FAIL"), (0.5, "PASS"), (1, "PASS")]
)
def test_independent_holdout_tolerance_and_case_explanations(tolerance, status):
    data = load_dataset(ROOT / "examples/retrieval.json")
    held = load_dataset(ROOT / "examples/retrieval-holdout.json")
    provider = PythonProvider("examples.retrieval_agent:predict", allow_paid=True)
    selection = optimize(data, provider)
    validation = validate_holdout(selection, held, provider, tolerance)
    report = make_report(selection, Pricing(), holdout=validation)
    assert report["holdout"]["status"] == status
    assert report["holdout"]["accuracy_delta"] == -0.5
    assert report["holdout"]["baseline"]["correct"] == 2
    assert report["holdout"]["final"]["correct"] == 1
    assert report["evaluation"]["selection_invocations"] == 104
    assert report["evaluation"]["holdout_invocations"] == 4
    assert report["evaluation"]["invocations"] == provider.invocations == 108
    assert report["deployment_ready"] == (status == "PASS")
    assert [case["case_id"] for case in report["holdout"]["regressed_cases"]] == ["logo"]
    regression = report["holdout"]["regressed_cases"][0]
    assert regression["expected"] == regression["baseline_prediction"]
    assert regression["expected"] != regression["final_prediction"]
    assert "company brand logo" in regression["input"]
    markdown = render_markdown(report)
    assert "logo" in markdown and "Selected-context prediction" in markdown
    assert "Small holdouts cannot establish generalization" in markdown
    schema = recommended_schema(report)
    assert schema["x-contextroast"]["holdout_status"] == status
    assert schema["x-contextroast"]["deployment_ready"] == (status == "PASS")
    if status == "FAIL":
        assert "not deployment-ready" in schema["title"]
    assert "company brand logo" not in json.dumps(schema)
    assert "baseline_prediction" not in json.dumps(removal_trace(report))
    assert report["projection"]["gross_savings_usd"] is None


def test_holdout_baseline_is_not_optimization_accuracy(dataset):
    class ConstantProvider(RequiredProvider):
        def generate(self, request):
            return Response(json.dumps({"answer": int(request.input != "holdout-1")}))

    provider = ConstantProvider()
    selection = optimize(dataset, provider)
    validation = validate_holdout(selection, holdout_for(dataset), provider)
    assert selection.baseline.accuracy == 1
    assert validation.baseline.accuracy == validation.final.accuracy == 0.5
    assert validation.status == "PASS"


def test_aggregate_pass_still_lists_regressed_and_improved_cases(dataset):
    class BalancedProvider(RequiredProvider):
        def generate(self, request):
            if request.input.startswith("holdout"):
                correct = (request.input == "holdout-0") == ("noise" in request.context)
                return Response(json.dumps({"answer": int(correct)}))
            return super().generate(request)

    provider = BalancedProvider()
    selection = optimize(dataset, provider)
    validation = validate_holdout(selection, holdout_for(dataset), provider)
    report = make_report(selection, Pricing(), holdout=validation)
    assert validation.status == "PASS"
    assert report["holdout"]["accuracy_delta"] == 0
    assert [case["case_id"] for case in report["holdout"]["regressed_cases"]] == ["holdout-0"]
    assert report["holdout"]["improved_case_ids"] == ["holdout-1"]
    assert "Improved holdout cases: holdout-1" in render_markdown(report)


def test_holdout_usage_enters_total_paid_cost_and_break_even(dataset):
    provider = MeteredProvider()
    selection = optimize(dataset, provider)
    validation = validate_holdout(selection, holdout_for(dataset), provider)
    report = make_report(selection, Pricing.parse("2", "8"), 10, holdout=validation)
    assert selection.usage.calls == 4  # report aggregation must not mutate selection
    assert report["evaluation"]["invocations"] == 8
    assert report["holdout"]["token_cost_usd"] == "0.00224"
    assert report["evaluation"]["execution_cost_usd"] == "0.00448"
    assert report["evaluation"]["underlying_api_calls"] == 8
    assert Decimal(report["projection"]["net_savings_after_evaluation_usd"]) == Decimal("-0.00048")
    assert report["projection"]["modeled_paid_evaluation_break_even_requests"] == 12


def test_unreported_python_holdout_usage_does_not_become_free(dataset):
    class PartiallyMetered(MeteredProvider):
        name = "python"

        def generate(self, request):
            response = super().generate(request)
            return Response(response.text) if request.input.startswith("holdout") else response

    provider = PartiallyMetered()
    selection = optimize(dataset, provider)
    validation = validate_holdout(selection, holdout_for(dataset), provider)
    report = make_report(selection, Pricing.parse("2", "8"), holdout=validation)
    assert report["baseline"]["inference_cost_usd"] is not None
    assert report["holdout"]["token_cost_usd"] is None
    assert report["evaluation"]["execution_cost_usd"] is None
    assert report["projection"]["net_savings_after_evaluation_usd"] is None
    assert report["evaluation"]["usage"]["reported_calls"] == 4
    assert report["cost_information"]["usage_source"] == "mixed"


def test_identical_subsets_are_validated_once(dataset):
    class BothRequired(RequiredProvider):
        def generate(self, request):
            self.seen.append(tuple(request.context))
            return Response(json.dumps({"answer": int(len(request.context) == 2)}))

    provider = BothRequired()
    selection = optimize(dataset, provider)
    validation = validate_holdout(selection, holdout_for(dataset), provider)
    assert selection.final.fields == selection.baseline.fields
    assert validation.baseline is validation.final
    assert validation.usage.calls == 2
    report = make_report(selection, Pricing(), holdout=validation)
    assert report["holdout"]["unique_subsets"] == 1
    assert report["evaluation"]["invocations"] == 5
    assert "No holdout cases changed" in render_markdown(report)


def test_zero_correctness_never_labels_context_deployment_ready(dataset):
    class AlwaysWrong(RequiredProvider):
        def generate(self, request):
            return Response('{"answer": 0}')

    provider = AlwaysWrong()
    selection = optimize(dataset, provider)
    validation = validate_holdout(selection, holdout_for(dataset), provider)
    report = make_report(selection, Pricing(), holdout=validation)
    assert validation.status == "PASS"  # no drop; still no useful accuracy
    assert report["deployment_ready"] is False
    assert "zero correctness" in " ".join(report["warnings"])


def test_holdout_prediction_failures_explain_case_regressions(dataset):
    class BrokenOnSelected(RequiredProvider):
        def generate(self, request):
            if request.input.startswith("holdout") and "noise" not in request.context:
                return Response("", error="synthetic_prediction_failure")
            return super().generate(request)

    provider = BrokenOnSelected()
    selection = optimize(dataset, provider)
    validation = validate_holdout(selection, holdout_for(dataset, 1), provider)
    report = make_report(selection, Pricing(), holdout=validation)
    assert report["holdout"]["status"] == "FAIL"
    regression = report["holdout"]["regressed_cases"][0]
    assert regression["baseline_prediction"] == 1
    assert regression["final_prediction"] is None
    assert regression["final_error"] == "synthetic_prediction_failure"
    assert "synthetic_prediction_failure" in render_markdown(report)


@pytest.mark.parametrize("tolerance", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_holdout_tolerance_makes_no_validation_calls(dataset, tolerance):
    provider = RequiredProvider()
    selection = optimize(dataset, provider)
    before = list(provider.seen)
    with pytest.raises(ValueError, match="holdout tolerance"):
        validate_holdout(selection, holdout_for(dataset), provider, tolerance)
    assert provider.seen == before


@pytest.mark.parametrize("incompatibility", ["fields", "instruction", "mock_task", "overlap"])
def test_incompatible_or_overlapping_holdouts_are_rejected(dataset, incompatibility):
    held = holdout_for(dataset)
    if incompatibility == "fields":
        held = replace(held, context_fields=("other",))
    elif incompatibility == "instruction":
        held = replace(held, instruction="Another task")
    elif incompatibility == "mock_task":
        held = replace(held, mock_task="shipping-quote")
    else:
        held = replace(
            held, cases=tuple(replace(case, id="changed-id", expected=0) for case in dataset.cases)
        )
    with pytest.raises(DatasetError):
        check_holdout(dataset, held, "mock")


def test_field_order_may_differ_but_provider_model_must_match(dataset):
    held = replace(holdout_for(dataset), context_fields=tuple(reversed(dataset.context_fields)))
    provider = RequiredProvider()
    selection = optimize(dataset, provider)
    assert validate_holdout(selection, held, provider).status == "PASS"
    other = RequiredProvider()
    other.model = "different-model"
    with pytest.raises(ValueError, match="same provider/model"):
        validate_holdout(selection, held, other)
    assert other.seen == []


def test_cli_failure_writes_reviewable_safe_exports_and_exits_one(tmp_path):
    output = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(ROOT / "examples/retrieval.json"),
            "--provider",
            "python",
            "--entrypoint",
            "examples.retrieval_agent:predict",
            "--allow-paid",
            "--export-context",
            "--holdout",
            str(ROOT / "examples/retrieval-holdout.json"),
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "Up to 132 callable invocations" in result.output
    assert "Holdout: FAIL" in result.output and "regressed cases: logo" in result.output
    assert "Small holdouts cannot establish generalization" in result.output
    report = json.loads((output / "report.json").read_text())
    assert report["kept_fields"] == ["billing_faq", "access_faq", "shipping_faq"]
    assert report["evaluation"]["invocations"] == 108
    assert (
        json.loads((output / "recommended-context.schema.json").read_text())["x-contextroast"][
            "deployment_ready"
        ]
        is False
    )


def test_cli_pass_accepts_configured_tolerance_and_retains_regressions(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(ROOT / "examples/retrieval.json"),
            "--provider",
            "python",
            "--entrypoint",
            "examples.retrieval_agent:predict",
            "--allow-paid",
            "--holdout",
            str(ROOT / "examples/retrieval-holdout.json"),
            "--holdout-tolerance",
            "0.5",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Holdout: PASS" in result.output and "regressed cases: logo" in result.output


@pytest.mark.parametrize("problem", ["missing", "malformed", "incompatible", "same_file", "nan"])
def test_bad_holdout_fails_before_loading_user_code(tmp_path, monkeypatch, problem):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Bad holdout must not load user code")

    monkeypatch.setattr("contextroast.cli.PythonProvider", forbidden)
    path = tmp_path / "holdout.json"
    if problem == "malformed":
        path.write_text("{")
    elif problem == "incompatible":
        raw = json.loads((ROOT / "examples/retrieval-holdout.json").read_text())
        raw["instruction"] = "Wrong task"
        path.write_text(json.dumps(raw))
    elif problem == "same_file":
        path = ROOT / "examples/retrieval.json"
    elif problem == "nan":
        path = ROOT / "examples/retrieval-holdout.json"
    options = ["--holdout-tolerance", "nan"] if problem == "nan" else []
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(ROOT / "examples/retrieval.json"),
            "--provider",
            "python",
            "--entrypoint",
            "never_import:predict",
            "--allow-paid",
            "--holdout",
            str(path),
            "--output-dir",
            str(tmp_path / "out"),
            *options,
        ],
    )
    assert result.exit_code == 2, result.output
    assert calls == []
