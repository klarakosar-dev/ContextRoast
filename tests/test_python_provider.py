import copy
import inspect
import json
import sys
from types import ModuleType

import pytest
from typer.testing import CliRunner

from contextroast.cli import app
from contextroast.data import dataset_from_dict, load_dataset
from contextroast.engine import evaluate, optimize
from contextroast.errors import ProviderError
from contextroast.exports import recommended_schema, removal_trace
from contextroast.providers import MockProvider, Request, score_response
from contextroast.python_provider import AgentCase, AgentResult, PythonProvider
from contextroast.reports import Pricing, make_report, render_markdown

from .conftest import ROOT


@pytest.fixture
def entrypoint(monkeypatch):
    module = ModuleType("contextroast_fixture_agent")
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def register(function):
        module.predict = function
        return module.__name__ + ":predict"

    return register


@pytest.mark.parametrize("prediction", [None, True, 1, 1.5, "text", [1, None], {"answer": 2}])
def test_structured_values_are_predictions_not_ambiguous_wrappers(entrypoint, prediction):
    adapter = PythonProvider(entrypoint(lambda case, context: prediction), allow_paid=True)
    response = adapter.generate(Request("task", "input", {}, "case-1"))
    assert score_response(response, prediction)[0]
    assert adapter.invocations == 1


def test_case_has_no_gold_or_full_context_and_mutations_are_isolated(entrypoint):
    original = {"nested": {"items": [1, 2]}}

    def function(case, context):
        assert isinstance(case, AgentCase)
        assert (case.id, case.input, case.instruction) == ("c1", "input", "task")
        assert not hasattr(case, "expected") and not hasattr(case, "context")
        return context["nested"]["items"].pop()

    adapter = PythonProvider(entrypoint(function), allow_paid=True)
    request = Request("task", "input", original, "c1")
    assert score_response(adapter.generate(request), 2)[0]
    assert score_response(adapter.generate(request), 2)[0]
    assert original == {"nested": {"items": [1, 2]}}


@pytest.mark.parametrize("consent", [False, None, 1, "true"])
def test_consent_is_required_before_module_import(tmp_path, monkeypatch, consent):
    monkeypatch.syspath_prepend(str(tmp_path))
    sentinel = tmp_path / "executed"
    (tmp_path / "untrusted_agent.py").write_text(
        "from pathlib import Path\nPath(__file__).with_name('executed').touch()\n"
        "def predict(case, context): return 1\n"
    )
    with pytest.raises(ProviderError, match="--allow-paid"):
        PythonProvider("untrusted_agent:predict", allow_paid=consent)
    assert not sentinel.exists()
    assert "untrusted_agent" not in sys.modules


@pytest.mark.parametrize("value", ["file.py", "a:b:c", "path/to/agent:fn", "a:", ":fn", "a:fn()"])
def test_entrypoint_syntax_rejected(value):
    with pytest.raises(ProviderError, match="module:function"):
        PythonProvider(value, allow_paid=True)


@pytest.mark.parametrize("value", ["contextroast_no_such_module:predict", "contextroast:missing"])
def test_loading_errors_are_actionable(value):
    with pytest.raises(ProviderError, match="Cannot load Python entrypoint"):
        PythonProvider(value, allow_paid=True)


def test_import_error_does_not_echo_private_message(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "broken_import_agent.py").write_text(
        "raise RuntimeError('private-import-secret')\n"
    )
    with pytest.raises(ProviderError) as error:
        PythonProvider("broken_import_agent:predict", allow_paid=True)
    assert "RuntimeError" in str(error.value)
    assert "private-import-secret" not in str(error.value)


def test_working_directory_modules_can_load_without_pythonpath(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [path for path in sys.path if path not in ("", str(tmp_path))])
    monkeypatch.delitem(sys.modules, "checkout_agent", raising=False)
    (tmp_path / "checkout_agent.py").write_text("def predict(case, context): return {'ok': True}\n")
    adapter = PythonProvider("checkout_agent:predict", allow_paid=True)
    assert score_response(adapter.generate(Request("", "", {})), {"ok": True})[0]
    monkeypatch.delitem(sys.modules, "checkout_agent", raising=False)


@pytest.mark.parametrize("function", [42, lambda one: 1, lambda case, context, required: 1])
def test_noncallable_or_incompatible_signatures(entrypoint, function):
    with pytest.raises(ProviderError):
        PythonProvider(entrypoint(function), allow_paid=True)


def test_async_functions_rejected_and_async_results_closed(entrypoint):
    async def asynchronous(case, context):
        return 1

    with pytest.raises(ProviderError, match="synchronous"):
        PythonProvider(entrypoint(asynchronous), allow_paid=True)
    coroutine = asynchronous(None, {})
    adapter = PythonProvider(entrypoint(lambda case, context: coroutine), allow_paid=True)
    response = adapter.generate(Request("", "", {}))
    assert response.error == "python_async_result_unsupported"
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


def test_exceptions_score_incorrect_and_invocations_remain_counted(entrypoint, dataset):
    def function(case, context):
        raise RuntimeError("private-call-secret")

    adapter = PythonProvider(entrypoint(function), allow_paid=True)
    run = optimize(dataset, adapter)
    report = make_report(run, Pricing.parse("2", "8"))
    assert report["baseline"]["correct"] == 0
    assert adapter.invocations == run.usage.calls == report["evaluation"]["invocations"] == 4
    assert report["evaluation"]["execution_cost_usd"] is None
    assert "private-call-secret" not in json.dumps(report)
    assert report["baseline"]["outcomes"][0]["error"] == "python_exception:RuntimeError"


@pytest.mark.parametrize("prediction", [float("nan"), {1: "non-string-key"}, (1, 2), object()])
def test_invalid_structured_results_preserve_valid_usage(entrypoint, prediction):
    adapter = PythonProvider(
        entrypoint(lambda case, context: AgentResult(prediction, 10, 2)), allow_paid=True
    )
    response = adapter.generate(Request("", "", {}))
    assert response.error == "invalid_python_prediction"
    assert (response.input_tokens, response.output_tokens) == (10, 2)


@pytest.mark.parametrize("usage", [(1, None), (None, 1), (True, 1), (-1, 0), (1.2, 3)])
def test_invalid_usage_is_an_explicit_failed_invocation(entrypoint, usage):
    adapter = PythonProvider(
        entrypoint(lambda case, context: AgentResult(1, *usage)), allow_paid=True
    )
    response = adapter.generate(Request("", "", {}))
    assert response.error == "invalid_python_usage"
    assert response.input_tokens is response.output_tokens is None


def test_python_money_unknown_without_usage_even_when_prices_exist(entrypoint, dataset):
    adapter = PythonProvider(
        entrypoint(lambda case, context: int("required" in context)), allow_paid=True
    )
    report = make_report(optimize(dataset, adapter), Pricing.parse("2", "8"))
    assert report["baseline"]["inference_cost_usd"] is None
    assert report["projection"]["gross_savings_usd"] is None
    assert report["evaluation"]["execution_cost_usd"] is None
    assert report["cost_information"]["measured_billing_cost_usd"] is None
    assert report["cost_information"]["monetary_source"] == "unavailable"
    assert report["token_estimate_basis"] == "serialized_case_context_proxy"
    assert "serialized proxy" in render_markdown(report)


def test_reported_usage_prices_and_unreported_trial_costs(entrypoint, dataset):
    def function(case, context):
        return AgentResult(int("required" in context), 300 if "noise" in context else 100, 20)

    adapter = PythonProvider(entrypoint(function), allow_paid=True)
    report = make_report(optimize(dataset, adapter), Pricing.parse("2", "8"), 10)
    assert report["evaluation"]["execution_cost_usd"] == "0.00224"
    assert report["projection"]["modeled_paid_evaluation_break_even_requests"] == 6
    assert report["cost_information"]["monetary_source"] == "estimated_from_reported_usage"
    assert report["cost_information"]["measured_billing_cost_usd"] is None
    assert report["evaluation"]["underlying_api_calls"] is None

    def partially_reported(case, context):
        return function(case, context) if context else 0

    partial = PythonProvider(entrypoint(partially_reported), allow_paid=True)
    report = make_report(optimize(dataset, partial), Pricing.parse("2", "8"), 10)
    assert report["baseline"]["inference_cost_usd"] is not None
    assert report["projection"]["gross_savings_usd"] is not None
    assert report["evaluation"]["execution_cost_usd"] is None
    assert report["projection"]["net_savings_after_evaluation_usd"] is None
    assert report["evaluation"]["execution_cost_basis"] == "unavailable"


def test_no_reported_usage_and_missing_pricing_remain_distinct(entrypoint, dataset):
    adapter = PythonProvider(
        entrypoint(lambda case, context: AgentResult(1, 100, 5)), allow_paid=True
    )
    report = make_report(optimize(dataset, adapter), Pricing())
    assert report["evaluation"]["usage"]["costing_source"] == "provider_reported"
    assert report["pricing"]["complete"] is False
    assert report["evaluation"]["execution_cost_usd"] is None


def test_python_cli_mock_backed_integration_and_exports(entrypoint, tmp_path):
    calls = []
    mock = MockProvider("support-routing")

    def function(case, context):
        calls.append((case.id, tuple(context)))
        response = mock.generate(Request(case.instruction, case.input, context))
        return json.loads(response.text)["answer"]

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(ROOT / "examples/support.json"),
            "--provider",
            "python",
            "--entrypoint",
            entrypoint(function),
            "--allow-paid",
            "--export-context",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "may incur real API charges" in result.output
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert len(calls) == report["evaluation"]["invocations"] == 120
    assert report["kept_fields"] == ["routing_policy", "account"]
    assert report["final"]["accuracy"] == 1
    schema = json.loads((tmp_path / "out/recommended-context.schema.json").read_text())
    assert schema == recommended_schema(report)
    assert schema["required"] == ["routing_policy", "account"]
    trace = json.loads((tmp_path / "out/removal-trace.json").read_text())
    assert trace == removal_trace(report)
    assert sum(trial["accepted"] for trial in trace["trials"]) == 3
    assert "predicted" not in json.dumps(trace)


@pytest.mark.parametrize(
    "options",
    [
        ["--provider", "python", "--entrypoint", "never_import:fn"],
        ["--provider", "python", "--allow-paid"],
        ["--entrypoint", "never_import:fn"],
        [
            "--provider",
            "python",
            "--entrypoint",
            "never_import:fn",
            "--allow-paid",
            "--tolerance",
            "nan",
        ],
    ],
)
def test_cli_invalid_configuration_never_loads_python(tmp_path, monkeypatch, options):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Invalid configuration must not import user code")

    monkeypatch.setattr("contextroast.python_provider.importlib.import_module", forbidden)
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
    assert result.exit_code == 2, result.output
    assert calls == []


def test_export_overwrite_is_checked_before_python_import(tmp_path, monkeypatch):
    (tmp_path / "removal-trace.json").write_text("keep")
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(ROOT / "examples/support.json"),
            "--provider",
            "python",
            "--entrypoint",
            "never_import:fn",
            "--allow-paid",
            "--export-context",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "Refusing to overwrite" in result.output
    assert (tmp_path / "removal-trace.json").read_text() == "keep"


def test_real_callable_elimination_and_separate_holdout():
    data = load_dataset(ROOT / "examples/retrieval.json")
    adapter = PythonProvider("examples.retrieval_agent:predict", allow_paid=True)
    run = optimize(data, adapter)
    assert run.baseline.correct == run.final.correct == 8
    assert run.final.fields == ("billing_faq", "access_faq", "shipping_faq")
    assert adapter.invocations == run.usage.calls == 104
    holdout = load_dataset(ROOT / "examples/retrieval-holdout.json")
    assert evaluate(holdout, adapter, holdout.context_fields, 0).correct == 2
    assert evaluate(holdout, adapter, run.final.fields, 1).correct == 1


def test_real_callable_has_no_fixed_field_dependencies():
    raw = json.loads((ROOT / "examples/retrieval.json").read_text())
    mapping = {name: f"source_{index}" for index, name in enumerate(raw["context_fields"])}
    raw["context_fields"] = [mapping[name] for name in raw["context_fields"]]
    for case in raw["cases"]:
        case["context"] = {mapping[name]: value for name, value in case["context"].items()}
    # A previously unused field becomes necessary when the query mix changes.
    logo = json.loads((ROOT / "examples/retrieval-holdout.json").read_text())["cases"][0]
    logo["context"] = {mapping[name]: value for name, value in logo["context"].items()}
    raw["cases"].append(logo)
    data = dataset_from_dict(copy.deepcopy(raw))
    run = optimize(data, PythonProvider("examples.retrieval_agent:predict", allow_paid=True))
    assert run.final.correct == 9
    assert mapping["marketing_faq"] in run.final.fields
    assert mapping["legacy_faq"] not in run.final.fields
