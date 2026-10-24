import json

import pytest

from contextroast.data import dataset_from_dict, load_dataset
from contextroast.engine import max_evaluations, optimize
from contextroast.providers import MockProvider, Response

from .conftest import ROOT


class RequiredProvider:
    name = "test"
    model = "deterministic"

    def __init__(self):
        self.seen = []

    def generate(self, request):
        self.seen.append(frozenset(request.context))
        answer = 1 if "required" in request.context else 0
        return Response(json.dumps({"answer": answer}))


@pytest.mark.parametrize(
    "name,kept",
    [
        ("support.json", {"routing_policy", "account"}),
        ("shipping.yaml", {"shipping_rates", "currency"}),
    ],
)
def test_meaningful_mock_elimination(name, kept):
    data = load_dataset(ROOT / "examples" / name)
    result = optimize(data, MockProvider(data.mock_task))
    assert result.baseline.accuracy == result.final.accuracy == 1
    assert set(result.final.fields) == kept
    assert result.final.usage.estimated_input_tokens < result.baseline.usage.estimated_input_tokens
    assert any(not trial.within_tolerance for trial in result.trials)
    assert result.usage.calls == len(result.evaluations) * len(data.cases)


def test_cache_and_rejected_trials_are_counted(dataset):
    provider = RequiredProvider()
    result = optimize(dataset, provider)
    assert result.final.fields == ("required",)
    assert len(provider.seen) == len(set(provider.seen)) == 4
    assert len(result.evaluations) == max_evaluations(2)
    assert result.usage.calls == 4  # baseline, two candidates, then empty context
    assert result.usage.costing_source == "estimated"
    assert result.usage.reported_calls == 0


def test_tolerance_is_relative_to_original_baseline(raw_dataset):
    raw_dataset["cases"] = [
        {"id": str(i), "input": str(i), "context": {"required": 1, "noise": 1}, "expected": 1}
        for i in range(4)
    ]

    class GradualProvider(RequiredProvider):
        def generate(self, request):
            # 4/4 full, 3/4 with either field, 2/4 with neither.
            correct_count = len(request.context) + 2
            return Response(json.dumps({"answer": int(int(request.input) < correct_count)}))

    result = optimize(dataset_from_dict(raw_dataset), GradualProvider(), tolerance=0.25)
    assert result.final.accuracy == 0.75
    assert len(result.final.fields) == 1  # a second 25-point drop must be rejected


def test_tolerance_boundary_and_redundancy(raw_dataset):
    class RedundantProvider(RequiredProvider):
        def generate(self, request):
            return Response(json.dumps({"answer": int(bool(request.context))}))

    result = optimize(dataset_from_dict(raw_dataset), RedundantProvider())
    assert len(result.final.fields) == 1
    assert result.final.accuracy == 1
    # Larger noise field is removed first when accuracy ties.
    assert result.final.fields == ("required",)
    all_removed = optimize(dataset_from_dict(raw_dataset), RedundantProvider(), tolerance=1)
    assert all_removed.final.fields == ()


def test_greedy_prioritizes_accuracy_over_token_reduction(raw_dataset):
    raw_dataset["cases"] = [
        {
            "id": str(i),
            "input": str(i),
            "context": {"required": 1, "noise": "x" * 1000},
            "expected": 1,
        }
        for i in range(4)
    ]

    class AccuracyProvider(RequiredProvider):
        def generate(self, request):
            if "noise" in request.context:
                answer = 1
            else:
                answer = int("required" in request.context and request.input != "0")
            return Response(json.dumps({"answer": answer}))

    result = optimize(dataset_from_dict(raw_dataset), AccuracyProvider(), tolerance=0.25)
    assert result.final.fields == ("noise",)
    assert result.final.accuracy == 1


def test_empty_fields_evaluated_once(raw_dataset):
    raw_dataset["context_fields"] = []
    raw_dataset["cases"][0]["context"] = {}
    result = optimize(dataset_from_dict(raw_dataset), RequiredProvider())
    assert result.baseline is result.final
    assert len(result.evaluations) == 1
    assert result.trials == []


@pytest.mark.parametrize("tolerance", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_tolerance(dataset, tolerance):
    provider = RequiredProvider()
    with pytest.raises(ValueError, match="tolerance"):
        optimize(dataset, provider, tolerance)
    assert provider.seen == []
