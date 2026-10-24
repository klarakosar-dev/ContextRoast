"""Greedy backward elimination with a fixed baseline and run-local subset cache."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from contextroast.data import Dataset
from contextroast.providers import (
    Provider,
    Request,
    estimate_input_tokens,
    estimate_tokens,
    score_response,
)


@dataclass
class TokenTotals:
    calls: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    reported_calls: int = 0
    reported_input_tokens: int = 0
    reported_output_tokens: int = 0
    costing_input_tokens: int = 0
    costing_output_tokens: int = 0

    @property
    def costing_source(self) -> str:
        if self.reported_calls == self.calls and self.calls:
            return "provider_reported"
        return "mixed" if self.reported_calls else "estimated"

    def add(self, other: TokenTotals) -> None:
        for key in self.__dataclass_fields__:
            setattr(self, key, getattr(self, key) + getattr(other, key))


@dataclass(frozen=True)
class Outcome:
    case_id: str
    correct: bool
    predicted: Any
    error: str | None


@dataclass
class Evaluation:
    id: int
    fields: tuple[str, ...]
    outcomes: tuple[Outcome, ...]
    usage: TokenTotals

    @property
    def correct(self) -> int:
        return sum(outcome.correct for outcome in self.outcomes)

    @property
    def accuracy(self) -> float:
        return self.correct / len(self.outcomes)


@dataclass
class Trial:
    round: int
    removed_field: str
    evaluation_id: int
    within_tolerance: bool
    accepted: bool = False


@dataclass
class Run:
    dataset: Dataset
    provider: str
    model: str
    tolerance: float
    baseline: Evaluation
    final: Evaluation
    evaluations: list[Evaluation]
    trials: list[Trial]
    usage: TokenTotals = field(default_factory=TokenTotals)


def max_evaluations(field_count: int) -> int:
    return 1 + field_count * (field_count + 1) // 2


def evaluate(
    dataset: Dataset, provider: Provider, fields: tuple[str, ...], index: int
) -> Evaluation:
    outcomes, totals = [], TokenTotals()
    for case in dataset.cases:
        request = Request(
            dataset.instruction,
            case.input,
            {name: case.context[name] for name in fields},
            case_id=case.id,
        )
        response = provider.generate(request)
        estimated_input, estimated_output = (
            estimate_input_tokens(request),
            estimate_tokens(response.text),
        )
        reported = all(
            type(number) is int and number >= 0
            for number in (response.input_tokens, response.output_tokens)
        )
        tokens = TokenTotals(
            calls=1,
            estimated_input_tokens=estimated_input,
            estimated_output_tokens=estimated_output,
            reported_calls=int(reported),
            reported_input_tokens=response.input_tokens if reported else 0,
            reported_output_tokens=response.output_tokens if reported else 0,
            costing_input_tokens=response.input_tokens if reported else estimated_input,
            costing_output_tokens=response.output_tokens if reported else estimated_output,
        )
        totals.add(tokens)
        correct, predicted, error = score_response(response, case.expected)
        outcomes.append(Outcome(case.id, correct, predicted, error))
    return Evaluation(index, fields, tuple(outcomes), totals)


def optimize(dataset: Dataset, provider: Provider, tolerance: float = 0.0) -> Run:
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("accuracy tolerance must be a finite number between 0 and 1")
    evaluations: list[Evaluation] = []
    cache: dict[tuple[str, ...], Evaluation] = {}

    def get_evaluation(fields: tuple[str, ...]) -> Evaluation:
        if fields not in cache:
            result = evaluate(dataset, provider, fields, len(evaluations))
            evaluations.append(result)
            cache[fields] = result
        return cache[fields]

    baseline = current = get_evaluation(dataset.context_fields)
    minimum_accuracy = baseline.accuracy - tolerance
    trials: list[Trial] = []
    round_number = 0
    while current.fields:
        round_number += 1
        feasible: list[tuple[Evaluation, Trial]] = []
        for name in current.fields:
            candidate = get_evaluation(tuple(field for field in current.fields if field != name))
            allowed = candidate.accuracy + 1e-12 >= minimum_accuracy
            trial = Trial(round_number, name, candidate.id, allowed)
            trials.append(trial)
            if allowed:
                feasible.append((candidate, trial))
        if not feasible:
            break
        # Stable ties follow declared field order. Favor accuracy, then smaller prompts.
        current, winner = max(
            feasible, key=lambda pair: (pair[0].accuracy, -pair[0].usage.estimated_input_tokens)
        )
        winner.accepted = True
    run = Run(
        dataset, provider.name, provider.model, tolerance, baseline, current, evaluations, trials
    )
    for result in evaluations:
        run.usage.add(result.usage)
    return run
