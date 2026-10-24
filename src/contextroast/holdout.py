"""Post-selection holdout checks. Nothing here chooses or changes context fields."""

from __future__ import annotations

import math
from dataclasses import dataclass

from contextroast.data import Dataset, canonical_json
from contextroast.engine import Evaluation, Run, TokenTotals, evaluate
from contextroast.errors import DatasetError
from contextroast.providers import Provider

HOLDOUT_WARNING = (
    "Small holdouts cannot establish generalization or statistical certainty. "
    "PASS only means the configured accuracy-drop gate passed on these cases; "
    "it is not a production-safety guarantee. Do not retune fields on this holdout."
)


def check_holdout(optimization: Dataset, holdout: Dataset, provider: str) -> None:
    """Check compatibility/obvious overlap before any imports or paid model calls."""
    if set(optimization.context_fields) != set(holdout.context_fields):
        raise DatasetError(
            "Holdout must declare the same context field names as the optimization dataset"
        )
    if optimization.instruction != holdout.instruction:
        raise DatasetError("Holdout must use the same task instruction as the optimization dataset")
    if provider == "mock" and optimization.mock_task != holdout.mock_task:
        raise DatasetError("Holdout mock_task must match the optimization dataset")
    requests = {(case.input, canonical_json(case.context)) for case in optimization.cases}
    if any((case.input, canonical_json(case.context)) in requests for case in holdout.cases):
        raise DatasetError(
            "Holdout repeats optimization requests. Supply separate, untouched cases; "
            "changing case IDs or labels does not make a request independent."
        )


@dataclass(frozen=True)
class HoldoutValidation:
    dataset: Dataset
    tolerance: float
    baseline: Evaluation
    final: Evaluation
    usage: TokenTotals

    @property
    def status(self) -> str:
        return (
            "PASS"
            if self.final.accuracy + 1e-12 >= self.baseline.accuracy - self.tolerance
            else "FAIL"
        )


def validate_holdout(
    run: Run, holdout: Dataset, provider: Provider, tolerance: float = 0.0
) -> HoldoutValidation:
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("holdout tolerance must be a finite number between 0 and 1")
    check_holdout(run.dataset, holdout, provider.name)
    if (provider.name, provider.model) != (run.provider, run.model):
        raise ValueError("Holdout must use the same provider/model as field selection")
    # The selection is already complete. Never call optimize() or mutate the Run here.
    baseline = evaluate(holdout, provider, run.baseline.fields, 0)
    final = (
        baseline
        if run.final.fields == run.baseline.fields
        else evaluate(holdout, provider, run.final.fields, 1)
    )
    usage = TokenTotals()
    usage.add(baseline.usage)
    if final is not baseline:
        usage.add(final.usage)
    return HoldoutValidation(holdout, tolerance, baseline, final, usage)
