"""Optional machine-readable field recommendations without copying context values."""

from typing import Any


def _validation_metadata(report: dict[str, Any]) -> dict[str, Any]:
    held = report.get("holdout", {})
    return {
        "holdout_status": held.get("status", "NOT_VALIDATED"),
        "deployment_ready": report.get("deployment_ready", False),
        "holdout_tolerance": held.get("tolerance"),
        "readiness_scope": report.get("readiness_scope", "not validated"),
        "warning": "Small holdouts cannot establish generalization. Review before deployment.",
    }


def recommended_schema(report: dict[str, Any]) -> dict[str, Any]:
    validation = _validation_metadata(report)
    suffix = " (not deployment-ready)" if not validation["deployment_ready"] else ""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"ContextRoast context fields — {validation['holdout_status']}{suffix}",
        "description": "Field names selected on the supplied evaluation set. Value types are "
        "unconstrained. Validate on a separate holdout before deployment.",
        "type": "object",
        "properties": {name: {} for name in report["kept_fields"]},
        "required": report["kept_fields"],
        "additionalProperties": False,
        "x-contextroast": validation,
    }


def removal_trace(report: dict[str, Any]) -> dict[str, Any]:
    by_id = {evaluation["id"]: evaluation for evaluation in report["evaluations"]}
    return {
        "schema_version": 1,
        "dataset": report["dataset"],
        "baseline_accuracy": report["baseline"]["accuracy"],
        "minimum_accuracy": report["minimum_accuracy"],
        "kept_fields": report["kept_fields"],
        "validation": _validation_metadata(report),
        "trials": [
            {
                **trial,
                "candidate_fields": by_id[trial["evaluation_id"]]["fields"],
                "accuracy": by_id[trial["evaluation_id"]]["accuracy"],
                "accuracy_delta_from_baseline": (
                    by_id[trial["evaluation_id"]]["accuracy"] - report["baseline"]["accuracy"]
                ),
            }
            for trial in report["trials"]
        ],
    }
