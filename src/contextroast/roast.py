# ruff: noqa: E501 -- static SVG template lines stay intact for readability.
"""Generate a self-contained SVG from safe report aggregates; never include private case content."""

from __future__ import annotations

import math
from typing import Any


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _accuracy(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Accuracy must be a finite fraction between zero and one")
    return float(value)


def render_roast_svg(report: dict[str, Any]) -> str:
    """Draw 1280x640 social card. Only fixed labels and validated numbers enter the SVG."""
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("Unsupported or invalid ContextRoast report")
    try:
        provider = report["provider"]
        if provider not in ("mock", "api", "python"):
            raise ValueError("Unknown report provider")
        baseline, final = report["baseline"], report["final"]
        before = _count(baseline["usage"]["estimated_input_tokens"], "Baseline input")
        after = _count(final["usage"]["estimated_input_tokens"], "Final input")
        accuracy_before = _accuracy(baseline["accuracy"])
        accuracy_after = _accuracy(final["accuracy"])
        kept, removed = report["kept_fields"], report["removed_fields"]
        if not isinstance(kept, list) or not isinstance(removed, list):
            raise ValueError("Invalid context field lists")
        if any(not isinstance(x, str) for x in kept + removed):
            raise ValueError("Context field names must be strings")
        status = report["holdout"]["status"]
        if status not in ("PASS", "FAIL", "NOT_VALIDATED"):
            raise ValueError("Invalid holdout status")
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete ContextRoast report") from exc

    if status == "FAIL":
        title = "The holdout caught a bad trim."
        subtitle = "Some removed context mattered on new questions."
        validation = "HOLDOUT FAILED / NOT DEPLOYMENT READY"
        accent = "#FDA4AF"
    elif status == "PASS":
        title = "Your agent can travel lighter."
        subtitle = "Holdout checks passed on the supplied cases."
        validation = "HOLDOUT PASSED / LIMITED CHECK"
        accent = "#86EFAC"
        if report.get("deployment_ready") is not True:
            validation = "HOLDOUT PASSED / QUALITY NOT ESTABLISHED"
    else:
        title = "Your agent packed extra baggage."
        subtitle = "Promising on this dataset. Holdout still needed."
        validation = "NOT VALIDATED / NOT DEPLOYMENT READY"
        accent = "#FCD34D"

    if provider == "mock":
        subtitle = "Synthetic mock only. Not a real-model benchmark."

    basis = (
        "SERIALIZED INPUT PROXY (NOT AGENT TOKEN USAGE)"
        if provider == "python"
        else "ESTIMATED INPUT TOKENS (NOT BILLING)"
    )
    provider_label = {
        "mock": "OFFLINE MOCK / NO REAL MODEL",
        "api": "OPENAI-COMPATIBLE API",
        "python": "PYTHON CALLABLE / MODEL USAGE MAY BE UNKNOWN",
    }[provider]
    fields = len(kept) + len(removed)
    reduction = 100 * (before - after) / before if before else 0
    bar_after = min(500, 500 * after / before) if before else 0

    # We deliberately never interpolate report dataset/model/field names, case IDs,
    # prompts, context values, predictions, user paths or pricing into this SVG.
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="640" viewBox="0 0 1280 640" role="img" aria-label="ContextRoast report summary">
<rect width="1280" height="640" rx="28" fill="#101827"/>
<circle cx="1180" cy="35" r="180" fill="#F97316" opacity=".08"/>
<rect x="60" y="45" width="6" height="34" rx="3" fill="#FB923C"/>
<text x="82" y="72" font-family="Arial,sans-serif" font-size="30" font-weight="bold" fill="#FFF7ED">CONTEXTROAST</text>
<text x="990" y="72" font-family="Arial,sans-serif" font-size="16" fill="#CBD5E1">CONTEXT ABLATION</text>
<text x="60" y="150" font-family="Arial,sans-serif" font-size="39" font-weight="bold" fill="#FFFFFF">{title}</text>
<text x="60" y="190" font-family="Arial,sans-serif" font-size="20" fill="#CBD5E1">{subtitle}</text>
<rect x="60" y="229" width="560" height="180" rx="16" fill="#1F2937"/>
<text x="85" y="272" font-family="Arial,sans-serif" font-size="17" fill="#CBD5E1">CONTEXT FIELDS REMOVED</text>
<text x="85" y="349" font-family="Arial,sans-serif" font-size="61" font-weight="bold" fill="#FDBA74">{len(removed)}/{fields}</text>
<text x="315" y="349" font-family="Arial,sans-serif" font-size="19" fill="#E2E8F0">Accuracy: {accuracy_before:.0%} to {accuracy_after:.0%}</text>
<rect x="650" y="229" width="570" height="180" rx="16" fill="#1F2937"/>
<text x="680" y="272" font-family="Arial,sans-serif" font-size="17" fill="#CBD5E1">INPUT-SIZE REDUCTION</text>
<text x="680" y="349" font-family="Arial,sans-serif" font-size="60" font-weight="bold" fill="#FDBA74">{reduction:.2f}%</text>
<text x="680" y="388" font-family="Arial,sans-serif" font-size="16" fill="#CBD5E1">{before:,} to {after:,}</text>
<text x="60" y="455" font-family="Arial,sans-serif" font-size="16" fill="#CBD5E1">BEFORE</text>
<rect x="180" y="440" width="500" height="18" rx="9" fill="#FCA5A5"/>
<text x="60" y="499" font-family="Arial,sans-serif" font-size="16" fill="#CBD5E1">AFTER</text>
<rect x="180" y="484" width="500" height="18" rx="9" fill="#374151"/>
<rect x="180" y="484" width="{bar_after:.2f}" height="18" rx="9" fill="#86EFAC"/>
<rect x="720" y="435" width="500" height="78" rx="12" fill="#293444"/>
<circle cx="748" cy="475" r="8" fill="{accent}"/>
<text x="770" y="481" font-family="Arial,sans-serif" font-size="17" font-weight="bold" fill="{accent}">{validation}</text>
<line x1="60" y1="538" x2="1220" y2="538" stroke="#475569"/>
<text x="60" y="575" font-family="Arial,sans-serif" font-size="15" fill="#CBD5E1">{basis}</text>
<text x="60" y="605" font-family="Arial,sans-serif" font-size="15" fill="#CBD5E1">{provider_label}</text>
<text x="1040" y="605" font-family="Arial,sans-serif" font-size="17" font-weight="bold" fill="#FDBA74">ContextRoast</text>
</svg>
"""
