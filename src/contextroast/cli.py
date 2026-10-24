from __future__ import annotations

import json
import math
import os
from decimal import DecimalException
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from contextroast import __version__
from contextroast.data import load_dataset
from contextroast.engine import max_evaluations, optimize
from contextroast.errors import ContextRoastError
from contextroast.exports import recommended_schema, removal_trace
from contextroast.holdout import HOLDOUT_WARNING, check_holdout, validate_holdout
from contextroast.providers import MockProvider, OpenAICompatibleProvider
from contextroast.python_provider import PythonProvider
from contextroast.reports import Pricing, make_report, render_markdown
from contextroast.roast import render_roast_svg

app = typer.Typer(
    help="Find removable LLM context fields. Offline by default.", no_args_is_help=True
)


class ProviderChoice(StrEnum):
    mock = "mock"
    api = "api"
    python = "python"


def _version(value: bool) -> None:
    if value:
        typer.echo(f"ContextRoast {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version and exit."),
    ] = False,
) -> None:
    pass


@app.command()
def validate(dataset: Annotated[Path, typer.Argument(help="JSON or YAML dataset.")]) -> None:
    """Check a dataset without making any model calls."""
    try:
        data = load_dataset(dataset)
    except ContextRoastError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(
        f"Valid: {data.name} — {len(data.cases)} cases, {len(data.context_fields)} context fields"
    )


@app.command()
def run(
    dataset: Annotated[Path, typer.Argument(help="JSON or YAML dataset.")],
    provider: Annotated[
        ProviderChoice, typer.Option(help="mock is offline; api/python require --allow-paid.")
    ] = ProviderChoice.mock,
    tolerance: Annotated[
        float,
        typer.Option(
            min=0, max=1, help="Allowed absolute accuracy drop; 0.05 = 5 percentage points."
        ),
    ] = 0.0,
    holdout: Annotated[
        Path | None,
        typer.Option(help="Separate untouched dataset; checked only after field selection."),
    ] = None,
    holdout_tolerance: Annotated[
        float,
        typer.Option(min=0, max=1, help="Allowed absolute accuracy drop on holdout; default: 0."),
    ] = 0.0,
    input_price: Annotated[
        str | None, typer.Option(help="USD per million input tokens; omitted = unknown.")
    ] = None,
    output_price: Annotated[
        str | None, typer.Option(help="USD per million output tokens; omitted = unknown.")
    ] = None,
    projected_requests: Annotated[
        int, typer.Option(min=0, help="Future request count for savings projection.")
    ] = 10_000,
    output_dir: Annotated[Path, typer.Option(help="Write report.json and report.md here.")] = Path(
        "reports"
    ),
    allow_paid: Annotated[
        bool,
        typer.Option(
            help="Permit API calls or trusted Python code/imports, including potential charges."
        ),
    ] = False,
    entrypoint: Annotated[
        str | None,
        typer.Option(help="Trusted module:function for --provider python; imports execute code."),
    ] = None,
    export_context: Annotated[
        bool,
        typer.Option(help="Also write recommended-context.schema.json and removal-trace.json."),
    ] = False,
    model: Annotated[
        str | None, typer.Option(help="Required API model ID; no model is selected implicitly.")
    ] = None,
    base_url: Annotated[
        str, typer.Option(help="OpenAI-compatible API base URL, including /v1 if required.")
    ] = "https://api.openai.com/v1",
    api_key_env: Annotated[
        str, typer.Option(help="Environment variable holding the API key.")
    ] = "OPENAI_API_KEY",
    timeout: Annotated[
        float, typer.Option(min=0.1, help="Seconds per API request; no automatic retries.")
    ] = 30.0,
    max_output_tokens: Annotated[
        int, typer.Option(min=1, help="API max_completion_tokens per call.")
    ] = 512,
) -> None:
    """Evaluate the baseline, remove context greedily, and write JSON + Markdown reports."""
    adapter = None
    try:
        data = load_dataset(dataset)
        pricing = Pricing.parse(input_price, output_price)
        if not math.isfinite(tolerance) or not math.isfinite(holdout_tolerance):
            raise ValueError("accuracy and holdout tolerances must be finite")
        holdout_data = load_dataset(holdout) if holdout is not None else None
        if holdout_data is not None:
            check_holdout(data, holdout_data, provider.value)
        if entrypoint is not None and provider != ProviderChoice.python:
            raise ValueError("--entrypoint requires --provider python")
        # Validate cheap inputs and output location before even constructing the API client.
        if provider == ProviderChoice.api and not allow_paid:
            raise ValueError("API calls require --provider api --allow-paid")
        if provider == ProviderChoice.python and not allow_paid:
            raise ValueError(
                "Python entrypoints execute trusted local code, including imports, and may incur "
                "real API charges. Use --allow-paid to consent before loading."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        filenames = ["report.json", "report.md"]
        if export_context:
            filenames += ["recommended-context.schema.json", "removal-trace.json"]
        for filename in filenames:
            target = output_dir / filename
            if target.exists():
                raise ValueError(f"Refusing to overwrite {target}; choose a new --output-dir")
        bound = max_evaluations(len(data.context_fields)) * len(data.cases)
        if holdout_data is not None:
            bound += 2 * len(holdout_data.cases)
        if provider == ProviderChoice.mock:
            adapter = MockProvider(data.mock_task)
        elif provider == ProviderChoice.python:
            if not entrypoint:
                raise ValueError("--entrypoint module:function is required for --provider python")
            typer.echo(
                "WARNING: Python entrypoints execute trusted local code without sandboxing, "
                "including imports, and may incur real API charges. Explicit consent enabled. "
                f"Up to {bound} callable invocations; "
                "internal API calls and spend are not bounded.",
                err=True,
            )
            adapter = PythonProvider(entrypoint, allow_paid=allow_paid)
        else:
            if not model:
                raise ValueError("--model is required for the API provider")
            adapter = OpenAICompatibleProvider(
                model=model,
                api_key=os.environ.get(api_key_env, ""),
                allow_paid=allow_paid,
                base_url=base_url,
                timeout=timeout,
                max_output_tokens=max_output_tokens,
            )
            typer.echo(
                f"API calls explicitly enabled: up to {bound} calls, "
                f"{max_output_tokens} completion tokens per call. No automatic retries.",
                err=True,
            )
        selection = optimize(data, adapter, tolerance)
        validation = (
            validate_holdout(selection, holdout_data, adapter, holdout_tolerance)
            if holdout_data is not None
            else None
        )
        result = make_report(selection, pricing, projected_requests, holdout=validation)
        (output_dir / "report.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (output_dir / "report.md").write_text(render_markdown(result), encoding="utf-8")
        if export_context:
            for filename, payload in (
                ("recommended-context.schema.json", recommended_schema(result)),
                ("removal-trace.json", removal_trace(result)),
            ):
                (output_dir / filename).write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
    except (ContextRoastError, OSError, ValueError, DecimalException) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc
    finally:
        if isinstance(adapter, OpenAICompatibleProvider):
            adapter.close()
    reduction_label = (
        "serialized input-proxy reduction"
        if provider == ProviderChoice.python
        else "estimated input-token reduction"
    )
    typer.echo(
        f"Accuracy: {result['baseline']['accuracy']:.2%} -> {result['final']['accuracy']:.2%}; "
        f"removed {len(result['removed_fields'])}/{len(data.context_fields)} fields; "
        f"{reduction_label}: "
        f"{result['changes']['estimated_input_reduction_fraction']:.2%}"
    )
    typer.echo(f"Reports: {output_dir / 'report.json'} and {output_dir / 'report.md'}")
    if holdout_data is not None:
        held = result["holdout"]
        regressed = ", ".join(case["case_id"] for case in held["regressed_cases"]) or "none"
        typer.echo(
            f"Holdout: {held['status']}; accuracy {held['baseline']['accuracy']:.2%} -> "
            f"{held['final']['accuracy']:.2%}; regressed cases: {regressed}"
        )
        typer.echo(f"Warning: {HOLDOUT_WARNING}", err=True)
        if held["status"] == "FAIL":
            raise typer.Exit(1)


@app.command("roast")
def roast(
    report: Annotated[Path, typer.Argument(help="An existing report.json; no model calls.")],
    output: Annotated[Path, typer.Option(help="New SVG output path.")] = Path("roast.svg"),
) -> None:
    """Create a shareable, privacy-preserving SVG report card."""
    try:
        if output.exists():
            raise ValueError(f"Refusing to overwrite {output}; choose another --output")
        payload = json.loads(report.read_text(encoding="utf-8"))
        svg = render_roast_svg(payload)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(svg, encoding="utf-8")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"ContextRoast SVG: {output} (aggregate metrics only)")
