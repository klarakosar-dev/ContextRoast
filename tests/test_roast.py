"""SVG shares aggregate metrics, not user data; CLI has no model/network dependencies."""

import json

import pytest
from typer.testing import CliRunner

from contextroast.cli import app
from contextroast.data import load_dataset
from contextroast.engine import optimize
from contextroast.providers import MockProvider
from contextroast.reports import Pricing, make_report
from contextroast.roast import render_roast_svg

from .conftest import ROOT


def example_report():
    data = load_dataset(ROOT / "examples/support.json")
    return make_report(optimize(data, MockProvider(data.mock_task)), Pricing())


def test_svg_labels_mock_as_synthetic():
    svg = render_roast_svg(example_report())
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert "NOT VALIDATED" in svg
    assert "OFFLINE MOCK / NO REAL MODEL" in svg
    assert "52.05%" in svg


def test_private_text_is_not_exported():
    report = example_report()
    secret = "</svg><script>do-not-share-this</script>"
    report["dataset"] = secret
    report["model"] = secret
    report["removed_fields"] = [secret]
    report["baseline"]["outcomes"][0]["predicted"] = secret
    report["baseline"]["outcomes"][0]["case_id"] = secret
    svg = render_roast_svg(report)
    assert secret not in svg
    assert "do-not-share-this" not in svg


@pytest.mark.parametrize(
    "status, phrase",
    [
        ("FAIL", "HOLDOUT FAILED"),
        ("PASS", "HOLDOUT PASSED"),
        ("NOT_VALIDATED", "NOT VALIDATED"),
    ],
)
def test_holdout_status_is_always_visible(status, phrase):
    report = example_report()
    report["holdout"]["status"] = status
    report["deployment_ready"] = status == "PASS"
    svg = render_roast_svg(report)
    assert phrase in svg
    if status != "PASS":
        assert "NOT DEPLOYMENT READY" in svg


def test_python_proxy_is_not_mislabeled():
    report = example_report()
    report["provider"] = "python"
    assert "SERIALIZED INPUT PROXY" in render_roast_svg(report)


def test_cli_creates_new_svg_but_never_overwrites(tmp_path):
    source = tmp_path / "report.json"
    target = tmp_path / "roast.svg"
    source.write_text(json.dumps(example_report()), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["roast", str(source), "--output", str(target)])
    assert result.exit_code == 0, result.output
    assert target.read_text(encoding="utf-8").startswith("<svg")
    assert runner.invoke(app, ["roast", str(source), "--output", str(target)]).exit_code == 2


@pytest.mark.parametrize("payload", [{}, {"schema_version": 2}, {"schema_version": 1}])
def test_bad_report_rejected(payload):
    with pytest.raises(ValueError):
        render_roast_svg(payload)


def test_corrupted_json_exits_two(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text("{bad", encoding="utf-8")
    destination = tmp_path / "roast.svg"
    result = CliRunner().invoke(app, ["roast", str(source), "--output", str(destination)])
    assert result.exit_code == 2
    assert not destination.exists()
