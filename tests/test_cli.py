import json

import httpx
import pytest
from typer.testing import CliRunner

from contextroast.cli import app
from contextroast.errors import ProviderError
from contextroast.providers import OpenAICompatibleProvider

from .conftest import ROOT

runner = CliRunner()


def test_cli_help_and_version():
    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert "0.2.2" in runner.invoke(app, ["--version"]).output
    assert runner.invoke(app, ["run", "--help"]).exit_code == 0


def test_validate_no_provider(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Validation must not instantiate a provider")

    monkeypatch.setattr("contextroast.cli.MockProvider", forbidden)
    result = runner.invoke(app, ["validate", str(ROOT / "examples/support.json")])
    assert result.exit_code == 0
    assert "8 cases, 5 context fields" in result.output
    assert runner.invoke(app, ["validate", str(tmp_path / "missing.json")]).exit_code == 2


def test_default_mock_offline_even_with_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "this-must-not-trigger-api")

    def forbidden(*args, **kwargs):
        raise AssertionError("Default run must not construct HTTP clients")

    monkeypatch.setattr(httpx, "Client", forbidden)
    result = runner.invoke(
        app, ["run", str(ROOT / "examples/support.json"), "--output-dir", str(tmp_path / "out")]
    )
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["provider"] == "mock"
    assert report["removed_fields"] == ["customer_notes", "recent_orders", "brand_voice"]
    assert report["projection"]["gross_savings_usd"] is None
    assert (tmp_path / "out/report.md").exists()
    assert "100.00% -> 100.00%" in result.output


def test_api_requires_flag_before_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Must require opt-in before creating HTTP client")

    monkeypatch.setattr(httpx, "Client", forbidden)
    result = runner.invoke(
        app,
        [
            "run",
            str(ROOT / "examples/support.json"),
            "--provider",
            "api",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 2
    assert "--allow-paid" in result.output
    assert not (tmp_path / "out/report.json").exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--tolerance", "-0.1"],
        ["--tolerance", "nan"],
        ["--input-price", "NaN"],
        ["--projected-requests", "-1"],
    ],
)
def test_invalid_numeric_options(tmp_path, extra):
    result = runner.invoke(
        app,
        ["run", str(ROOT / "examples/support.json"), "--output-dir", str(tmp_path / "out"), *extra],
    )
    assert result.exit_code == 2


def test_no_accidental_report_overwrite(tmp_path):
    (tmp_path / "report.json").write_text("precious")
    result = runner.invoke(
        app, ["run", str(ROOT / "examples/support.json"), "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 2
    assert "Refusing to overwrite" in result.output
    assert (tmp_path / "report.json").read_text() == "precious"


def test_api_end_to_end_with_fake_transport(tmp_path, monkeypatch):
    sent = []

    def handler(request):
        payload = json.loads(request.content)
        sent.append(payload)
        context = json.loads(payload["messages"][1]["content"])["context"]
        answer = 1 if "required" in context else 0
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": json.dumps({"answer": answer})},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100 + 10 * len(context), "completion_tokens": 5},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    class TestAdapter(OpenAICompatibleProvider):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, client=client)

    monkeypatch.setattr("contextroast.cli.OpenAICompatibleProvider", TestAdapter)
    monkeypatch.setenv("TEST_DIET_KEY", "secret-test-key")
    dataset_path = tmp_path / "data.json"
    dataset_path.write_text(
        json.dumps(
            {
                "version": 1,
                "name": "api-fixture",
                "instruction": "Return one with required context.",
                "context_fields": ["required", "noise"],
                "cases": [
                    {
                        "id": "NEVER-SEND-ID",
                        "input": "question",
                        "context": {"required": 1, "noise": "padding"},
                        "expected": 1,
                    }
                ],
            }
        )
    )
    result = runner.invoke(
        app,
        [
            "run",
            str(dataset_path),
            "--provider",
            "api",
            "--allow-paid",
            "--model",
            "fake-model",
            "--api-key-env",
            "TEST_DIET_KEY",
            "--input-price",
            "2",
            "--output-price",
            "8",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    client.close()
    assert result.exit_code == 0, result.output
    report_text = (tmp_path / "out/report.json").read_text()
    report = json.loads(report_text)
    assert len(sent) == report["evaluation"]["usage"]["calls"] == 4
    assert "secret-test-key" not in report_text
    assert "NEVER-SEND-ID" not in json.dumps(sent)
    assert "up to 4 calls" in result.output
    assert report["evaluation"]["execution_cost_basis"] == "provider_reported"
    assert report["kept_fields"] == ["required"]


def test_provider_failure_returns_error_without_partial_report(tmp_path, monkeypatch):
    class BrokenMock:
        def __init__(self, task):
            self.name, self.model = "mock", task

        def generate(self, request):
            raise ProviderError("synthetic failure")

    monkeypatch.setattr("contextroast.cli.MockProvider", BrokenMock)
    result = runner.invoke(
        app, ["run", str(ROOT / "examples/support.json"), "--output-dir", str(tmp_path / "out")]
    )
    assert result.exit_code == 2
    assert "synthetic failure" in result.output
    assert not (tmp_path / "out/report.json").exists()
