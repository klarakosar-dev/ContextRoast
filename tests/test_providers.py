import json

import httpx
import pytest

from contextroast.data import canonical_json, load_dataset
from contextroast.errors import ProviderError
from contextroast.providers import (
    MockProvider,
    OpenAICompatibleProvider,
    Request,
    Response,
    estimate_input_tokens,
    estimate_tokens,
    score_response,
)

from .conftest import ROOT


@pytest.mark.parametrize(
    "text,expected,correct,error",
    [
        (' {"answer": {"b": 2, "a": [true, null]}} ', {"a": [True, None], "b": 2}, True, None),
        ('{"answer": [2, 1]}', [1, 2], False, None),
        ('{"answer": "YES"}', "yes", False, None),
        ('{"answer": "ok "}', "ok", False, None),
        ('{"answer": true}', 1, False, None),
        ('{"answer": 1.0}', 1, False, None),
        ('{"answer": {"a": 1, "b": 2}}', {"a": 1}, False, None),
        ('{"answer": null}', None, True, None),
        ('{"answer": 1, "extra": 2}', 1, False, "expected_answer_wrapper"),
        ('{"value": 1}', 1, False, "expected_answer_wrapper"),
        ("[]", 1, False, "expected_answer_wrapper"),
        ('```json\n{"answer": 1}\n```', 1, False, "invalid_json"),
        ('{"answer": NaN}', 1, False, "invalid_json"),
        ('{"answer": 1e999}', 1, False, "invalid_json"),
        ('{"answer": 1, "answer": 2}', 2, False, "invalid_json"),
    ],
)
def test_exact_structured_scoring(text, expected, correct, error):
    actual, _, actual_error = score_response(Response(text), expected)
    assert actual == correct
    assert actual_error == error


def test_error_response_cannot_match():
    assert score_response(Response('{"answer": 1}', error="incomplete_response"), 1) == (
        False,
        None,
        "incomplete_response",
    )


def test_token_estimator_includes_instructions_and_utf8():
    assert estimate_tokens("hello") == 2
    assert estimate_tokens("🙂") == 1
    assert estimate_tokens("") == 0
    request = Request("instruction", "question", {"field_name": "value"})
    assert estimate_input_tokens(request) > estimate_tokens(canonical_json(request.context))


def test_mock_answers_do_not_depend_on_expected_answers():
    data = load_dataset(ROOT / "examples/support.json")
    case = data.cases[0]
    request = Request(data.instruction, case.input, case.context)
    response = MockProvider(data.mock_task).generate(request)
    assert score_response(response, case.expected)[0]
    assert not score_response(response, {"this": "was not derived from expected"})[0]
    assert not hasattr(request, "expected")


def test_mock_missing_context_and_task_errors():
    with pytest.raises(ProviderError, match="mock_task"):
        MockProvider(None)
    response = MockProvider("shipping-quote").generate(Request("", "invalid", {}))
    assert json.loads(response.text)["answer"] == {"error": "missing_or_invalid_context"}


def api_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def envelope(content='{"answer": 1}', usage=None, finish_reason="stop", refusal=None):
    return {
        "choices": [
            {"message": {"content": content, "refusal": refusal}, "finish_reason": finish_reason}
        ],
        "usage": usage,
    }


def test_api_requires_explicit_opt_in_before_client_creation():
    with pytest.raises(ProviderError, match="allow-paid"):
        OpenAICompatibleProvider(model="test", api_key="secret")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": ""},
        {"api_key": ""},
        {"base_url": "http://external.example/v1"},
        {"base_url": "https://user:secret@example.test/v1"},
        {"base_url": "https://example.test/v1?key=secret"},
        {"timeout": float("nan")},
        {"max_output_tokens": 0},
    ],
)
def test_invalid_api_configuration(kwargs):
    params = {"model": "test", "api_key": "secret", "allow_paid": True} | kwargs
    with pytest.raises(ProviderError):
        OpenAICompatibleProvider(**params)


def test_api_payload_usage_and_no_answer_leakage():
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer test-secret"
        payload = json.loads(request.content)
        assert payload["model"] == "test-model"
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["max_completion_tokens"] == 99
        assert "expected" not in request.content.decode()
        assert "case_id" not in request.content.decode()
        assert "JSON" in payload["messages"][0]["content"]
        return httpx.Response(
            200, json=envelope(usage={"prompt_tokens": 123, "completion_tokens": 7})
        )

    with api_client(handler) as client:
        provider = OpenAICompatibleProvider(
            model="test-model",
            api_key="test-secret",
            allow_paid=True,
            base_url="http://localhost:8080/v1/",
            max_output_tokens=99,
            client=client,
        )
        response = provider.generate(Request("Return a number", "question", {"a": 1}))
        assert response.input_tokens == 123
        assert response.output_tokens == 7
        provider.close()  # injected clients remain owned by their caller
        assert not client.is_closed
    assert len(requests) == 1
    assert str(requests[0].url) == "http://localhost:8080/v1/chat/completions"


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": 9},
        {"prompt_tokens": True, "completion_tokens": 4},
        {"prompt_tokens": -1, "completion_tokens": 4},
    ],
)
def test_invalid_or_partial_usage_stays_unknown(usage):
    with api_client(lambda request: httpx.Response(200, json=envelope(usage=usage))) as client:
        provider = OpenAICompatibleProvider(
            model="test", api_key="key", allow_paid=True, client=client
        )
        result = provider.generate(Request("", "", {}))
    assert result.input_tokens is result.output_tokens is None


@pytest.mark.parametrize(
    "reason,refusal,error",
    [("length", None, "incomplete_response"), ("stop", "no", "provider_refusal")],
)
def test_api_output_failure_preserves_usage(reason, refusal, error):
    with api_client(
        lambda request: httpx.Response(
            200,
            json=envelope(
                usage={"prompt_tokens": 10, "completion_tokens": 3},
                finish_reason=reason,
                refusal=refusal,
            ),
        )
    ) as client:
        result = OpenAICompatibleProvider(
            model="test", api_key="key", allow_paid=True, client=client
        ).generate(Request("", "", {}))
    assert result.error == error
    assert result.input_tokens == 10


@pytest.mark.parametrize("status", [302, 401, 429, 500])
def test_api_http_failure_no_retry_no_secret(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="private-provider-body-secret")

    with api_client(handler) as client:
        provider = OpenAICompatibleProvider(
            model="test", api_key="key", allow_paid=True, client=client
        )
        with pytest.raises(ProviderError, match=f"HTTP {status}") as caught:
            provider.generate(Request("", "", {}))
    assert "private-provider" not in str(caught.value)
    assert len(calls) == 1


def test_api_transport_failure():
    def handler(request):
        raise httpx.ReadTimeout("private-secret")

    with api_client(handler) as client:
        provider = OpenAICompatibleProvider(
            model="test", api_key="key", allow_paid=True, client=client
        )
        with pytest.raises(ProviderError, match="No retries") as caught:
            provider.generate(Request("", "", {}))
    assert "private-secret" not in str(caught.value)


@pytest.mark.parametrize(
    "body", [{}, {"choices": []}, {"choices": [{"message": {"content": [1]}}]}]
)
def test_malformed_api_envelopes(body):
    with api_client(lambda request: httpx.Response(200, json=body)) as client:
        provider = OpenAICompatibleProvider(
            model="test", api_key="key", allow_paid=True, client=client
        )
        with pytest.raises(ProviderError, match="Malformed"):
            provider.generate(Request("", "", {}))
