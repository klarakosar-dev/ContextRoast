"""Providers receive only instructions, input, and the selected context."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

from contextroast.data import canonical_json, parse_json
from contextroast.errors import ProviderError


@dataclass(frozen=True)
class Request:
    instruction: str
    input: str
    context: dict[str, Any]
    case_id: str | None = None

    def messages(self) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": self.instruction + '\nReturn only JSON: {"answer": <your answer>}.',
            },
            {
                "role": "user",
                "content": canonical_json({"input": self.input, "context": self.context}),
            },
        ]


@dataclass(frozen=True)
class Response:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


class Provider(Protocol):
    name: str
    model: str

    def generate(self, request: Request) -> Response: ...


def estimate_tokens(text: str) -> int:
    """Deliberately simple heuristic, not a model tokenizer or billing measurement."""
    return math.ceil(len(text.encode("utf-8")) / 4)


def estimate_input_tokens(request: Request) -> int:
    # Includes instruction, serialized context/input, and approximate message framing.
    return sum(estimate_tokens(message["content"]) + 4 for message in request.messages()) + 2


class MockProvider:
    name = "mock"

    def __init__(self, task: str | None) -> None:
        if task not in ("support-routing", "shipping-quote"):
            raise ProviderError(
                "mock_task must be 'support-routing' or 'shipping-quote' for mock runs"
            )
        self.model = task

    def generate(self, request: Request) -> Response:
        try:
            answer = (
                self._support(request)
                if self.model == "support-routing"
                else self._shipping(request)
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            answer = {"error": "missing_or_invalid_context"}
        return Response(canonical_json({"answer": answer}))

    @staticmethod
    def _support(request: Request) -> dict[str, str]:
        policy = request.context["routing_policy"]
        account = request.context["account"]
        message = request.input.lower()
        queue = policy["default_queue"]
        for route in policy["routes"]:
            if any(keyword.lower() in message for keyword in route["keywords"]):
                queue = route["queue"]
                break
        priority = "high" if account["tier"] in policy["priority_tiers"] else "normal"
        return {"queue": queue, "priority": priority}

    @staticmethod
    def _shipping(request: Request) -> dict[str, Any]:
        match = re.fullmatch(r"Ship ([1-9][0-9]*) kg to ([A-Z]{2})\.", request.input)
        if not match:
            raise ValueError("unsupported shipping input")
        weight, country = match.groups()
        tariff = request.context["shipping_rates"][country]
        return {
            "amount_cents": tariff["base_cents"] + int(weight) * tariff["per_kg_cents"],
            "currency": request.context["currency"],
        }


class OpenAICompatibleProvider:
    """One Chat Completions request per case; no implicit retries or network at import."""

    name = "api"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        allow_paid: bool = False,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        max_output_tokens: int = 512,
        client: Any = None,
    ) -> None:
        if allow_paid is not True:
            raise ProviderError("API calls require explicit --allow-paid consent")
        if not model.strip() or not api_key.strip():
            raise ProviderError(
                "API provider requires --model and a non-empty API key environment variable"
            )
        parsed = urlsplit(base_url)
        if (
            not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.scheme not in ("https", "http")
            or (
                parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1")
            )
        ):
            raise ProviderError("base URL must be HTTPS (HTTP is allowed only for localhost)")
        if (
            not math.isfinite(timeout)
            or timeout <= 0
            or type(max_output_tokens) is not int
            or max_output_tokens <= 0
        ):
            raise ProviderError(
                "timeout must be positive and finite; max output tokens must be a positive integer"
            )
        try:
            import httpx
        except ImportError as exc:
            raise ProviderError("Install the API extra: pip install 'contextroast[api]'") from exc
        self.model = model
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._timeout = timeout
        self._max_output_tokens = max_output_tokens
        self._client = client if client is not None else httpx.Client(follow_redirects=False)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def generate(self, request: Request) -> Response:
        import httpx

        try:
            response = self._client.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self.model,
                    "messages": request.messages(),
                    "response_format": {"type": "json_object"},
                    "max_completion_tokens": self._max_output_tokens,
                },
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            # Avoid echoing credentials, context, or raw provider error bodies.
            raise ProviderError(
                "API transport failed. No retries were made; "
                "earlier calls may have incurred charges."
            ) from exc
        if response.status_code != 200:
            raise ProviderError(
                f"API returned HTTP {response.status_code}. No retries were made; "
                "earlier calls may have incurred charges."
            )
        try:
            body = response.json()
            usage = body.get("usage") or {}
            prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
            if any(type(number) is not int or number < 0 for number in (prompt, completion)):
                prompt = completion = None
            choice = body["choices"][0]
            message = choice["message"]
            text = message.get("content") or ""
            if not isinstance(text, str):
                raise ValueError("content must be text")
            error = None
            if message.get("refusal"):
                error = "provider_refusal"
            elif choice.get("finish_reason") != "stop":
                error = "incomplete_response"
            return Response(text, prompt, completion, error)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            raise ProviderError(
                "Malformed API response envelope; charges may have been incurred."
            ) from exc


def score_response(response: Response, expected: Any) -> tuple[bool, Any, str | None]:
    if response.error:
        return False, None, response.error
    try:
        parsed = parse_json(response.text)
        if not isinstance(parsed, dict) or set(parsed) != {"answer"}:
            return False, None, "expected_answer_wrapper"
        answer = parsed["answer"]
        return canonical_json(answer) == canonical_json(expected), answer, None
    except (ValueError, TypeError, RecursionError):
        return False, None, "invalid_json"
