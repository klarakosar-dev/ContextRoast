"""Explicitly consented loading of trusted, synchronous Python agent functions."""

from __future__ import annotations

import copy
import importlib
import inspect
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contextroast.data import canonical_json, validate_json_value
from contextroast.errors import DatasetError, ProviderError
from contextroast.providers import Request, Response


@dataclass(frozen=True)
class AgentCase:
    """The public case view: no expected answer or unabridged context is exposed."""

    id: str | None
    input: str
    instruction: str


@dataclass(frozen=True)
class AgentResult:
    """Optional aggregate model usage across ALL calls inside one invocation."""

    prediction: Any
    input_tokens: int | None = None
    output_tokens: int | None = None


class PythonProvider:
    name = "python"

    def __init__(self, entrypoint: str, *, allow_paid: bool = False) -> None:
        # Module import itself executes code and may incur charges. Consent comes first.
        if allow_paid is not True:
            raise ProviderError(
                "Python entrypoints execute trusted local code, including imports, and may incur "
                "API charges. Explicit --allow-paid consent is required before loading."
            )
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", entrypoint):
            raise ProviderError(
                "--entrypoint must be module:function (no file paths or expressions)"
            )
        module_name, function_name = entrypoint.split(":")
        # Match `python -m` behavior for functions living in the working checkout.
        cwd = str(Path.cwd())
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        try:
            module = importlib.import_module(module_name)
            function = getattr(module, function_name)
        except Exception as exc:
            raise ProviderError(
                f"Cannot load Python entrypoint ({type(exc).__name__}). "
                "Check its import path and dependencies; import-time code may already have run."
            ) from None
        if not callable(function) or inspect.iscoroutinefunction(function):
            raise ProviderError("Python entrypoint must be a synchronous callable")
        try:
            inspect.signature(function).bind(AgentCase(None, "", ""), {})
        except (TypeError, ValueError) as exc:
            raise ProviderError(
                "Python callable must accept (case, context) as positional arguments"
            ) from exc
        self.model = entrypoint
        self._function = function
        self.invocations = 0

    def generate(self, request: Request) -> Response:
        case = AgentCase(request.case_id, request.input, request.instruction)
        context = copy.deepcopy(request.context)
        self.invocations += 1
        try:
            result = self._function(case, context)
        except Exception as exc:
            # Never put user exception messages (which may contain secrets) in reports.
            return Response("", error=f"python_exception:{type(exc).__name__}")
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            return Response("", error="python_async_result_unsupported")
        input_tokens = output_tokens = None
        if isinstance(result, AgentResult):
            input_tokens, output_tokens = result.input_tokens, result.output_tokens
            if (input_tokens, output_tokens) != (None, None) and any(
                type(value) is not int or value < 0 for value in (input_tokens, output_tokens)
            ):
                return Response("", error="invalid_python_usage")
            result = result.prediction
        try:
            validate_json_value(result, "Python prediction")
            text = canonical_json({"answer": result})
        except (DatasetError, ValueError, TypeError, RecursionError):
            return Response("", input_tokens, output_tokens, "invalid_python_prediction")
        return Response(text, input_tokens, output_tokens)
