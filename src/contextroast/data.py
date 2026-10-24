"""Strict JSON/YAML loading. Expected answers never enter provider requests."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from contextroast.errors import DatasetError


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def parse_json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


class _UniqueLoader(yaml.SafeLoader):
    pass


def _yaml_mapping(loader: _UniqueLoader, node: yaml.MappingNode) -> dict[str, Any]:
    return _unique_pairs(loader.construct_pairs(node, deep=True))


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping)


def validate_json_value(value: Any, location: str) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            validate_json_value(item, location)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            validate_json_value(item, location)
        return
    raise DatasetError(f"{location} must contain finite JSON values and string object keys")


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{location} must be a non-empty string")
    return value


def _keys(value: Any, required: set[str], optional: set[str], location: str) -> None:
    if not isinstance(value, dict):
        raise DatasetError(f"{location} must be an object")
    missing, unknown = required - value.keys(), value.keys() - required - optional
    if missing:
        raise DatasetError(f"{location}: missing keys {sorted(missing)}")
    if unknown:
        raise DatasetError(f"{location}: unknown keys {sorted(unknown, key=str)}")


@dataclass(frozen=True)
class Case:
    id: str
    input: str
    context: dict[str, Any]
    expected: Any


@dataclass(frozen=True)
class Dataset:
    name: str
    instruction: str
    context_fields: tuple[str, ...]
    cases: tuple[Case, ...]
    mock_task: str | None = None


def dataset_from_dict(raw: Any) -> Dataset:
    _keys(
        raw, {"version", "name", "instruction", "context_fields", "cases"}, {"mock_task"}, "dataset"
    )
    if type(raw["version"]) is not int or raw["version"] != 1:
        raise DatasetError("dataset.version must be 1")
    name = _text(raw["name"], "dataset.name")
    instruction = _text(raw["instruction"], "dataset.instruction")
    fields = raw["context_fields"]
    if not isinstance(fields, list):
        raise DatasetError("context_fields must be a list of unique field names (it may be empty)")
    for field in fields:
        _text(field, "context_fields entry")
    if len(set(fields)) != len(fields):
        raise DatasetError("context_fields must be unique")
    mock_task = raw.get("mock_task")
    if mock_task is not None:
        _text(mock_task, "mock_task")
    if not isinstance(raw["cases"], list) or not raw["cases"]:
        raise DatasetError("cases must be a non-empty list")
    cases, ids = [], set()
    for index, item in enumerate(raw["cases"]):
        where = f"cases[{index}]"
        _keys(item, {"id", "input", "context", "expected"}, set(), where)
        case_id = _text(item["id"], f"{where}.id")
        if case_id in ids:
            raise DatasetError(f"duplicate case id: {case_id!r}")
        ids.add(case_id)
        case_input = _text(item["input"], f"{where}.input")
        context = item["context"]
        if not isinstance(context, dict) or set(context) != set(fields):
            raise DatasetError(f"{where}.context must have exactly the declared context_fields")
        validate_json_value(context, f"{where}.context")
        validate_json_value(item["expected"], f"{where}.expected")
        cases.append(Case(case_id, case_input, context, item["expected"]))
    return Dataset(name, instruction, tuple(fields), tuple(cases), mock_task)


def load_dataset(path: Path) -> Dataset:
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            raw = parse_json(text)
        elif path.suffix.lower() in (".yaml", ".yml"):
            raw = yaml.load(text, Loader=_UniqueLoader)
        else:
            raise DatasetError("dataset file must end in .json, .yaml, or .yml")
        return dataset_from_dict(raw)
    except DatasetError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError, yaml.YAMLError) as exc:
        raise DatasetError(f"Cannot load dataset {path}: {exc}") from exc
