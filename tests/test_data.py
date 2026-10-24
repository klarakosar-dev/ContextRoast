import copy
import json

import pytest
import yaml

from contextroast.data import dataset_from_dict, load_dataset
from contextroast.errors import DatasetError

from .conftest import ROOT


@pytest.mark.parametrize("name", ["support.json", "shipping.yaml"])
def test_real_examples_load(name):
    data = load_dataset(ROOT / "examples" / name)
    assert len(data.cases) >= 6
    assert len(data.context_fields) >= 4


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_json_yaml_equivalence(tmp_path, raw_dataset, suffix):
    path = tmp_path / ("data" + suffix)
    text = json.dumps(raw_dataset) if suffix == ".json" else yaml.safe_dump(raw_dataset)
    path.write_text(text)
    assert load_dataset(path) == dataset_from_dict(raw_dataset)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("version", True),
        ("version", 2),
        ("name", " "),
        ("instruction", None),
        ("context_fields", "required"),
        ("context_fields", ["required", "required"]),
        ("context_fields", [3]),
        ("cases", []),
        ("mock_task", 123),
    ],
)
def test_invalid_dataset_values(raw_dataset, key, value):
    raw_dataset[key] = value
    with pytest.raises(DatasetError):
        dataset_from_dict(raw_dataset)


@pytest.mark.parametrize("location", ["root", "case"])
def test_unknown_and_missing_keys(raw_dataset, location):
    target = raw_dataset if location == "root" else raw_dataset["cases"][0]
    target["typo"] = "bad"
    with pytest.raises(DatasetError, match="unknown keys"):
        dataset_from_dict(raw_dataset)
    del target["typo"]
    del target["name" if location == "root" else "expected"]
    with pytest.raises(DatasetError, match="missing keys"):
        dataset_from_dict(raw_dataset)


def test_duplicate_case_ids(raw_dataset):
    raw_dataset["cases"].append(copy.deepcopy(raw_dataset["cases"][0]))
    with pytest.raises(DatasetError, match="duplicate case id"):
        dataset_from_dict(raw_dataset)


@pytest.mark.parametrize("context", [{}, {"required": 1}, {"required": 1, "noise": 2, "extra": 3}])
def test_consistent_named_fields(raw_dataset, context):
    raw_dataset["cases"][0]["context"] = context
    with pytest.raises(DatasetError, match="exactly"):
        dataset_from_dict(raw_dataset)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {1: "numeric key"}, {"a": {1, 2}}])
def test_non_json_values_rejected(raw_dataset, value):
    raw_dataset["cases"][0]["expected"] = value
    with pytest.raises(DatasetError, match="finite JSON"):
        dataset_from_dict(raw_dataset)


@pytest.mark.parametrize("suffix,text", [(".json", '{"a": 1, "a": 2}'), (".yaml", "a: 1\na: 2\n")])
def test_duplicate_document_keys(tmp_path, suffix, text):
    path = tmp_path / ("duplicate" + suffix)
    path.write_text(text)
    with pytest.raises(DatasetError, match="duplicate key"):
        load_dataset(path)


@pytest.mark.parametrize(
    "suffix,text", [(".json", '{"x": NaN}'), (".yaml", "!!python/object:unsafe {}")]
)
def test_unsafe_or_nonfinite_document(tmp_path, suffix, text):
    path = tmp_path / ("unsafe" + suffix)
    path.write_text(text)
    with pytest.raises(DatasetError):
        load_dataset(path)


def test_missing_file_and_extension(tmp_path):
    with pytest.raises(DatasetError, match="Cannot load"):
        load_dataset(tmp_path / "missing.json")
    path = tmp_path / "data.txt"
    path.write_text("{}")
    with pytest.raises(DatasetError, match="must end"):
        load_dataset(path)


def test_no_context_allowed(raw_dataset):
    raw_dataset["context_fields"] = []
    raw_dataset["cases"][0]["context"] = {}
    assert dataset_from_dict(raw_dataset).context_fields == ()
