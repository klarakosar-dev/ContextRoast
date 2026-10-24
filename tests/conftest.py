from __future__ import annotations

import socket
from pathlib import Path

import pytest

from contextroast.data import dataset_from_dict

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Tests must not access a real network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def raw_dataset():
    return {
        "version": 1,
        "name": "fixture",
        "instruction": "Return the number one.",
        "context_fields": ["required", "noise"],
        "cases": [
            {
                "id": "one",
                "input": "question",
                "context": {"required": "yes", "noise": "x" * 100},
                "expected": 1,
            }
        ],
    }


@pytest.fixture
def dataset(raw_dataset):
    return dataset_from_dict(raw_dataset)
