from __future__ import annotations

import pytest
from pydantic import ValidationError

from engine.model_config import AvailableModelName, ModelConfig


def test_defaults() -> None:
    cfg = ModelConfig(name="claude-opus-4-7")
    assert cfg.temperature is None
    assert cfg.maximum_output_tokens is None
    assert cfg.parallel_tool_calls is True


def test_model_name_literal_enforced() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(name="not-a-real-model")  # type: ignore[arg-type]


def test_all_names_listed() -> None:
    expected = {
        "claude-opus-4-7",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "gpt-5.4",
        "gpt-5.4-mini",
    }
    actual = set(AvailableModelName.__args__)  # type: ignore[attr-defined]
    assert actual == expected
