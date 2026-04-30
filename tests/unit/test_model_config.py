from __future__ import annotations

import pytest
from pydantic import ValidationError

from engine.model_config import ModelConfig, max_reasoning_effort_for_model


def test_defaults() -> None:
    cfg = ModelConfig(name="claude-opus-4-7")
    assert cfg.temperature is None
    assert cfg.maximum_output_tokens is None
    assert cfg.parallel_tool_calls is True
    assert cfg.reasoning_effort is None


def test_arbitrary_model_name_accepted() -> None:
    """Any string is valid — the configured OpenAI-compatible endpoint decides."""
    cfg = ModelConfig(name="meta-llama/Llama-3.1-70B-Instruct")
    assert cfg.name == "meta-llama/Llama-3.1-70B-Instruct"


def test_reasoning_effort_accepts_supported_values() -> None:
    for effort in ("none", "minimal", "low", "medium", "high", "xhigh"):
        cfg = ModelConfig(name="gpt-5", reasoning_effort=effort)
        assert cfg.reasoning_effort == effort


def test_reasoning_effort_rejects_unsupported_value() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(name="gpt-5", reasoning_effort="ultra")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name,expected",
    [
        # xhigh tier.
        ("gpt-5.1-codex-max", "xhigh"),
        ("gpt-5.4", "xhigh"),
        ("gpt-5.4-mini", "xhigh"),
        ("gpt-5.5", "xhigh"),
        ("gpt-5.5-pro", "xhigh"),
        ("GPT-5.5", "xhigh"),
        # high tier — other gpt-5 family + o-series.
        ("gpt-5", "high"),
        ("gpt-5-pro", "high"),
        ("gpt-5.1", "high"),
        ("gpt-5.1-mini", "high"),
        ("o1", "high"),
        ("o1-mini", "high"),
        ("o3", "high"),
        ("o3-mini", "high"),
        ("o4-mini", "high"),
        # Non-reasoning / unknown families.
        ("claude-opus-4-7", None),
        ("gpt-4o", None),
        ("gpt-4-turbo", None),
        ("meta-llama/Llama-3.1-70B-Instruct", None),
    ],
)
def test_max_reasoning_effort_for_model(name: str, expected: str | None) -> None:
    assert max_reasoning_effort_for_model(name) == expected


def test_effective_reasoning_effort_uses_explicit_override() -> None:
    cfg = ModelConfig(name="gpt-5.5", reasoning_effort="low")
    assert cfg.effective_reasoning_effort() == "low"


def test_effective_reasoning_effort_falls_back_to_model_max() -> None:
    assert ModelConfig(name="gpt-5.5").effective_reasoning_effort() == "xhigh"
    assert ModelConfig(name="gpt-5").effective_reasoning_effort() == "high"
    assert ModelConfig(name="o1-mini").effective_reasoning_effort() == "high"


def test_effective_reasoning_effort_none_for_unknown_model() -> None:
    assert ModelConfig(name="claude-opus-4-7").effective_reasoning_effort() is None


def test_to_sdk_model_settings_omits_reasoning_for_unknown_model() -> None:
    settings = ModelConfig(name="claude-opus-4-7").to_sdk_model_settings()
    assert settings.reasoning is None
    assert settings.temperature is None
    assert settings.max_tokens is None
    assert settings.parallel_tool_calls is True


def test_to_sdk_model_settings_defaults_to_model_max_for_reasoning_model() -> None:
    settings = ModelConfig(name="gpt-5.5").to_sdk_model_settings()
    assert settings.reasoning is not None
    assert settings.reasoning.effort == "xhigh"


def test_to_sdk_model_settings_forwards_explicit_override() -> None:
    settings = ModelConfig(
        name="gpt-5.5",
        temperature=0.2,
        maximum_output_tokens=1024,
        parallel_tool_calls=False,
        reasoning_effort="low",
    ).to_sdk_model_settings()
    assert settings.reasoning is not None
    assert settings.reasoning.effort == "low"
    assert settings.temperature == 0.2
    assert settings.max_tokens == 1024
    assert settings.parallel_tool_calls is False
