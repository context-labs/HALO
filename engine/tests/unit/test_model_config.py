from __future__ import annotations

from engine.model_config import ModelConfig


def test_defaults() -> None:
    cfg = ModelConfig(name="claude-opus-4-7")
    assert cfg.temperature is None
    assert cfg.maximum_output_tokens is None
    assert cfg.parallel_tool_calls is True


def test_arbitrary_model_name_accepted() -> None:
    """Any string is valid — the configured OpenAI-compatible endpoint decides."""
    cfg = ModelConfig(name="meta-llama/Llama-3.1-70B-Instruct")
    assert cfg.name == "meta-llama/Llama-3.1-70B-Instruct"
