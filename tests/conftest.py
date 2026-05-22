from __future__ import annotations

import os
from pathlib import Path

import pytest
from agents import set_default_openai_api

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    """Set a placeholder ``OPENAI_API_KEY`` unless the run is targeting live tests.

    Engine eagerly constructs an ``AsyncOpenAI`` at run start, which raises if
    neither the env var nor ``model_provider.api_key`` is set. Non-live tests
    monkeypatch ``Runner.run_streamed`` so no real LLM call fires — they only
    need construction to succeed. Live tests rely on the unset env var as a
    skip signal, so we leave it alone when ``-m live`` is the selected marker.

    Also pins the OpenAI Agents SDK's default API to ``chat_completions``.
    The SDK defaults to the Responses API (``/v1/responses``), which many
    OpenAI-compatible providers (Inference.net, OpenRouter, …) do not serve;
    ``engine.main`` already pins it at run start, but live integration tests
    that directly invoke tool factories (``wired_tools``) never go through
    ``stream_engine_async`` and would otherwise inherit the Responses default.
    """
    markexpr = (config.getoption("-m", default="") or "").strip()
    if markexpr != "live":
        os.environ.setdefault("OPENAI_API_KEY", "test")
    set_default_openai_api("chat_completions")


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR
