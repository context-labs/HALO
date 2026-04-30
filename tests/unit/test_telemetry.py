"""Unit tests for engine.telemetry.setup_telemetry routing & shutdown."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "inference_catalyst_tracing",
    reason="telemetry extra not installed; install with `uv sync --extra telemetry`",
)

from engine.telemetry import setup_telemetry  # noqa: E402


def test_setup_returns_none_when_disabled(monkeypatch) -> None:
    # Even with the token set, enable=False short-circuits before any
    # routing decision is made.
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "irrelevant")

    handle = setup_telemetry(enable=False)

    assert handle is None


def test_setup_picks_catalyst_when_token_set(monkeypatch) -> None:
    """With CATALYST_OTLP_TOKEN set, setup_telemetry calls
    inference_catalyst_tracing.setup() and clears the openai-agents
    default trace processor list."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "tok")
    monkeypatch.setenv("CATALYST_OTLP_ENDPOINT", "https://example.invalid")

    catalyst_calls: list[None] = []

    class _StubCatalyst:
        def shutdown(self) -> None: ...

    def _stub_setup() -> _StubCatalyst:
        catalyst_calls.append(None)
        return _StubCatalyst()

    cleared: list[list] = []

    def _stub_set_trace_processors(procs: list) -> None:
        cleared.append(list(procs))

    monkeypatch.setattr(
        "inference_catalyst_tracing.setup",
        _stub_setup,
    )
    monkeypatch.setattr(
        "engine.telemetry.setup.set_trace_processors",
        _stub_set_trace_processors,
    )

    handle = setup_telemetry(enable=True, run_id="abc")

    assert handle is not None
    assert len(catalyst_calls) == 1, "inference_catalyst_tracing.setup must be called once"
    assert cleared == [[]], "openai-agents default processor list must be cleared"

    # Idempotency baseline: shutdown must not raise.
    handle.shutdown()
