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
