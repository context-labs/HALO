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


def test_setup_picks_local_when_token_unset(monkeypatch, tmp_path) -> None:
    """Without CATALYST_OTLP_TOKEN, setup_telemetry attaches the
    InferenceOtlpFileProcessor to the openai-agents SDK and writes JSONL
    to HALO_TELEMETRY_PATH (or the run-id default)."""
    monkeypatch.delenv("CATALYST_OTLP_TOKEN", raising=False)
    out_path = tmp_path / "halo-telemetry.jsonl"
    monkeypatch.setenv("HALO_TELEMETRY_PATH", str(out_path))

    cleared: list[list] = []

    def _stub_set_trace_processors(procs: list) -> None:
        cleared.append(list(procs))

    monkeypatch.setattr(
        "engine.telemetry.setup.set_trace_processors",
        _stub_set_trace_processors,
    )

    attached: list = []

    real_attach = __import__(
        "engine.telemetry.local_processor",
        fromlist=["attach_local_processor"],
    ).attach_local_processor

    def _spy_attach(**kwargs):
        attached.append(kwargs)
        return real_attach(**kwargs)

    monkeypatch.setattr(
        "engine.telemetry.setup.attach_local_processor",
        _spy_attach,
    )

    handle = setup_telemetry(enable=True, run_id="abc")

    assert handle is not None
    assert cleared == [[]]
    assert len(attached) == 1
    assert attached[0]["path"] == str(out_path)
    assert attached[0]["service_name"] == "halo-engine"

    handle.shutdown()
    # File must exist (re-open to check; the InferenceOtlpFileProcessor opens
    # the file in __init__ even if no spans have been written yet).
    assert out_path.exists()


def test_local_path_default_uses_run_id(monkeypatch, tmp_path) -> None:
    """When HALO_TELEMETRY_PATH is unset, the local file is named
    halo-telemetry-{run_id}.jsonl in the current working directory."""
    monkeypatch.delenv("CATALYST_OTLP_TOKEN", raising=False)
    monkeypatch.delenv("HALO_TELEMETRY_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    handle = setup_telemetry(enable=True, run_id="run123")

    assert handle is not None
    expected = tmp_path / "halo-telemetry-run123.jsonl"
    assert expected.exists(), f"expected {expected} to exist"

    handle.shutdown()


def test_shutdown_is_idempotent(monkeypatch) -> None:
    """Calling shutdown twice does not raise and only flushes the backend once."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "tok")

    calls: list[None] = []

    class _StubCatalyst:
        def shutdown(self) -> None:
            calls.append(None)

    monkeypatch.setattr(
        "inference_catalyst_tracing.setup",
        lambda: _StubCatalyst(),
    )
    monkeypatch.setattr(
        "engine.telemetry.setup.set_trace_processors",
        lambda procs: None,
    )

    handle = setup_telemetry(enable=True, run_id="x")
    assert handle is not None

    handle.shutdown()
    handle.shutdown()  # second call — must be a no-op

    assert len(calls) == 1, "backend.shutdown should be invoked exactly once"


def test_shutdown_swallows_backend_errors(monkeypatch) -> None:
    """A backend that raises during shutdown must not propagate the error;
    the engine's outer try/finally must not be masked."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "tok")

    class _ExplodingCatalyst:
        def shutdown(self) -> None:
            raise RuntimeError("flush kaboom")

    monkeypatch.setattr(
        "inference_catalyst_tracing.setup",
        lambda: _ExplodingCatalyst(),
    )
    monkeypatch.setattr(
        "engine.telemetry.setup.set_trace_processors",
        lambda procs: None,
    )

    handle = setup_telemetry(enable=True, run_id="x")
    assert handle is not None
    handle.shutdown()  # must NOT raise


def test_clears_default_openai_dashboard_processor_on_local_path(
    monkeypatch, tmp_path
) -> None:
    """Local backend also clears the openai-agents default processor."""
    monkeypatch.delenv("CATALYST_OTLP_TOKEN", raising=False)
    monkeypatch.setenv("HALO_TELEMETRY_PATH", str(tmp_path / "out.jsonl"))

    cleared: list[list] = []

    def _stub_set_trace_processors(procs: list) -> None:
        cleared.append(list(procs))

    monkeypatch.setattr(
        "engine.telemetry.setup.set_trace_processors",
        _stub_set_trace_processors,
    )

    handle = setup_telemetry(enable=True, run_id="x")
    assert handle is not None
    assert cleared == [[]]
    handle.shutdown()
