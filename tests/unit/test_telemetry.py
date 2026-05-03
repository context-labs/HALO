"""Unit tests for engine.telemetry.setup_telemetry & shutdown."""

from __future__ import annotations

from engine.telemetry import setup_telemetry


def test_setup_returns_none_when_disabled() -> None:
    handle = setup_telemetry(enable=False, run_id="unused")
    assert handle is None


def test_setup_attaches_local_processor(monkeypatch, tmp_path) -> None:
    """setup_telemetry attaches the InferenceOtlpFileProcessor to the
    openai-agents SDK at the path indicated by HALO_TELEMETRY_PATH."""
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

    from engine.telemetry.local_processor import attach_local_processor as real_attach

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
    monkeypatch.delenv("HALO_TELEMETRY_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    handle = setup_telemetry(enable=True, run_id="run123")

    assert handle is not None
    expected = tmp_path / "halo-telemetry-run123.jsonl"
    assert expected.exists(), f"expected {expected} to exist"

    handle.shutdown()


def test_clears_default_openai_dashboard_processor(monkeypatch, tmp_path) -> None:
    """setup_telemetry clears the openai-agents default trace processor list
    so HALO's own LLM activity does not leak to the OpenAI dashboard."""
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


def test_local_path_stamps_halo_run_id(monkeypatch, tmp_path) -> None:
    """The local backend includes halo.run_id in ExportContext.extra_resource_attributes."""
    monkeypatch.setenv("HALO_TELEMETRY_PATH", str(tmp_path / "out.jsonl"))
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)

    captured: list = []

    from engine.telemetry.local_processor import attach_local_processor as real_attach

    def _spy(**kwargs):
        captured.append(kwargs)
        return real_attach(**kwargs)

    monkeypatch.setattr("engine.telemetry.setup.attach_local_processor", _spy)

    handle = setup_telemetry(enable=True, run_id="run-xyz")
    assert handle is not None
    assert len(captured) == 1
    assert captured[0]["extra_resource_attributes"] == {"halo.run_id": "run-xyz"}

    handle.shutdown()


def test_shutdown_is_idempotent(monkeypatch, tmp_path) -> None:
    """Calling shutdown twice does not raise and only flushes the backend once."""
    monkeypatch.setenv("HALO_TELEMETRY_PATH", str(tmp_path / "out.jsonl"))
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)

    calls: list[None] = []

    class _StubProcessor:
        def shutdown(self) -> None:
            calls.append(None)

    monkeypatch.setattr(
        "engine.telemetry.setup.attach_local_processor",
        lambda **kwargs: _StubProcessor(),
    )

    handle = setup_telemetry(enable=True, run_id="x")
    assert handle is not None

    handle.shutdown()
    handle.shutdown()  # second call — must be a no-op

    assert len(calls) == 1, "backend.shutdown should be invoked exactly once"


def test_shutdown_swallows_backend_errors(monkeypatch, tmp_path) -> None:
    """A backend that raises during shutdown must not propagate the error;
    the engine's outer try/finally must not be masked."""
    monkeypatch.setenv("HALO_TELEMETRY_PATH", str(tmp_path / "out.jsonl"))
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)

    class _ExplodingProcessor:
        def shutdown(self) -> None:
            raise RuntimeError("flush kaboom")

    monkeypatch.setattr(
        "engine.telemetry.setup.attach_local_processor",
        lambda **kwargs: _ExplodingProcessor(),
    )

    handle = setup_telemetry(enable=True, run_id="x")
    assert handle is not None
    handle.shutdown()  # must NOT raise
