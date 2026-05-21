"""Unit tests for engine.telemetry.setup_telemetry & shutdown."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from engine.telemetry import resolve_run_id, setup_telemetry


@pytest.fixture(autouse=True)
def _clear_catalyst_env(monkeypatch) -> None:
    """Local-path tests rely on ``CATALYST_OTLP_TOKEN`` being unset; a stray
    value in a developer's shell would silently route them through
    ``_setup_catalyst``. Catalyst-path tests below re-set this themselves.

    Also clears every ``CATALYST_TRACING_*`` env (the generic passthrough
    in ``_setup_catalyst`` would otherwise turn any developer-set var
    into a ``halo.*`` resource attribute and break deterministic
    assertions), plus the explicit catalyst service config envs."""
    monkeypatch.delenv("CATALYST_OTLP_TOKEN", raising=False)
    monkeypatch.delenv("CATALYST_SERVICE_NAME", raising=False)
    monkeypatch.delenv("CATALYST_SERVICE_VERSION", raising=False)
    monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
    for name in [k for k in os.environ if k.startswith("CATALYST_TRACING_")]:
        monkeypatch.delenv(name, raising=False)


def test_setup_returns_none_when_disabled(monkeypatch) -> None:
    cleared: list[list] = []

    monkeypatch.setattr(
        "engine.telemetry.setup.set_trace_processors",
        lambda procs: cleared.append(list(procs)),
    )

    handle = setup_telemetry(enable=False, run_id="unused")

    assert handle is None
    assert cleared == [[]]


def test_halo_agent_span_passes_conversation_id_as_session_id(monkeypatch) -> None:
    from engine.telemetry import tracing

    captured: dict[str, object] = {}

    class _FakeSpan:
        pass

    @contextmanager
    def _fake_agent_span(
        tracer: object,
        *,
        span_name: str,
        system: str,
        agent_id: str | None,
        session_id: str | None,
    ) -> Iterator[_FakeSpan]:
        captured["tracer"] = tracer
        captured["span_name"] = span_name
        captured["system"] = system
        captured["agent_id"] = agent_id
        captured["session_id"] = session_id
        yield _FakeSpan()

    monkeypatch.setenv("CATALYST_TRACING_CONVERSATION_ID", "  conv-123  ")
    monkeypatch.setattr(tracing.trace, "get_tracer", lambda name: f"tracer:{name}")
    monkeypatch.setattr(tracing, "agent_span", _fake_agent_span)

    with tracing.halo_agent_span(span_name="halo-root.run", agent_id="halo"):
        pass

    assert captured == {
        "tracer": "tracer:halo-engine",
        "span_name": "halo-root.run",
        "system": "openai",
        "agent_id": "halo",
        "session_id": "conv-123",
    }


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
    """The local backend includes halo.run.id in ExportContext.extra_resource_attributes.

    Dotted key matches the convention the catalyst-side runtime uses
    (``halo.run.id`` in ``halo/src/transport_client/otel_logger.py``)
    so dashboard filters work uniformly across the two emitters."""
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
    assert captured[0]["extra_resource_attributes"] == {"halo.run.id": "run-xyz"}

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


# ---------------------------------------------------------------------------
# Catalyst-path tests: stub `inference_catalyst_tracing.setup` so no real
# OTLP traffic leaves the test process.
# ---------------------------------------------------------------------------


class _StubCatalystBackend:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _install_stub_catalyst(monkeypatch) -> list[_StubCatalystBackend]:
    """Replace the ``catalyst_setup`` reference bound in
    ``engine.telemetry.setup`` with a stub. Returns the list the stub
    appends backends to (one entry per setup() call).

    The stub mirrors the real ``inference_catalyst_tracing.setup``
    keyword-only signature (rather than swallowing ``**kwargs``) so
    that if ``_setup_catalyst`` ever starts passing a kwarg the real
    function doesn't accept, the test fails loudly here instead of
    silently masking the bug. Any new kwarg actually supported by the
    real function should be added here as well.
    """
    backends: list[_StubCatalystBackend] = []

    def _stub_setup(
        *,
        endpoint: str | None = None,
        token: str | None = None,
        service_name: str | None = None,
        service_version: str | None = None,
        debug: bool | None = None,
        batching: str | None = None,
    ) -> _StubCatalystBackend:
        be = _StubCatalystBackend()
        backends.append(be)
        return be

    monkeypatch.setattr("engine.telemetry.setup.catalyst_setup", _stub_setup)
    return backends


def test_setup_picks_catalyst_when_token_set(monkeypatch) -> None:
    """When CATALYST_OTLP_TOKEN is set, setup_telemetry routes to the
    Catalyst backend and does not touch the local file path."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.delenv("HALO_TELEMETRY_PATH", raising=False)
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)

    backends = _install_stub_catalyst(monkeypatch)

    # Mirror attach_local_processor's keyword-only signature so any future
    # signature change (or accidental positional call from _setup_local)
    # surfaces here as a TypeError instead of being silently absorbed by
    # **kwargs. Same contract as the catalyst stub above.
    local_calls: list = []

    def _stub_attach_local(
        *,
        path: str,
        service_name: str,
        project_id: str,
        extra_resource_attributes=None,
    ):
        local_calls.append(
            {
                "path": path,
                "service_name": service_name,
                "project_id": project_id,
                "extra_resource_attributes": extra_resource_attributes,
            }
        )

    monkeypatch.setattr(
        "engine.telemetry.setup.attach_local_processor",
        _stub_attach_local,
    )

    handle = setup_telemetry(enable=True, run_id="run-cat")

    assert handle is not None
    assert len(backends) == 1, "Catalyst setup() must be called exactly once"
    assert local_calls == [], "local backend must not be touched on Catalyst path"

    handle.shutdown()
    assert backends[0].shutdown_calls == 1


def test_catalyst_path_sets_service_name_default(monkeypatch) -> None:
    """When the user has not set CATALYST_SERVICE_NAME, _setup_catalyst
    defaults it to 'halo-engine' before calling setup()."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.delenv("CATALYST_SERVICE_NAME", raising=False)
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="run-default")
    assert handle is not None
    assert os.environ.get("CATALYST_SERVICE_NAME") == "halo-engine"
    handle.shutdown()


def test_catalyst_path_respects_user_service_name(monkeypatch) -> None:
    """A user-set CATALYST_SERVICE_NAME must NOT be overwritten."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setenv("CATALYST_SERVICE_NAME", "my-service")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="run-user")
    assert handle is not None
    assert os.environ.get("CATALYST_SERVICE_NAME") == "my-service"
    handle.shutdown()


def test_catalyst_path_stamps_halo_run_id(monkeypatch) -> None:
    """halo.run.id is appended to OTEL_RESOURCE_ATTRIBUTES; pre-existing
    attributes must be preserved."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "deployment.environment=dev")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="run-abc")
    assert handle is not None

    attrs = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    assert "deployment.environment=dev" in attrs
    assert "halo.run.id=run-abc" in attrs
    handle.shutdown()


def test_catalyst_path_replaces_prior_halo_run_id(monkeypatch) -> None:
    """Repeated calls to setup_telemetry in the same process must not
    accumulate stale halo.run.id entries in OTEL_RESOURCE_ATTRIBUTES."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "deployment.environment=dev")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    h1 = setup_telemetry(enable=True, run_id="run-aaa")
    assert h1 is not None
    h2 = setup_telemetry(enable=True, run_id="run-bbb")
    assert h2 is not None

    attrs = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    halo_tokens = [t for t in attrs.split(",") if t.strip().startswith("halo.run.id=")]
    assert halo_tokens == ["halo.run.id=run-bbb"], f"expected only the latest run_id, got {attrs!r}"
    assert "deployment.environment=dev" in attrs

    h1.shutdown()
    h2.shutdown()


# ---------------------------------------------------------------------------
# Catalyst-deployed identity: CATALYST_TRACING_* resource attribute
# passthrough (lowercased + ``_`` → ``.``), halo.engine.version stamping,
# constant service.name. Together these form the contract HALO presents
# to the Catalyst-launched Modal sandbox for trace identity.
# ---------------------------------------------------------------------------


def _attr_tokens(name: str) -> list[str]:
    """Return the list of OTEL_RESOURCE_ATTRIBUTES tokens with prefix ``{name}=``."""
    raw = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    return [t for t in raw.split(",") if t.strip().startswith(f"{name}=")]


def test_catalyst_team_id_does_not_change_service_name(monkeypatch) -> None:
    """Regression: an injected CATALYST_TRACING_TEAM_ID must NOT mutate
    service.name. Team / project / etc. grouping flows entirely through
    namespaced halo.* resource attributes (via the generic passthrough)
    so service.name stays a stable top-level identifier across all HALO
    runs."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setenv("CATALYST_TRACING_TEAM_ID", "team-7")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="r")
    assert handle is not None
    assert os.environ.get("CATALYST_SERVICE_NAME") == "halo-engine"
    handle.shutdown()


def test_catalyst_team_id_stamps_resource_attribute(monkeypatch) -> None:
    """CATALYST_TRACING_TEAM_ID lands as a namespaced halo.team.id
    resource attribute via the generic passthrough. Dotted convention
    matches the catalyst-side runtime
    (``halo/src/transport_client/otel_logger.py``) so dashboard
    filters work uniformly across runtime + engine emitters."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setenv("CATALYST_TRACING_TEAM_ID", "team-7")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="r")
    assert handle is not None
    assert _attr_tokens("halo.team.id") == ["halo.team.id=team-7"]
    handle.shutdown()


def test_catalyst_no_team_id_omits_team_attribute(monkeypatch) -> None:
    """Standalone runs (no CATALYST_TRACING_TEAM_ID) don't stamp a
    halo.team.id attribute and service.name stays the constant
    'halo-engine'."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="r")
    assert handle is not None
    assert os.environ.get("CATALYST_SERVICE_NAME") == "halo-engine"
    assert _attr_tokens("halo.team.id") == []
    handle.shutdown()


def test_catalyst_team_id_replaces_prior_attribute(monkeypatch) -> None:
    """Re-running setup with a different team_id must not accumulate
    stale halo.team.id tokens, mirroring the halo.run.id contract."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setenv("CATALYST_TRACING_TEAM_ID", "team-aaa")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    h1 = setup_telemetry(enable=True, run_id="r1")
    assert h1 is not None

    monkeypatch.setenv("CATALYST_TRACING_TEAM_ID", "team-bbb")
    h2 = setup_telemetry(enable=True, run_id="r2")
    assert h2 is not None

    assert _attr_tokens("halo.team.id") == ["halo.team.id=team-bbb"]
    h1.shutdown()
    h2.shutdown()


def test_catalyst_path_stamps_engine_version(monkeypatch) -> None:
    """halo.engine.version is stamped from importlib.metadata so a
    Catalyst dashboard can split spans by HALO release for regressions."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    monkeypatch.setattr("engine.telemetry.setup._halo_engine_version", lambda: "9.9.9-test")
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="r")
    assert handle is not None
    assert _attr_tokens("halo.engine.version") == ["halo.engine.version=9.9.9-test"]
    handle.shutdown()


def test_catalyst_path_defaults_service_version(monkeypatch) -> None:
    """CATALYST_SERVICE_VERSION defaults to the halo-engine package
    version when unset; a user-pinned value wins."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    monkeypatch.setattr("engine.telemetry.setup._halo_engine_version", lambda: "9.9.9-test")
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="r")
    assert handle is not None
    assert os.environ.get("CATALYST_SERVICE_VERSION") == "9.9.9-test"
    handle.shutdown()


def test_catalyst_path_respects_user_service_version(monkeypatch) -> None:
    """A user-pinned CATALYST_SERVICE_VERSION must not be overwritten."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setenv("CATALYST_SERVICE_VERSION", "custom-version")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    monkeypatch.setattr("engine.telemetry.setup._halo_engine_version", lambda: "9.9.9-test")
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="r")
    assert handle is not None
    assert os.environ.get("CATALYST_SERVICE_VERSION") == "custom-version"
    handle.shutdown()


def test_catalyst_team_id_blank_treated_as_unset(monkeypatch) -> None:
    """A whitespace-only CATALYST_TRACING_TEAM_ID must NOT produce an
    empty halo.team.id token — Catalyst is likely to leave the var as
    empty string rather than unsetting it."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setenv("CATALYST_TRACING_TEAM_ID", "   ")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="r")
    assert handle is not None
    assert os.environ.get("CATALYST_SERVICE_NAME") == "halo-engine"
    assert _attr_tokens("halo.team.id") == []
    handle.shutdown()


# ---------------------------------------------------------------------------
# resolve_run_id: caller-injectable run id via CATALYST_TRACING_RUN_ID.
# ---------------------------------------------------------------------------


def test_resolve_run_id_returns_uuid_when_env_unset() -> None:
    """No env override → fresh hex uuid (32 lowercase hex chars)."""
    rid = resolve_run_id()
    assert len(rid) == 32
    assert all(c in "0123456789abcdef" for c in rid)


def test_resolve_run_id_honors_env_override(monkeypatch) -> None:
    """CATALYST_TRACING_RUN_ID set → returned verbatim so Catalyst's
    bookkeeping and HALO's telemetry agree on the run identifier."""
    monkeypatch.setenv("CATALYST_TRACING_RUN_ID", "catalyst-injected-run-42")
    assert resolve_run_id() == "catalyst-injected-run-42"


def test_resolve_run_id_treats_empty_env_as_unset(monkeypatch) -> None:
    """An empty string is not a valid run id; fall back to a uuid so we
    never produce traces with run_id=''."""
    monkeypatch.setenv("CATALYST_TRACING_RUN_ID", "")
    rid = resolve_run_id()
    assert rid != ""
    assert len(rid) == 32


# ---------------------------------------------------------------------------
# Generic CATALYST_TRACING_* → halo.<name> resource attribute passthrough.
# Lets Catalyst inject arbitrary metadata fields without HALO code changes.
# ---------------------------------------------------------------------------


def test_catalyst_passthrough_unknown_env_becomes_halo_attr(monkeypatch) -> None:
    """Any CATALYST_TRACING_<NAME> env (one HALO has never heard of)
    lands as halo.<name> on every span. This is the contract that lets
    Catalyst evolve its injected metadata without HALO releases."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setenv("CATALYST_TRACING_USER_ID", "user-123")
    monkeypatch.setenv("CATALYST_TRACING_DEPLOYMENT_ENV", "staging")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="r")
    assert handle is not None
    assert _attr_tokens("halo.user.id") == ["halo.user.id=user-123"]
    assert _attr_tokens("halo.deployment.env") == ["halo.deployment.env=staging"]
    handle.shutdown()


def test_catalyst_passthrough_translates_underscores_to_dots(monkeypatch) -> None:
    """CATALYST_TRACING_TEAM_ID → halo.team.id (lowercased + ``_`` →
    ``.``). The dotted form matches what the catalyst-side runtime
    already emits for its known fields (``halo.run.id``,
    ``halo.team.id``, ``halo.project.id`` in
    ``halo/src/transport_client/otel_logger.py``) so dashboard filters
    work uniformly across both emitters."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setenv("CATALYST_TRACING_TEAM_ID", "team-7")
    monkeypatch.setenv("CATALYST_TRACING_PROJECT_ID", "proj-9")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="r")
    assert handle is not None
    assert _attr_tokens("halo.team.id") == ["halo.team.id=team-7"]
    assert _attr_tokens("halo.project.id") == ["halo.project.id=proj-9"]
    raw = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    # No snake_case leftovers, no upper-case mirrors.
    assert "halo.team_id=" not in raw
    assert "halo.project_id=" not in raw
    assert "halo.TEAM" not in raw
    handle.shutdown()


def test_catalyst_passthrough_skips_run_id_to_avoid_duplicate(monkeypatch) -> None:
    """halo.run.id has a canonical source (the resolved run_id passed
    into _setup_catalyst); the generic loop must skip
    CATALYST_TRACING_RUN_ID so we don't emit two halo.run.id tokens."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setenv("CATALYST_TRACING_RUN_ID", "catalyst-run-1")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="catalyst-run-1")
    assert handle is not None
    # Exactly one halo.run.id token, with the resolved value.
    assert _attr_tokens("halo.run.id") == ["halo.run.id=catalyst-run-1"]
    handle.shutdown()


def test_catalyst_passthrough_skips_blank_values(monkeypatch) -> None:
    """A whitespace-only value is treated as unset (Catalyst is more
    likely to leave a var blank than to actually unset it). No empty
    halo.<name>= token should appear."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setenv("CATALYST_TRACING_USER_ID", "   ")
    monkeypatch.setenv("CATALYST_TRACING_PROJECT_ID", "")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="r")
    assert handle is not None
    raw = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    assert "halo.user.id" not in raw
    assert "halo.project.id" not in raw
    handle.shutdown()


def test_catalyst_passthrough_dropped_on_repeat_setup(monkeypatch) -> None:
    """A passthrough key that's no longer in the env on a later setup
    must NOT linger in OTEL_RESOURCE_ATTRIBUTES. Generalizes the same
    contract as halo.run.id replacement to dynamic fields."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setenv("CATALYST_TRACING_USER_ID", "user-aaa")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    h1 = setup_telemetry(enable=True, run_id="r1")
    assert h1 is not None
    assert _attr_tokens("halo.user.id") == ["halo.user.id=user-aaa"]

    monkeypatch.delenv("CATALYST_TRACING_USER_ID", raising=False)
    h2 = setup_telemetry(enable=True, run_id="r2")
    assert h2 is not None
    assert _attr_tokens("halo.user.id") == []

    h1.shutdown()
    h2.shutdown()


def test_catalyst_passthrough_preserves_non_halo_resource_attrs(monkeypatch) -> None:
    """Pre-existing non-halo OTEL_RESOURCE_ATTRIBUTES tokens (e.g. a
    user-set deployment.environment from another tool) must survive the
    halo.* cleanup pass."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        "deployment.environment=dev,team.owner=infra",
    )
    monkeypatch.setenv("CATALYST_TRACING_USER_ID", "user-1")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="r")
    assert handle is not None
    raw = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    assert "deployment.environment=dev" in raw
    assert "team.owner=infra" in raw
    assert "halo.user.id=user-1" in raw
    handle.shutdown()


def test_catalyst_passthrough_emits_deterministic_order(monkeypatch) -> None:
    """The generic-passthrough segment of OTEL_RESOURCE_ATTRIBUTES is
    emitted in sorted key order so the env value is stable across runs
    — important for tests and any consumer that string-compares."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setenv("CATALYST_TRACING_ZULU", "z")
    monkeypatch.setenv("CATALYST_TRACING_ALPHA", "a")
    monkeypatch.setenv("CATALYST_TRACING_MIKE", "m")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="r")
    assert handle is not None
    raw = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    # Index of each token, then check sorted by suffix.
    idx_alpha = raw.index("halo.alpha=")
    idx_mike = raw.index("halo.mike=")
    idx_zulu = raw.index("halo.zulu=")
    assert idx_alpha < idx_mike < idx_zulu
    handle.shutdown()


# ---------------------------------------------------------------------------
# Security: validate environment-injected inputs that flow into a file
# path or into OTEL_RESOURCE_ATTRIBUTES.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    [
        "../../../etc/passwd",
        "..\\..\\windows",
        "id/with/slashes",
        "id with spaces",
        "id;with;semicolons",
        "id\nwith\nnewlines",
        "a" * 200,  # exceeds 128-char cap
    ],
)
def test_resolve_run_id_rejects_unsafe_values_and_falls_back_to_uuid(
    monkeypatch, caplog, bad_value
) -> None:
    """Unsafe CATALYST_TRACING_RUN_ID values (path traversal, control
    chars, oversized) must fall back to a fresh uuid so the value
    can't escape its intended use as a filename / attribute fragment.
    A WARNING is logged so the rejection isn't silent."""
    monkeypatch.setenv("CATALYST_TRACING_RUN_ID", bad_value)
    with caplog.at_level(logging.WARNING, logger="engine.telemetry.setup"):
        rid = resolve_run_id()
    assert rid != bad_value
    assert len(rid) == 32
    assert all(c in "0123456789abcdef" for c in rid)
    assert "CATALYST_TRACING_RUN_ID rejected" in caplog.text


@pytest.mark.parametrize(
    "good_value",
    [
        "abc123",
        "uuid-like-1234-5678",
        "with.dots",
        "with_underscores",
        "Mixed-Case_42.0",
        "a" * 128,  # exactly at the cap is allowed
    ],
)
def test_resolve_run_id_accepts_safe_values(monkeypatch, good_value) -> None:
    """Values within the safe charset (alphanumerics, ``-_.``) and length
    cap pass through untouched. This is the contract for any caller
    (Catalyst, CI, manual smoke tests) injecting a run id."""
    monkeypatch.setenv("CATALYST_TRACING_RUN_ID", good_value)
    assert resolve_run_id() == good_value


def test_resolve_run_id_strips_whitespace_before_validating(monkeypatch) -> None:
    """Surrounding whitespace is treated as a leading/trailing accident
    rather than as part of the id; the trimmed value is what's
    validated. Matches how blank values are handled."""
    monkeypatch.setenv("CATALYST_TRACING_RUN_ID", "  run-42  ")
    assert resolve_run_id() == "run-42"


def test_catalyst_passthrough_value_with_comma_does_not_inject_attributes(
    monkeypatch,
) -> None:
    """A passthrough value containing ``,`` or ``=`` must NOT inject a
    sibling attribute into OTEL_RESOURCE_ATTRIBUTES. The value is
    percent-encoded on emit; the OTel resource detector decodes it
    losslessly so the attribute value carries the original literal."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setenv("CATALYST_TRACING_USER_ID", "legit,injected.key=evil")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="r")
    assert handle is not None
    raw = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    # The injection attempt must not surface as a sibling attribute key.
    assert "injected.key=" not in raw
    # The original literal value must be recoverable from the encoded form.
    from urllib.parse import unquote

    user_tokens = [t for t in raw.split(",") if t.startswith("halo.user.id=")]
    assert len(user_tokens) == 1
    decoded = unquote(user_tokens[0].removeprefix("halo.user.id="))
    assert decoded == "legit,injected.key=evil"
    handle.shutdown()


def test_catalyst_passthrough_value_with_equals_only_does_not_inject(
    monkeypatch,
) -> None:
    """``=`` alone (no preceding ``,``) must also be encoded — without
    that, the OTel parser would still see only one token but treat
    the second ``=`` as part of an oversized value, which is the
    contract for OTel but worth pinning."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setenv("CATALYST_TRACING_USER_ID", "k=v")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="r")
    assert handle is not None
    raw = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    # The encoded form has %3D in place of the second '='.
    assert "halo.user.id=k%3Dv" in raw
    handle.shutdown()


def test_safe_run_id_value_is_not_over_encoded_in_resource_attrs(
    monkeypatch,
) -> None:
    """Regression: a normal safe run id (alphanumeric + ``-``) must
    appear verbatim in halo.run.id — encoding kicks in only for
    reserved characters. Catches accidentally-aggressive encoding
    that would force operators to decode every value."""
    monkeypatch.setenv("CATALYST_OTLP_TOKEN", "test-token")
    monkeypatch.setenv("CATALYST_TRACING_RUN_ID", "run-abc-123")
    monkeypatch.setattr("engine.telemetry.setup.set_trace_processors", lambda procs: None)
    _install_stub_catalyst(monkeypatch)

    handle = setup_telemetry(enable=True, run_id="run-abc-123")
    assert handle is not None
    assert _attr_tokens("halo.run.id") == ["halo.run.id=run-abc-123"]
    handle.shutdown()
