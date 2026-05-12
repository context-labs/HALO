"""HALO engine telemetry: opt-in tracing for HALO itself.

Init lifecycle is owned by ``stream_engine_async`` in ``engine/main.py``.
Callers pass ``telemetry=True`` to opt in.

Routing rule (decided here, not by callers):

* ``CATALYST_OTLP_TOKEN`` is set → spans go to inference.net Catalyst
  over OTLP via ``inference_catalyst_tracing``.
* ``CATALYST_OTLP_TOKEN`` is unset → spans go to a local JSONL file at
  ``$HALO_TELEMETRY_PATH`` (default ``./halo-telemetry-{run_id}.jsonl``)
  via ``InferenceOtlpFileProcessor``.

Both backends carry a ``halo.run.id`` resource attribute so traces can
be filtered to a single HALO run server-side.

Catalyst-deployed runs (HALO running inside a Catalyst-launched Modal
sandbox) get extra identity stamped from the env vars Catalyst injects:

* ``CATALYST_TRACING_RUN_ID`` → consumed by ``resolve_run_id`` so a
  caller-injected run id flows through both backends. Falls back to a
  fresh ``uuid4().hex`` when unset.
* Generic passthrough: **any** ``CATALYST_TRACING_<NAME>=<value>`` env
  var becomes a ``halo.<name>=<value>`` resource attribute on every
  span, with ``<name>`` lowercased and ``_`` translated to ``.`` to
  match the dotted convention the catalyst-side runtime uses
  (``halo.run.id``, ``halo.team.id``, ``halo.project.id``, …; see
  ``halo/src/transport_client/otel_logger.py`` in the inference
  monorepo). Lets Catalyst inject new metadata fields without HALO
  releases.

``service.name`` is intentionally **constant** (`halo-engine`) — team /
project / user grouping flows through the namespaced ``halo.*`` resource
attributes above so a Catalyst dashboard filter is unambiguous and
service.name stays a stable top-level identifier across all HALO runs.

User-set ``CATALYST_SERVICE_NAME`` / ``CATALYST_SERVICE_VERSION`` always
win over the defaults below.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Protocol
from urllib.parse import quote

from agents import set_trace_processors
from inference_catalyst_tracing import setup as catalyst_setup

from engine.telemetry.local_processor import attach_local_processor

_CATALYST_TRACING_PREFIX = "CATALYST_TRACING_"

# A run id is interpolated into a local file path
# (``halo-telemetry-{run_id}.jsonl``) and into otel resource attributes,
# so the charset has to be safe for both. Allow alphanumerics + the
# punctuation that Catalyst is likely to use (uuids with hyphens, dotted
# segments, underscores). Anything else — including path separators —
# is rejected and ``resolve_run_id`` falls back to a fresh uuid.
_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_MAX_RUN_ID_LEN = 128

_logger = logging.getLogger(__name__)


class _Shutdownable(Protocol):
    """Backend protocol — both Catalyst (``CatalystTracing``) and local
    (``InferenceOtlpFileProcessor``) implement ``shutdown()``."""

    def shutdown(self) -> None: ...


class TelemetryHandle:
    """Owns shutdown for the telemetry backend. Idempotent.

    ``shutdown()`` swallows backend errors so it cannot mask an engine
    exception in an outer ``finally``.
    """

    def __init__(self, *, backend: _Shutdownable) -> None:
        self._backend = backend
        self._closed = False

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._backend.shutdown()
        except Exception:
            pass


def resolve_run_id() -> str:
    """Return the HALO run id, honoring a Catalyst-injected override.

    When HALO runs inside a Catalyst-launched sandbox, Catalyst sets
    ``CATALYST_TRACING_RUN_ID`` so its own bookkeeping and HALO's
    telemetry agree on the run identifier. Standalone runs get a fresh
    ``uuid4().hex``.

    Validates the env value against ``_SAFE_RUN_ID_RE`` because the
    run id is interpolated into the local telemetry file path
    (``halo-telemetry-{run_id}.jsonl``); without validation, a value
    like ``../../../etc/passwd`` would write outside the working
    directory. Rejected values fall back to a fresh uuid and the
    rejection is logged at WARNING.
    """
    raw = os.environ.get("CATALYST_TRACING_RUN_ID", "").strip()
    if not raw:
        return uuid.uuid4().hex
    if len(raw) <= _MAX_RUN_ID_LEN and _SAFE_RUN_ID_RE.match(raw):
        return raw
    _logger.warning(
        "CATALYST_TRACING_RUN_ID rejected (length<=%d and charset %s required); "
        "falling back to a generated uuid.",
        _MAX_RUN_ID_LEN,
        _SAFE_RUN_ID_RE.pattern,
    )
    return uuid.uuid4().hex


def setup_telemetry(*, enable: bool, run_id: str) -> TelemetryHandle | None:
    """Initialize tracing. Returns None when ``enable`` is False.

    Routing rule:
      - ``CATALYST_OTLP_TOKEN`` set → OTLP via ``inference_catalyst_tracing``
      - otherwise                    → local JSONL via ``InferenceOtlpFileProcessor``

    Always clears the openai-agents SDK's default tracing processor list
    so HALO's own LLM activity does not leak to the OpenAI dashboard.
    """
    set_trace_processors([])
    if not enable:
        return None

    if os.environ.get("CATALYST_OTLP_TOKEN"):
        return _setup_catalyst(run_id=run_id)
    return _setup_local(run_id=run_id)


def _halo_engine_version() -> str:
    """Return the installed halo-engine package version, or ``"unknown"``.

    Packaging metadata is the source of truth (``pyproject.toml``).
    ``PackageNotFoundError`` covers running from a source checkout
    that hasn't been ``pip install``-ed — telemetry init shouldn't
    fail in that case, just degrade to a sentinel version string.
    """
    try:
        return version("halo-engine")
    except PackageNotFoundError:
        return "unknown"


def _format_attr_token(key: str, value: str) -> str:
    """Render a single ``OTEL_RESOURCE_ATTRIBUTES`` token, percent-encoding
    the value so embedded ``,`` / ``=`` can't inject sibling attributes.

    ``OTEL_RESOURCE_ATTRIBUTES`` is comma-delimited ``key=value`` pairs
    with values percent-decoded by the OTel resource detector (per the
    spec, values are W3C Baggage-encoded). Without encoding, a value
    like ``team-7,injected.key=evil`` would parse as TWO attributes
    (``halo.team.id=team-7`` and ``injected.key=evil``). Using
    ``quote(value, safe='')`` encodes everything outside the unreserved
    URL set, which the OTel detector decodes back losslessly.
    """
    return f"{key}={quote(value, safe='')}"


def _env_suffix_to_attr_name(suffix: str) -> str:
    """``TEAM_ID`` → ``team.id``. Lowercase and translate ``_`` → ``.``.

    The dotted convention matches what the catalyst-side HALO runtime
    already emits for its known fields (``halo.run.id``, ``halo.team.id``,
    ``halo.project.id``, etc.) so dashboard filters and superuser queries
    can use a single key shape across runtime-emitted spans / logs and
    engine-emitted spans.
    """
    return suffix.lower().replace("_", ".")


def _collect_dynamic_halo_attrs(env: Mapping[str, str]) -> list[str]:
    """Translate every ``CATALYST_TRACING_<NAME>=<value>`` env var into a
    ``halo.<name>=<value>`` token suitable for ``OTEL_RESOURCE_ATTRIBUTES``.

    Convention: ``CATALYST_TRACING_TEAM_ID=team-7`` →
    ``halo.team.id=team-7`` (lowercased, ``_`` → ``.``). Empty /
    whitespace-only values are skipped (Catalyst is more likely to
    leave a var as ``""`` than to actually unset it). Sorted iteration
    keeps ``OTEL_RESOURCE_ATTRIBUTES`` deterministic across runs.
    """
    out: list[str] = []
    for key in sorted(env):
        if not key.startswith(_CATALYST_TRACING_PREFIX):
            continue
        # halo.run.id is added explicitly from the resolved run_id (which
        # works even on standalone runs where this env is unset), so
        # skip the env var here to avoid emitting a duplicate token.
        if key == "CATALYST_TRACING_RUN_ID":
            continue
        suffix = key.removeprefix(_CATALYST_TRACING_PREFIX)
        if not suffix:
            continue
        value = env[key].strip()
        if not value:
            continue
        out.append(_format_attr_token(f"halo.{_env_suffix_to_attr_name(suffix)}", value))
    return out


def _setup_catalyst(*, run_id: str) -> TelemetryHandle:
    # service.name is intentionally constant. Team / project / user
    # grouping flows through the namespaced halo.* resource attributes
    # produced by the generic CATALYST_TRACING_* passthrough below, so
    # service.name stays a stable top-level identifier across all HALO
    # runs and dashboards filter by halo.team.id / halo.project.id /
    # etc. for the secondary axes.
    os.environ.setdefault("CATALYST_SERVICE_NAME", "halo-engine")
    os.environ.setdefault("CATALYST_SERVICE_VERSION", _halo_engine_version())

    existing = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "").strip()
    # Drop any halo.* tokens we appended on a prior call so repeated
    # invocations in the same process (library usage / repeated setup
    # in tests) don't accumulate stale entries — including dynamic
    # fields from CATALYST_TRACING_* env vars that may have been
    # unset between calls.
    kept = [t for t in existing.split(",") if t and not t.strip().startswith("halo.")]
    halo_attrs = [
        _format_attr_token("halo.run.id", run_id),
        _format_attr_token("halo.engine.version", _halo_engine_version()),
        *_collect_dynamic_halo_attrs(os.environ),
    ]
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = ",".join([*kept, *halo_attrs])

    backend = catalyst_setup()
    return TelemetryHandle(backend=backend)


def _setup_local(*, run_id: str) -> TelemetryHandle:
    path = os.environ.get("HALO_TELEMETRY_PATH") or f"halo-telemetry-{run_id}.jsonl"
    processor = attach_local_processor(
        path=path,
        service_name="halo-engine",
        project_id="halo-engine",
        extra_resource_attributes={"halo.run.id": run_id},
    )
    return TelemetryHandle(backend=processor)
