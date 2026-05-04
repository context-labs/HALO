"""HALO engine telemetry: opt-in tracing for HALO itself.

Init lifecycle is owned by ``stream_engine_async`` in ``engine/main.py``.
Callers pass ``telemetry=True`` to opt in.

Routing rule (decided here, not by callers):

* ``CATALYST_OTLP_TOKEN`` is set → spans go to inference.net Catalyst
  over OTLP via ``inference_catalyst_tracing``. Requires the optional
  ``telemetry`` extra (Python >= 3.11).
* ``CATALYST_OTLP_TOKEN`` is unset → spans go to a local JSONL file at
  ``$HALO_TELEMETRY_PATH`` (default ``./halo-telemetry-{run_id}.jsonl``)
  via ``InferenceOtlpFileProcessor``.

Both backends carry a ``halo.run_id`` resource attribute so traces can
be filtered to a single HALO run server-side.
"""

from __future__ import annotations

import os
from typing import Protocol

from agents import set_trace_processors

from engine.telemetry.local_processor import attach_local_processor


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


def setup_telemetry(*, enable: bool, run_id: str) -> TelemetryHandle | None:
    """Initialize tracing. Returns None when ``enable`` is False.

    Routing rule:
      - ``CATALYST_OTLP_TOKEN`` set → OTLP via ``inference_catalyst_tracing``
      - otherwise                    → local JSONL via ``InferenceOtlpFileProcessor``

    Always clears the openai-agents SDK's default tracing processor list
    so HALO's own LLM activity does not leak to the OpenAI dashboard.
    """
    if not enable:
        return None

    set_trace_processors([])

    if os.environ.get("CATALYST_OTLP_TOKEN"):
        return _setup_catalyst(run_id=run_id)
    return _setup_local(run_id=run_id)


def _setup_catalyst(*, run_id: str) -> TelemetryHandle:
    try:
        from inference_catalyst_tracing import setup
    except ImportError as exc:
        raise RuntimeError(
            "Telemetry is enabled and CATALYST_OTLP_TOKEN is set, but the "
            "optional 'telemetry' extra is not installed. Install with: "
            "pip install 'halo-engine[telemetry]' (requires Python >=3.11)."
        ) from exc

    os.environ.setdefault("CATALYST_SERVICE_NAME", "halo-engine")
    existing = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "").strip()
    halo_attr = f"halo.run_id={run_id}"
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = f"{existing},{halo_attr}" if existing else halo_attr

    backend = setup()
    return TelemetryHandle(backend=backend)


def _setup_local(*, run_id: str) -> TelemetryHandle:
    path = os.environ.get("HALO_TELEMETRY_PATH") or f"halo-telemetry-{run_id}.jsonl"
    processor = attach_local_processor(
        path=path,
        service_name="halo-engine",
        project_id="halo-engine",
        extra_resource_attributes={"halo.run_id": run_id},
    )
    return TelemetryHandle(backend=processor)
