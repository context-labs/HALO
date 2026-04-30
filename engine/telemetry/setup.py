"""HALO engine telemetry: opt-in OTLP-or-local tracing for HALO itself.

Init lifecycle is owned by ``stream_engine_async`` in ``engine/main.py``.
Callers pass ``telemetry=True`` to opt in. Routing is decided here, by
presence of ``CATALYST_OTLP_TOKEN`` in the environment.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from agents import set_trace_processors

from engine.telemetry.local_processor import attach_local_processor


class TelemetryHandle:
    """Owns shutdown for whichever backend was selected. Idempotent.
    ``shutdown()`` swallows backend errors so it cannot mask an engine
    exception in an outer ``finally``.
    """

    def __init__(self, *, backend: Any) -> None:
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


def setup_telemetry(*, enable: bool, run_id: str | None = None) -> TelemetryHandle | None:
    """Initialize tracing. Returns None when ``enable`` is False (default).

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
        return _setup_catalyst()
    return _setup_local(run_id=run_id)


def _setup_catalyst() -> TelemetryHandle:
    try:
        from inference_catalyst_tracing import setup
    except ImportError as exc:
        raise RuntimeError(
            "Telemetry is enabled but the optional 'telemetry' extra is "
            "not installed. Install with: pip install 'halo-engine[telemetry]' "
            "(requires Python >=3.11)."
        ) from exc

    backend = setup()
    return TelemetryHandle(backend=backend)


def _setup_local(*, run_id: str | None) -> TelemetryHandle:
    rid = run_id or uuid.uuid4().hex
    path = os.environ.get("HALO_TELEMETRY_PATH") or f"halo-telemetry-{rid}.jsonl"
    service_name = os.environ.get("CATALYST_SERVICE_NAME", "halo-engine")
    processor = attach_local_processor(
        path=path,
        service_name=service_name,
        project_id="halo-engine",
    )
    return TelemetryHandle(backend=processor)
