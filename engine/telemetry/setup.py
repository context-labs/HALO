"""HALO engine telemetry: opt-in local JSONL tracing for HALO itself.

Init lifecycle is owned by ``stream_engine_async`` in ``engine/main.py``.
Callers pass ``telemetry=True`` to opt in; spans are written to a local
JSONL file at ``$HALO_TELEMETRY_PATH`` (default:
``./halo-telemetry-{run_id}.jsonl``).

A direct upload path to inference.net Catalyst is planned but currently
blocked on an upstream incompatibility between ``catalyst-tracing``'s
OpenAI instrumentation and the ``with_streaming_response.create`` path
that ``openai-agents`` >= 0.11.0 uses. See
``docs/superpowers/plans/2026-05-01-restore-catalyst-telemetry.md`` for
the restoration plan.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from agents import set_trace_processors

from engine.telemetry.local_processor import attach_local_processor


class TelemetryHandle:
    """Owns shutdown for the telemetry backend. Idempotent.

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

    Attaches the local JSONL processor to the openai-agents SDK. Path
    defaults to ``./halo-telemetry-{run_id}.jsonl``; override with
    ``$HALO_TELEMETRY_PATH``.

    Always clears the openai-agents SDK's default tracing processor list
    so HALO's own LLM activity does not leak to the OpenAI dashboard.
    """
    if not enable:
        return None

    set_trace_processors([])

    rid = run_id or uuid.uuid4().hex
    path = os.environ.get("HALO_TELEMETRY_PATH") or f"halo-telemetry-{rid}.jsonl"
    processor = attach_local_processor(
        path=path,
        service_name="halo-engine",
        project_id="halo-engine",
        extra_resource_attributes={"halo.run_id": rid},
    )
    return TelemetryHandle(backend=processor)
