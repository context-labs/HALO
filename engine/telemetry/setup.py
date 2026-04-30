"""HALO engine telemetry: opt-in OTLP-or-local tracing for HALO itself.

Init lifecycle is owned by ``stream_engine_async`` in ``engine/main.py``.
Callers pass ``telemetry=True`` to opt in. Routing is decided here, by
presence of ``CATALYST_OTLP_TOKEN`` in the environment.
"""

from __future__ import annotations


class TelemetryHandle:
    """Owns shutdown for whichever backend was selected. Idempotent."""

    def __init__(self) -> None:
        self._closed = False

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Backends register their flush in subclasses (Tasks 4-5).


def setup_telemetry(*, enable: bool, run_id: str | None = None) -> TelemetryHandle | None:
    """Initialize tracing. Returns None when ``enable`` is False (default)."""
    if not enable:
        return None
    raise NotImplementedError("Backends added in Tasks 4-5")
