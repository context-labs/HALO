"""HALO-side OpenInference AGENT-span helper.

Wrapper around ``inference_catalyst_tracing.agent_span`` that:

* Falls back to a no-op when the optional ``telemetry`` extra is not
  installed (default ``uv sync`` path) — no import error, no overhead.
* Falls back to a no-op when no global ``TracerProvider`` is registered
  (i.e. ``setup_telemetry`` chose the local backend or telemetry is off)
  — the OTel proxy tracer's spans go nowhere.

Used by the engine to open AGENT-kind spans around the root run and
each subagent invocation. catalyst-tracing's openai instrumentation
attaches its per-call ``OpenAI Responses`` spans to whatever span is
active in OTel context, so wrapping these scopes is what gives Catalyst
dashboards a single trace per HALO run with a hierarchical agent tree.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:
    from inference_catalyst_tracing import agent_span as _agent_span
    from opentelemetry import trace as _trace

    _HAS_TELEMETRY = True
except ImportError:
    _HAS_TELEMETRY = False


_TRACER_NAME = "halo-engine"


@contextmanager
def halo_agent_span(*, name: str, system: str = "openai") -> Iterator[Any]:
    """Open an OpenInference AGENT span around a chunk of HALO agent work.

    ``name`` becomes the span's ``agent.name`` and the prefix of its
    span name (``f"{name}.run"``). ``system`` becomes ``gen_ai.system``.

    Yields the ``AgentSpanHandle`` from catalyst-tracing when telemetry
    is wired, or ``None`` when not — callers must tolerate ``None``.
    """
    if not _HAS_TELEMETRY:
        yield None
        return
    tracer = _trace.get_tracer(_TRACER_NAME)
    with _agent_span(tracer, name=name, system=system) as span:
        yield span
