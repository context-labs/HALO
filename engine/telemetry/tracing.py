"""Catalyst-only AGENT-span wrapper.

Only meaningful for the Catalyst telemetry backend (``_setup_catalyst``
in ``engine.telemetry.setup``). The local JSONL backend doesn't need
these wrappers — it captures whatever spans the openai-agents SDK and
OpenAI instrumentation produce, with no extra grouping.

Why Catalyst needs it: catalyst-tracing's OpenAI instrumentation attaches
each ``OpenAI Responses`` span to whatever OTel span is active in
context. Without an outer AGENT span, every per-call span ends up at
the root of the trace and Catalyst dashboards show a flat list instead
of an agent tree. Wrapping the root run and each subagent invocation
in ``halo_agent_span`` is what gives Catalyst one trace per HALO run
with a proper hierarchy.

When telemetry is off or the local backend is in use, no global
``TracerProvider`` is registered and ``opentelemetry.trace.get_tracer``
returns the no-op proxy tracer. ``agent_span`` then yields a handle
backed by a ``NonRecordingSpan`` whose attribute setters are no-ops,
so callers don't need to special-case the off path.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from inference_catalyst_tracing import AgentSpanHandle, agent_span
from opentelemetry import trace

_TRACER_NAME = "halo-engine"


@contextmanager
def halo_agent_span(*, name: str, system: str = "openai") -> Iterator[AgentSpanHandle]:
    """Open an OpenInference AGENT span around a chunk of HALO agent work.

    ``name`` becomes the span's ``agent.name`` and the prefix of its
    span name (``f"{name}.run"``). ``system`` becomes ``gen_ai.system``.

    Yields the ``AgentSpanHandle`` from catalyst-tracing. When telemetry
    is off or the local backend is active, the handle wraps a
    no-op ``NonRecordingSpan`` and all attribute setters silently no-op.
    """
    tracer = trace.get_tracer(_TRACER_NAME)
    with agent_span(tracer, name=name, system=system) as span:
        yield span
