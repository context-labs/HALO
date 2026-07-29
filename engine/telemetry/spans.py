"""Backend-agnostic timing spans for engine internals.

Sibling of :mod:`engine.telemetry.tracing`, which exists only to give
Catalyst an AGENT-span hierarchy. This module answers a different question:
**where does a HALO run's wall-clock actually go?**

Before this, a run produced spans for the root agent, each subagent, and
whatever the openai-agents SDK emitted for LLM calls. Everything else — tool
execution, context compaction, trace-index building — was invisible, so a
30-minute run could only be attributed as "LLM time versus an unexplained
remainder". These spans close that gap.

Uses ``opentelemetry.trace.get_tracer`` directly rather than the Catalyst
wrapper, so the spans appear on whatever backend is configured (Catalyst, the
local JSONL processor, or the runtime wrapper's provider) and cost nothing
when telemetry is off: with no provider registered the OTel API returns a
proxy tracer whose spans are non-recording and whose attribute setters are
no-ops. Callers never need to branch on whether tracing is enabled.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

_TRACER_NAME = "halo-engine"


@contextmanager
def engine_span(name: str, **attributes: Any) -> Iterator[trace.Span]:
    """Time a unit of engine work, recording exceptions before re-raising.

    ``attributes`` are set on the span up front so a span that dies to an
    exception still carries the context needed to interpret it — which tool,
    how many items, which file. ``None`` values are dropped rather than
    stringified, so an unset attribute is absent instead of the literal
    ``"None"``.

    Exceptions are recorded and the span status set to ERROR, then re-raised
    unchanged: this is an observation wrapper and must never alter control
    flow.
    """
    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as error:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            raise
