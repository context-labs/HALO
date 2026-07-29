"""``engine_span`` wraps engine work without altering its behaviour.

This sits on every tool call, so the invariants that matter are the boring
ones: it must not swallow exceptions, must not turn an unset attribute into
the string ``"None"``, and must cost nothing when telemetry is off.
"""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from engine.telemetry.spans import engine_span


@pytest.fixture
def exporter(monkeypatch) -> InMemorySpanExporter:
    """Install a real recording provider for the duration of one test.

    ``trace.set_tracer_provider`` refuses to overwrite an existing global, so
    the private attribute is patched directly — the supported alternative is
    a fresh interpreter per test.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider, raising=False)
    return exporter


def test_records_the_span_with_its_attributes(exporter: InMemorySpanExporter) -> None:
    with engine_span("halo.tool", **{"tool.name": "view_trace"}) as span:
        span.set_attribute("tool.ok", True)

    finished = exporter.get_finished_spans()
    assert [s.name for s in finished] == ["halo.tool"]
    assert finished[0].attributes is not None
    assert finished[0].attributes["tool.name"] == "view_trace"
    assert finished[0].attributes["tool.ok"] is True


def test_drops_none_attributes_instead_of_stringifying(
    exporter: InMemorySpanExporter,
) -> None:
    """An unset attribute should be absent, not the literal ``"None"``."""
    with engine_span("halo.tool", **{"tool.name": "x", "agent.id": None}):
        pass

    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    assert "agent.id" not in attributes


def test_records_the_exception_exactly_once_and_reraises(
    exporter: InMemorySpanExporter,
) -> None:
    """An observation wrapper must never change control flow — or double-log.

    ``start_as_current_span`` already records the exception and sets ERROR
    status. Catching it here to do the same would attach a second ``exception``
    event to every failing tool call, so the count is asserted, not just the
    presence.
    """
    boom = RuntimeError("tool blew up")

    with pytest.raises(RuntimeError) as caught:
        with engine_span("halo.tool", **{"tool.name": "x"}):
            raise boom

    assert caught.value is boom
    finished = exporter.get_finished_spans()[0]
    assert finished.status.status_code is StatusCode.ERROR
    assert [e.name for e in finished.events] == ["exception"]


def test_is_a_noop_without_a_provider() -> None:
    """With telemetry off the OTel API hands back non-recording spans.

    Callers must not have to branch on whether tracing is enabled, so setting
    attributes on the yielded span has to stay safe.
    """
    with engine_span("halo.tool", **{"tool.name": "x"}) as span:
        span.set_attribute("tool.ok", True)
        assert span.is_recording() is False


def test_a_failed_tool_span_carries_tool_ok_false(
    exporter: InMemorySpanExporter,
) -> None:
    """A failing tool must leave ``tool.ok=False``, not an absent key.

    The dispatcher converts tool exceptions into a string returned to the
    model, so the span's ERROR status is the only other failure signal. If
    ``tool.ok`` were set only on the success path it would be missing rather
    than false on failures, and a ``tool.ok = false`` query would silently
    match nothing — the exact opposite of what the attribute is for.
    """
    with pytest.raises(RuntimeError):
        with engine_span("halo.tool", **{"tool.name": "view_trace", "tool.ok": False}):
            raise RuntimeError("tool blew up")

    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    assert attributes["tool.ok"] is False


def test_a_successful_tool_span_flips_tool_ok_true(
    exporter: InMemorySpanExporter,
) -> None:
    with engine_span("halo.tool", **{"tool.name": "view_trace", "tool.ok": False}) as span:
        span.set_attribute("tool.ok", True)

    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    assert attributes["tool.ok"] is True
