"""Tests for the per-call response budget on ``view_trace`` and ``view_spans``.

Constructs a synthetic trace whose spans, after the surgical-cap truncation, still
exceed the 150_000-byte response budget so we can exercise the oversized-summary
fallback path without needing an enormous fixture file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore


def _big_span(*, trace_id: str, span_id: str, payload_chars: int) -> str:
    """One JSONL line with a large ``input.value`` so post-truncation spans stay big."""
    return (
        json.dumps(
            {
                "trace_id": trace_id,
                "span_id": span_id,
                "parent_span_id": "",
                "trace_state": "",
                "name": "big",
                "kind": "SPAN_KIND_INTERNAL",
                "start_time": "2026-04-23T05:32:00.000000000Z",
                "end_time": "2026-04-23T05:32:01.000000000Z",
                "status": {"code": "STATUS_CODE_OK", "message": ""},
                "resource": {"attributes": {"service.name": "agent-big"}},
                "scope": {"name": "@test/scope", "version": "0.0.1"},
                "attributes": {
                    "openinference.span.kind": "AGENT",
                    "inference.export.schema_version": 1,
                    "inference.project_id": "prj_test",
                    "inference.observation_kind": "AGENT",
                    "inference.agent_name": "agent-big",
                    # Large head-of-payload survives the per-attribute truncation
                    # (4 KB on view_trace/search_trace, 16 KB on view_spans).
                    "input.value": "X" * payload_chars,
                },
            }
        )
        + "\n"
    )


@pytest.mark.asyncio
async def test_view_trace_returns_oversized_summary_over_byte_budget(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "big.jsonl"
    # 50 spans × 4 KB attribute value > 150_000-byte view_trace budget after the
    # 4 KB per-attribute discovery cap.
    with trace_path.open("w") as fh:
        for i in range(50):
            fh.write(_big_span(trace_id="t-big", span_id=f"s-big-{i}", payload_chars=4096))

    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    store = TraceStore.load(trace_path=trace_path, index_path=index_path)

    view = store.view_trace("t-big")
    assert view.spans == []
    summary = view.oversized
    assert summary is not None
    assert summary.span_count == 50
    assert summary.truncated_response_bytes > summary.response_bytes_budget
    assert summary.response_bytes_budget == 150_000
    assert summary.span_response_bytes_max > 0
    assert summary.top_span_names[0][0] == "big"
    assert "search_trace" in summary.recommendation


@pytest.mark.asyncio
async def test_view_spans_returns_oversized_summary_over_byte_budget(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "big.jsonl"
    # Each span keeps a 16 KB ``input.value`` after the surgical cap, so 12 spans
    # (~192 KB) cleanly exceed the 150_000-byte view_spans budget.
    with trace_path.open("w") as fh:
        for i in range(12):
            fh.write(_big_span(trace_id="t-big", span_id=f"s-big-{i}", payload_chars=16384))

    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    store = TraceStore.load(trace_path=trace_path, index_path=index_path)

    view = store.view_spans("t-big", [f"s-big-{i}" for i in range(12)])
    assert view.spans == []
    summary = view.oversized
    assert summary is not None
    assert summary.span_count == 12
    assert summary.response_bytes_budget == 150_000
    assert summary.truncated_response_bytes > summary.response_bytes_budget
    assert "search_span" in summary.recommendation


@pytest.mark.asyncio
async def test_view_spans_under_budget_returns_spans(tmp_path: Path) -> None:
    trace_path = tmp_path / "big.jsonl"
    # Two small-but-nontrivial spans well under budget.
    with trace_path.open("w") as fh:
        for i in range(2):
            fh.write(_big_span(trace_id="t-big", span_id=f"s-big-{i}", payload_chars=1024))

    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    store = TraceStore.load(trace_path=trace_path, index_path=index_path)

    view = store.view_spans("t-big", ["s-big-0"])
    assert [s.span_id for s in view.spans] == ["s-big-0"]
    assert view.oversized is None
