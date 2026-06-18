from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.models.trace_index_models import TraceIndexRow
from engine.traces.trace_index_builder import TraceIndexBuilder


def _span(
    *,
    span_id: str,
    parent_span_id: str,
    attributes: dict,
    resource_project_id: str = "prj_a",
    status_code: str = "STATUS_CODE_OK",
) -> dict:
    return {
        "trace_id": "t-validation",
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "trace_state": "",
        "name": span_id,
        "kind": "SPAN_KIND_INTERNAL",
        "start_time": "2026-04-23T00:00:00.000000000Z",
        "end_time": "2026-04-23T00:00:01.000000000Z",
        "status": {"code": status_code, "message": ""},
        "resource": {"attributes": {"service.name": "svc", "halo.project.id": resource_project_id}},
        "scope": {"name": "test", "version": "0.0.1"},
        "attributes": attributes,
    }


@pytest.mark.asyncio
async def test_index_records_trace_health_counters(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces.jsonl"
    spans = [
        _span(
            span_id="root",
            parent_span_id="",
            attributes={
                "openinference.span.kind": "AGENT",
                "agent.id": "agent-1",
                "inference.project_id": "prj_a",
            },
        ),
        _span(
            span_id="tool",
            parent_span_id="missing-parent",
            status_code="STATUS_CODE_ERROR",
            attributes={
                "openinference.span.kind": "TOOL",
                "tool.name": "run_code",
                "inference.project_id": "prj_b",
            },
        ),
    ]
    trace_path.write_text("\n".join(json.dumps(span) for span in spans) + "\n")

    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=TraceIndexConfig(),
    )
    rows = [TraceIndexRow.model_validate_json(line) for line in index_path.read_text().splitlines()]

    assert len(rows) == 1
    row = rows[0]
    assert row.agent_ids == ["agent-1"]
    assert row.missing_parent_count == 1
    assert row.missing_agent_identity_count == 1
    assert row.project_id_mismatch_count == 1
    assert row.otel_error_span_count == 1
    assert row.tool_error_span_count == 1
