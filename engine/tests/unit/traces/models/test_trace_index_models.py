from __future__ import annotations

from engine.traces.models.trace_index_models import TraceIndexMeta, TraceIndexRow


def test_row_roundtrip() -> None:
    row = TraceIndexRow(
        trace_id="t1",
        byte_offsets=[0, 512],
        byte_lengths=[512, 256],
        span_count=2,
        start_time="2026-04-23T00:00:00Z",
        end_time="2026-04-23T00:00:01Z",
        has_errors=False,
        service_names=["svc"],
        model_names=["claude-sonnet-4-5"],
        total_input_tokens=100,
        total_output_tokens=50,
        project_id="prj_1",
        agent_names=["agent-a"],
    )
    blob = row.model_dump_json()
    restored = TraceIndexRow.model_validate_json(blob)
    assert restored == row


def test_meta_defaults() -> None:
    meta = TraceIndexMeta(schema_version=1, trace_count=3)
    assert meta.schema_version == 1
    assert meta.trace_count == 3
