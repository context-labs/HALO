from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.models.trace_index_models import TraceIndexMeta, TraceIndexRow
from engine.traces.trace_index_builder import TraceIndexBuilder


@pytest.mark.asyncio
async def test_ensure_index_exists_default_path_returned(tmp_path: Path) -> None:
    trace_path = tmp_path / "t.jsonl"
    trace_path.write_text("")
    # Pre-create the index so builder returns without rebuilding
    default_index = Path(str(trace_path) + ".engine-index.jsonl")
    default_meta = Path(str(trace_path) + ".engine-index.meta.json")
    default_index.write_text("")
    default_meta.write_text('{"schema_version":1,"trace_count":0}')

    result_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=TraceIndexConfig(),
    )
    assert result_path == default_index


@pytest.mark.asyncio
async def test_ensure_index_exists_explicit_override(tmp_path: Path) -> None:
    trace_path = tmp_path / "t.jsonl"
    trace_path.write_text("")
    custom_index = tmp_path / "custom.idx.jsonl"
    custom_meta = Path(str(custom_index) + ".meta.json")
    custom_index.write_text("")
    custom_meta.write_text('{"schema_version":1,"trace_count":0}')

    result_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=TraceIndexConfig(index_path=custom_index),
    )
    assert result_path == custom_index


@pytest.mark.asyncio
async def test_build_index_from_tiny_fixture(tmp_path: Path, fixtures_dir: Path) -> None:
    src = fixtures_dir / "tiny_traces.jsonl"
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes(src.read_bytes())

    result_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=TraceIndexConfig(),
    )
    assert result_path.exists()
    meta_path = TraceIndexBuilder._meta_path_for(result_path)
    assert meta_path.exists()

    meta = TraceIndexMeta.model_validate_json(meta_path.read_text())
    assert meta.schema_version == 1
    assert meta.trace_count == 3

    rows = [TraceIndexRow.model_validate_json(line) for line in result_path.read_text().splitlines()]
    rows_by_id = {r.trace_id: r for r in rows}
    assert set(rows_by_id) == {"t-aaaa", "t-bbbb", "t-cccc"}

    bb = rows_by_id["t-bbbb"]
    assert bb.has_errors is True
    assert "gpt-5.4" in bb.model_names
    assert bb.total_input_tokens == 200
    assert bb.total_output_tokens == 40

    with trace_path.open("rb") as fh:
        fh.seek(bb.byte_offsets[0])
        blob = fh.read(bb.byte_lengths[0])
    span = json.loads(blob)
    assert span["span_id"] == "s-bbbb-1"
