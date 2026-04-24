from __future__ import annotations

from pathlib import Path

import pytest

from engine.traces.models.trace_index_config import TraceIndexConfig
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
