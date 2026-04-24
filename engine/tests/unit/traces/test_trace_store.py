from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore


@pytest_asyncio.fixture
async def built_store(tmp_path: Path, fixtures_dir: Path) -> TraceStore:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    return TraceStore.load(trace_path=trace_path, index_path=index_path)


@pytest.mark.asyncio
async def test_load_sets_trace_count(built_store: TraceStore) -> None:
    assert built_store.trace_count == 3


@pytest.mark.asyncio
async def test_view_trace_returns_span_records(built_store: TraceStore) -> None:
    view = built_store.view_trace("t-bbbb")
    assert view.trace_id == "t-bbbb"
    assert len(view.spans) == 2
    assert view.spans[0].span_id == "s-bbbb-1"
    assert view.spans[1].attributes["llm.model_name"] == "gpt-5.4"


@pytest.mark.asyncio
async def test_view_trace_unknown_raises(built_store: TraceStore) -> None:
    with pytest.raises(KeyError):
        built_store.view_trace("unknown")
