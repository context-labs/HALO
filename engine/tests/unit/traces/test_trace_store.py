from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.models.trace_query_models import TraceFilters
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


@pytest.mark.asyncio
async def test_query_filter_has_errors(built_store: TraceStore) -> None:
    result = built_store.query_traces(
        filters=TraceFilters(has_errors=True),
        limit=10,
        offset=0,
    )
    assert result.total == 1
    assert len(result.traces) == 1
    assert result.traces[0].trace_id == "t-bbbb"


@pytest.mark.asyncio
async def test_query_filter_model_intersection(built_store: TraceStore) -> None:
    result = built_store.query_traces(
        filters=TraceFilters(model_names=["claude-haiku-4-5"]),
        limit=10,
        offset=0,
    )
    assert {t.trace_id for t in result.traces} == {"t-cccc"}


@pytest.mark.asyncio
async def test_count_traces_with_and_without_filter(built_store: TraceStore) -> None:
    assert built_store.count_traces(TraceFilters()).total == 3
    assert built_store.count_traces(TraceFilters(has_errors=True)).total == 1


@pytest.mark.asyncio
async def test_overview_full(built_store: TraceStore) -> None:
    ov = built_store.get_overview(TraceFilters())
    assert ov.total_traces == 3
    assert ov.total_spans == 6
    assert "agent-a" in ov.agent_names
    assert "gpt-5.4" in ov.model_names
    assert ov.error_trace_count == 1
    assert ov.total_input_tokens == 100 + 200 + 30
    assert ov.total_output_tokens == 50 + 40 + 10


@pytest.mark.asyncio
async def test_search_returns_matches(built_store: TraceStore) -> None:
    result = built_store.search_trace("t-bbbb", "tool failure")
    assert result.match_count >= 1
    assert any("tool failure" in m for m in result.matches)


@pytest.mark.asyncio
async def test_search_no_match(built_store: TraceStore) -> None:
    result = built_store.search_trace("t-aaaa", "nonexistent-needle")
    assert result.match_count == 0
    assert result.matches == []


@pytest.mark.asyncio
async def test_render_trace_under_budget(built_store: TraceStore) -> None:
    rendered = built_store.render_trace("t-aaaa", budget=4000)
    assert "t-aaaa" in rendered
    assert "s-aaaa-1" in rendered
    assert "s-aaaa-2" in rendered


@pytest.mark.asyncio
async def test_render_trace_truncates_when_over_budget(built_store: TraceStore) -> None:
    rendered = built_store.render_trace("t-aaaa", budget=200)
    assert rendered.endswith("... [truncated]")
    assert len(rendered) <= 200 + len("... [truncated]")


@pytest.mark.asyncio
async def test_paths_exposed_publicly(built_store: TraceStore) -> None:
    assert built_store.trace_path.is_file()
    assert built_store.index_path.is_file()
