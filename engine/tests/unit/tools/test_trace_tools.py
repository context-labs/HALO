from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from engine.tools.tool_protocol import ToolContext
from engine.tools.trace_tools import (
    CountTracesTool,
    GetDatasetOverviewTool,
    QueryTracesTool,
    SearchTraceTool,
    ViewTraceTool,
)
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.models.trace_query_models import (
    CountTracesArguments,
    DatasetOverviewArguments,
    QueryTracesArguments,
    SearchTraceArguments,
    TraceFilters,
    ViewTraceArguments,
)
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore


@pytest_asyncio.fixture
async def ctx(tmp_path: Path, fixtures_dir: Path) -> ToolContext:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = await TraceIndexBuilder.ensure_index_exists(trace_path=trace_path, config=TraceIndexConfig())
    store = TraceStore.load(trace_path=trace_path, index_path=index_path)
    return ToolContext.model_construct(trace_store=store)


@pytest.mark.asyncio
async def test_query_traces_tool(ctx: ToolContext) -> None:
    tool = QueryTracesTool()
    result = await tool.run(ctx, QueryTracesArguments(filters=TraceFilters()))
    assert result.result.total == 3


@pytest.mark.asyncio
async def test_count_traces_tool(ctx: ToolContext) -> None:
    tool = CountTracesTool()
    result = await tool.run(ctx, CountTracesArguments(filters=TraceFilters(has_errors=True)))
    assert result.result.total == 1


@pytest.mark.asyncio
async def test_view_trace_tool(ctx: ToolContext) -> None:
    tool = ViewTraceTool()
    result = await tool.run(ctx, ViewTraceArguments(trace_id="t-aaaa"))
    assert len(result.result.spans) == 2


@pytest.mark.asyncio
async def test_search_trace_tool(ctx: ToolContext) -> None:
    tool = SearchTraceTool()
    result = await tool.run(ctx, SearchTraceArguments(trace_id="t-bbbb", pattern="tool failure"))
    assert result.result.match_count >= 1


@pytest.mark.asyncio
async def test_overview_tool(ctx: ToolContext) -> None:
    tool = GetDatasetOverviewTool()
    result = await tool.run(ctx, DatasetOverviewArguments(filters=TraceFilters()))
    assert result.result.total_traces == 3
