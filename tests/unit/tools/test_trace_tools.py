from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from engine.tools.tool_protocol import ToolContext
from engine.tools.trace_tools import (
    CountTracesTool,
    GetDatasetOverviewTool,
    QueryTracesTool,
    SearchSpanTool,
    SearchTraceTool,
    ViewSpansTool,
    ViewTraceTool,
)
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.models.trace_query_models import (
    CountTracesArguments,
    DatasetOverviewArguments,
    QueryTracesArguments,
    SearchSpanArguments,
    SearchTraceArguments,
    TraceFilters,
    ViewSpansArguments,
    ViewTraceArguments,
)
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore


@pytest_asyncio.fixture
async def ctx(tmp_path: Path, fixtures_dir: Path) -> ToolContext:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    store = TraceStore.load(trace_path=trace_path, index_path=index_path)
    return ToolContext.model_construct(trace_store=store)


@pytest.mark.asyncio
async def test_query_traces_tool(ctx: ToolContext) -> None:
    tool = QueryTracesTool()
    result = await tool.run(ctx, QueryTracesArguments(filters=TraceFilters()))
    assert result.result.total == 3
    # raw_jsonl_bytes is on every summary so the agent can size traces.
    assert all(t.raw_jsonl_bytes > 0 for t in result.result.traces)


@pytest.mark.asyncio
async def test_query_traces_tool_with_regex_pattern(ctx: ToolContext) -> None:
    tool = QueryTracesTool()
    result = await tool.run(
        ctx,
        QueryTracesArguments(filters=TraceFilters(regex_pattern="STATUS_CODE_ERROR")),
    )
    assert {t.trace_id for t in result.result.traces} == {"t-bbbb"}


@pytest.mark.asyncio
async def test_count_traces_tool(ctx: ToolContext) -> None:
    tool = CountTracesTool()
    result = await tool.run(ctx, CountTracesArguments(filters=TraceFilters(has_errors=True)))
    assert result.result.total == 1


@pytest.mark.asyncio
async def test_count_traces_tool_with_regex_pattern(ctx: ToolContext) -> None:
    tool = CountTracesTool()
    result = await tool.run(
        ctx,
        CountTracesArguments(filters=TraceFilters(regex_pattern="claude-haiku-4-5")),
    )
    assert result.result.total == 1


@pytest.mark.asyncio
async def test_view_trace_tool(ctx: ToolContext) -> None:
    tool = ViewTraceTool()
    result = await tool.run(ctx, ViewTraceArguments(trace_id="t-aaaa"))
    assert len(result.result.spans) == 2
    assert result.result.oversized is None


@pytest.mark.asyncio
async def test_view_spans_tool(ctx: ToolContext) -> None:
    tool = ViewSpansTool()
    result = await tool.run(ctx, ViewSpansArguments(trace_id="t-bbbb", span_ids=["s-bbbb-2"]))
    assert [s.span_id for s in result.result.spans] == ["s-bbbb-2"]
    assert result.result.oversized is None


@pytest.mark.asyncio
async def test_search_trace_tool(ctx: ToolContext) -> None:
    tool = SearchTraceTool()
    result = await tool.run(
        ctx,
        SearchTraceArguments(trace_id="t-bbbb", regex_pattern="tool failure"),
    )
    assert result.result.match_count == 1
    assert result.result.matches[0].match_text == "tool failure"
    assert result.result.matches[0].span_id == "s-bbbb-2"


@pytest.mark.asyncio
async def test_search_trace_tool_invalid_regex_raises(ctx: ToolContext) -> None:
    tool = SearchTraceTool()
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        await tool.run(
            ctx,
            SearchTraceArguments(trace_id="t-bbbb", regex_pattern="("),
        )


@pytest.mark.asyncio
async def test_search_span_tool(ctx: ToolContext) -> None:
    tool = SearchSpanTool()
    result = await tool.run(
        ctx,
        SearchSpanArguments(
            trace_id="t-bbbb",
            span_id="s-bbbb-2",
            regex_pattern="tool failure",
        ),
    )
    assert result.result.match_count == 1
    assert result.result.matches[0].match_text == "tool failure"


@pytest.mark.asyncio
async def test_overview_tool(ctx: ToolContext) -> None:
    tool = GetDatasetOverviewTool()
    result = await tool.run(ctx, DatasetOverviewArguments(filters=TraceFilters()))
    assert result.result.total_traces == 3
    assert result.result.raw_jsonl_bytes > 0


@pytest.mark.asyncio
async def test_overview_tool_with_regex_pattern(ctx: ToolContext) -> None:
    tool = GetDatasetOverviewTool()
    result = await tool.run(
        ctx,
        DatasetOverviewArguments(filters=TraceFilters(regex_pattern="STATUS_CODE_ERROR")),
    )
    assert result.result.total_traces == 1
    assert result.result.error_trace_count == 1
