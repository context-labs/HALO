from __future__ import annotations

from engine.tools.tool_protocol import ToolContext
from engine.traces.models.trace_query_models import (
    CountTracesArguments,
    CountTracesResult,
    DatasetOverviewArguments,
    DatasetOverviewResult,
    QueryTracesArguments,
    QueryTracesResult,
    SearchTraceArguments,
    SearchTraceResult,
    ViewTraceArguments,
    ViewTraceResult,
)


class GetDatasetOverviewTool:
    name = "get_dataset_overview"
    description = "Return high-level stats about the trace dataset: counts, services, models, totals."
    arguments_model = DatasetOverviewArguments
    result_model = DatasetOverviewResult

    async def run(self, tool_context: ToolContext, arguments: DatasetOverviewArguments) -> DatasetOverviewResult:
        store = tool_context.require_trace_store()
        return DatasetOverviewResult(result=store.get_overview(arguments.filters))


class QueryTracesTool:
    name = "query_traces"
    description = "List trace summaries matching filters with pagination."
    arguments_model = QueryTracesArguments
    result_model = QueryTracesResult

    async def run(self, tool_context: ToolContext, arguments: QueryTracesArguments) -> QueryTracesResult:
        store = tool_context.require_trace_store()
        return QueryTracesResult(
            result=store.query_traces(
                filters=arguments.filters,
                limit=arguments.limit,
                offset=arguments.offset,
            )
        )


class CountTracesTool:
    name = "count_traces"
    description = "Count traces matching filters."
    arguments_model = CountTracesArguments
    result_model = CountTracesResult

    async def run(self, tool_context: ToolContext, arguments: CountTracesArguments) -> CountTracesResult:
        store = tool_context.require_trace_store()
        return CountTracesResult(result=store.count_traces(arguments.filters))


class ViewTraceTool:
    name = "view_trace"
    description = "Return all spans of a trace by id."
    arguments_model = ViewTraceArguments
    result_model = ViewTraceResult

    async def run(self, tool_context: ToolContext, arguments: ViewTraceArguments) -> ViewTraceResult:
        store = tool_context.require_trace_store()
        return ViewTraceResult(result=store.view_trace(arguments.trace_id))


class SearchTraceTool:
    name = "search_trace"
    description = "Substring search inside the spans of one trace."
    arguments_model = SearchTraceArguments
    result_model = SearchTraceResult

    async def run(self, tool_context: ToolContext, arguments: SearchTraceArguments) -> SearchTraceResult:
        store = tool_context.require_trace_store()
        return SearchTraceResult(result=store.search_trace(arguments.trace_id, arguments.pattern))
