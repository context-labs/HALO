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
    """Tool wrapper around ``TraceStore.get_overview``: dataset-level rollup of counts and totals."""

    name = "get_dataset_overview"
    description = (
        "Return high-level stats about the trace dataset: counts, services, models, totals."
    )
    arguments_model = DatasetOverviewArguments
    result_model = DatasetOverviewResult

    async def run(
        self, tool_context: ToolContext, arguments: DatasetOverviewArguments
    ) -> DatasetOverviewResult:
        """Compute the overview over the filtered subset of traces."""
        store = tool_context.require_trace_store()
        return DatasetOverviewResult(result=store.get_overview(arguments.filters))


class QueryTracesTool:
    """Tool wrapper around ``TraceStore.query_traces``: paginated TraceSummary listing for filters."""

    name = "query_traces"
    description = "List trace summaries matching filters with pagination."
    arguments_model = QueryTracesArguments
    result_model = QueryTracesResult

    async def run(
        self, tool_context: ToolContext, arguments: QueryTracesArguments
    ) -> QueryTracesResult:
        """Apply filters and slice with limit/offset; ``total`` is the unsliced match count."""
        store = tool_context.require_trace_store()
        return QueryTracesResult(
            result=store.query_traces(
                filters=arguments.filters,
                limit=arguments.limit,
                offset=arguments.offset,
            )
        )


# TODO: Tool definitions should inherent from a base class / be restricted
class CountTracesTool:
    """Tool wrapper around ``TraceStore.count_traces``: cheap count without materializing summaries."""

    name = "count_traces"
    description = "Count traces matching filters."
    arguments_model = CountTracesArguments
    result_model = CountTracesResult

    async def run(
        self, tool_context: ToolContext, arguments: CountTracesArguments
    ) -> CountTracesResult:
        """Return the number of traces matching ``arguments.filters``."""
        store = tool_context.require_trace_store()
        return CountTracesResult(result=store.count_traces(arguments.filters))


class ViewTraceTool:
    """Tool wrapper around ``TraceStore.view_trace``: full typed span list for one trace id."""

    name = "view_trace"
    description = "Return all spans of a trace by id."
    arguments_model = ViewTraceArguments
    result_model = ViewTraceResult

    async def run(
        self, tool_context: ToolContext, arguments: ViewTraceArguments
    ) -> ViewTraceResult:
        """Read all spans for ``trace_id`` from the JSONL via the index byte offsets."""
        store = tool_context.require_trace_store()
        return ViewTraceResult(result=store.view_trace(arguments.trace_id))


class SearchTraceTool:
    """Tool wrapper around ``TraceStore.search_trace``: substring search confined to one trace."""

    name = "search_trace"
    description = "Substring search inside the spans of one trace."
    arguments_model = SearchTraceArguments
    result_model = SearchTraceResult

    async def run(
        self, tool_context: ToolContext, arguments: SearchTraceArguments
    ) -> SearchTraceResult:
        """Return spans (as raw JSON) of ``trace_id`` containing ``pattern`` as a substring."""
        store = tool_context.require_trace_store()
        return SearchTraceResult(result=store.search_trace(arguments.trace_id, arguments.pattern))
