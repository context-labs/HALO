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
    ViewSpansArguments,
    ViewTraceArguments,
    ViewTraceResult,
)


class GetDatasetOverviewTool:
    """Tool wrapper around ``TraceStore.get_overview``: dataset-level rollup of counts and totals."""

    name = "get_dataset_overview"
    description = (
        "Return high-level stats about the trace dataset: counts, services, models, totals, "
        "and a `sample_trace_ids` list (up to 20) of the first matching trace ids — call this "
        "before `view_trace` so you have real ids to pass."
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
    description = (
        "Return ALL spans of a trace by id. Per-attribute payloads (input.value, "
        "output.value, llm.input_messages, etc.) are head-capped at ~4KB each, "
        "with a marker showing the original length.\n\n"
        "If the trace's truncated total exceeds the per-call budget (~150K chars), "
        "this tool returns an empty `spans` list and an `oversized` summary instead "
        "of the full payload. The summary contains span_count, char totals, per-span "
        "size min/median/max, top_span_names (most frequent span names with counts), "
        "and error_span_count. When you receive an `oversized` response, DO NOT retry "
        "view_trace — switch to `search_trace(trace_id, pattern)` to surface only the "
        "spans matching a specific substring, then `view_spans(trace_id, span_ids=[...])` "
        "to read those spans surgically.\n\n"
        "Best used for traces you already know are small (e.g. span_count ≤ ~50 from "
        "a query_traces summary). For unknown or large traces, go straight to "
        "search_trace + view_spans."
    )
    arguments_model = ViewTraceArguments
    result_model = ViewTraceResult

    async def run(
        self, tool_context: ToolContext, arguments: ViewTraceArguments
    ) -> ViewTraceResult:
        """Read all spans for ``trace_id`` from the JSONL via the index byte offsets."""
        store = tool_context.require_trace_store()
        return ViewTraceResult(result=store.view_trace(arguments.trace_id))


class ViewSpansTool:
    """Tool wrapper around ``TraceStore.view_spans``: read a chosen subset of spans by id."""

    name = "view_spans"
    description = (
        "Return only the named spans from a trace, with the same per-attribute "
        "truncation as `view_trace`. Use this after `search_trace` has surfaced "
        "interesting spans to materialize them in full (or as full as the cap "
        "allows) without dragging in the rest of the trace. Pass up to 200 "
        "`span_ids`. Span ids that don't match any span in the trace are silently "
        "skipped."
    )
    arguments_model = ViewSpansArguments
    result_model = ViewTraceResult

    async def run(
        self, tool_context: ToolContext, arguments: ViewSpansArguments
    ) -> ViewTraceResult:
        """Read only the requested spans for ``trace_id`` from the JSONL."""
        store = tool_context.require_trace_store()
        return ViewTraceResult(
            result=store.view_spans(arguments.trace_id, arguments.span_ids)
        )


class SearchTraceTool:
    """Tool wrapper around ``TraceStore.search_trace``: substring search confined to one trace."""

    name = "search_trace"
    description = (
        "Substring-search inside one trace; returns matching spans (with the same "
        "per-attribute truncation as `view_trace`). Pattern matches against the raw "
        "on-disk JSON, so you can target attribute keys (`STATUS_CODE_ERROR`, "
        "`MaxTurnsExceeded`), tool names (`spotify__login`), or any literal substring. "
        "Pair with `view_spans` for a low-context way to inspect a long trace."
    )
    arguments_model = SearchTraceArguments
    result_model = SearchTraceResult

    async def run(
        self, tool_context: ToolContext, arguments: SearchTraceArguments
    ) -> SearchTraceResult:
        """Return spans (as raw JSON) of ``trace_id`` containing ``pattern`` as a substring."""
        store = tool_context.require_trace_store()
        return SearchTraceResult(result=store.search_trace(arguments.trace_id, arguments.pattern))
