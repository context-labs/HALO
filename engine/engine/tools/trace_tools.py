from __future__ import annotations

from engine.tools.tool_protocol import ToolContext
from engine.traces.models.trace_query_models import (
    CountTracesArguments,
    CountTracesResult,
    DatasetOverviewArguments,
    DatasetOverviewResult,
    QueryTracesArguments,
    QueryTracesResult,
    SearchSpanArguments,
    SearchSpanResult,
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
        "Dataset rollup: counts, services, models, totals, `raw_jsonl_bytes`, and "
        "`sample_trace_ids` (real ids to pass to view/search tools). Call this first "
        "to size the dataset. `filters.regex_pattern` is opt-in raw-span scanning; "
        "narrow with indexed filter fields first."
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
    description = (
        "Paginated trace summaries; each carries `raw_jsonl_bytes` so you can size "
        "traces before calling `view_trace`. Before adding `filters.regex_pattern` "
        "(opt-in raw-span scanning), narrow with indexed filter fields and confirm "
        "the candidate count via `get_dataset_overview`/`count_traces`."
    )
    arguments_model = QueryTracesArguments
    result_model = QueryTracesResult

    async def run(
        self, tool_context: ToolContext, arguments: QueryTracesArguments
    ) -> QueryTracesResult:
        """Apply filters and slice with limit/offset."""
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
    description = (
        "Count traces matching `filters`. Use to size a candidate set (e.g. via "
        "indexed filters) before adding `filters.regex_pattern` (opt-in raw-span "
        "scanning)."
    )
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
        "Return all spans of a trace (per-attribute payloads head-capped ~4KB). "
        "If the response would exceed ~150KB, returns `oversized` summary instead — "
        "switch to `search_trace` + `view_spans`/`search_span`. Use `raw_jsonl_bytes` "
        "from `query_traces` to decide if a trace is small enough."
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
        "Return named spans (up to 200) at a 16KB per-attribute cap (4× `view_trace`). "
        "If the response would exceed ~150KB, returns `oversized` summary — switch to "
        "`search_span` for large individual spans or call with a smaller set."
    )
    arguments_model = ViewSpansArguments
    result_model = ViewTraceResult

    async def run(
        self, tool_context: ToolContext, arguments: ViewSpansArguments
    ) -> ViewTraceResult:
        """Read only the requested spans for ``trace_id`` from the JSONL."""
        store = tool_context.require_trace_store()
        return ViewTraceResult(result=store.view_spans(arguments.trace_id, arguments.span_ids))


class SearchTraceTool:
    """Tool wrapper around ``TraceStore.search_trace``: regex match records confined to one trace."""

    name = "search_trace"
    description = (
        "Regex-search a trace (Python regex over raw span JSON). Returns up to "
        "`max_matches` `SpanMatchRecord`s (span metadata + matched text + context) "
        "plus unbounded `match_count` and `has_more`. Follow up with `view_spans` or "
        "`search_span` on the returned `span_id`s."
    )
    arguments_model = SearchTraceArguments
    result_model = SearchTraceResult

    async def run(
        self, tool_context: ToolContext, arguments: SearchTraceArguments
    ) -> SearchTraceResult:
        """Run a bounded regex search across all spans of ``trace_id``."""
        store = tool_context.require_trace_store()
        return SearchTraceResult(
            result=store.search_trace(
                trace_id=arguments.trace_id,
                regex_pattern=arguments.regex_pattern,
                context_buffer_chars=arguments.context_buffer_chars,
                max_matches=arguments.max_matches,
            )
        )


class SearchSpanTool:
    """Tool wrapper around ``TraceStore.search_span``: regex match records inside a single span."""

    name = "search_span"
    description = (
        "Regex-search a single span. Same shape as `search_trace`. Use when one span "
        "is too large for `view_spans`."
    )
    arguments_model = SearchSpanArguments
    result_model = SearchSpanResult

    async def run(
        self, tool_context: ToolContext, arguments: SearchSpanArguments
    ) -> SearchSpanResult:
        """Run a bounded regex search inside a single span of ``trace_id``."""
        store = tool_context.require_trace_store()
        return SearchSpanResult(
            result=store.search_span(
                trace_id=arguments.trace_id,
                span_id=arguments.span_id,
                regex_pattern=arguments.regex_pattern,
                context_buffer_chars=arguments.context_buffer_chars,
                max_matches=arguments.max_matches,
            )
        )
