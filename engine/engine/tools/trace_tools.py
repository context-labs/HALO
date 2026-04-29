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
        "Return high-level stats about the trace dataset: counts, services, models, "
        "totals, `raw_jsonl_bytes`, and a `sample_trace_ids` list (up to 20) of real "
        "matching trace ids — call this before `view_trace` so you have real ids to "
        "pass.\n\n"
        "Indexed `filters` are cheap. `filters.regex_pattern` is the one scan-heavy "
        "filter — use it only after narrowing with the indexed fields or after "
        "confirming via `raw_jsonl_bytes` that the matched dataset is small enough "
        "to scan cheaply."
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
        "List trace summaries matching `filters` with pagination. Each summary "
        "includes `raw_jsonl_bytes` so you can decide whether `view_trace` is safe "
        "before calling it.\n\n"
        "Indexed `filters` are cheap. `filters.regex_pattern` is the one scan-heavy "
        "filter — prefer narrowing with the indexed fields first, or sizing the "
        "dataset via `get_dataset_overview` before issuing it."
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
        "Count traces matching `filters`. `filters.regex_pattern` is the one "
        "scan-heavy filter — same caveats as on `query_traces`/`get_dataset_overview`."
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
        "Return ALL spans of a trace by id. Per-attribute payloads (input.value, "
        "output.value, llm.input_messages, etc.) are head-capped at ~4096 chars each, "
        "with a marker showing the original length.\n\n"
        "If the trace's truncated total exceeds the per-call budget (~150_000 bytes), "
        "this tool returns an empty `spans` list and an `oversized` summary instead "
        "of the full payload. The summary contains span_count, "
        "`truncated_response_bytes`, `response_bytes_budget`, per-span response size "
        "min/median/max in bytes, top_span_names (most frequent span names with "
        "counts), and error_span_count. When you receive an `oversized` response, "
        "DO NOT retry view_trace — switch to `search_trace(trace_id, regex_pattern)` "
        "to surface only the matches you need, then `view_spans(trace_id, span_ids="
        "[...])` for surgical reads (or `search_span` for an individual large span).\n\n"
        "Best used for traces you already know are small (use the `raw_jsonl_bytes` "
        "from `query_traces` summaries to decide). For unknown or large traces, go "
        "straight to search_trace + view_spans."
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
        "truncation as `view_trace`. Use after `search_trace` has surfaced "
        "interesting `span_id`s.\n\n"
        "Avoid calling on individual spans whose `raw_jsonl_bytes` exceeds ~4096, "
        "or where the total selected size exceeds ~150_000. The tool also enforces "
        "these budgets: if the truncated selected spans exceed the per-call byte "
        "budget, `spans` is returned empty and `oversized` carries summary "
        "statistics + a recommendation to use `search_span` for large individual "
        "spans or to call again with a smaller `span_ids` set.\n\n"
        "Pass up to 200 `span_ids`. Span ids that don't match any span in the trace "
        "are silently skipped."
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
        "Regex-search inside one trace. Returns up to `max_matches` `SpanMatchRecord`s "
        "(span metadata + matched text + surrounding context) plus the unbounded "
        "`match_count` and a `has_more` flag.\n\n"
        "`regex_pattern` is a Python regex matched against the raw on-disk JSON, "
        "so it can target attribute keys (`STATUS_CODE_ERROR`, `MaxTurnsExceeded`), "
        "tool names (`spotify__login`), or any literal substring. Invalid regex "
        "fails fast.\n\n"
        "Use this for traces too large for `view_trace`. After identifying interesting "
        "`span_id`s in the records, follow up with `view_spans` for small spans "
        "(`raw_jsonl_bytes` ≤ ~4096) or `search_span` for large individual spans."
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
        "Regex-search inside a single span. Returns up to `max_matches` "
        "`SpanMatchRecord`s with matched text, surrounding context, and span "
        "metadata; `match_count` is the unbounded total.\n\n"
        "Use this when `view_spans` of a particular span is too large (its "
        "`raw_jsonl_bytes` exceeds ~4096, or `view_spans` returned `oversized` "
        "for a set including this span). Pair with the `span_id`s surfaced by "
        "`search_trace`. Invalid regex fails fast."
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
