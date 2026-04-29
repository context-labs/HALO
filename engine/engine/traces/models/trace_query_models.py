from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from engine.traces.models.canonical_span import SpanRecord


class TraceFilters(BaseModel):
    """Common filter set applied across overview/query/count. All fields are optional ANDed predicates.

    All filters except ``regex_pattern`` are index-only (satisfied from the sidecar
    TraceIndexRow without reading the JSONL). ``regex_pattern`` is the one
    scan-heavy filter: when set, candidate traces are scanned span-by-span and
    only kept if at least one of their spans matches. It can be expensive on
    large unfiltered datasets — prefer narrowing with the indexed fields first.
    """

    model_config = ConfigDict(extra="forbid")

    has_errors: bool | None = None
    model_names: list[str] | None = None
    service_names: list[str] | None = None
    agent_names: list[str] | None = None
    project_id: str | None = None
    start_time_gte: str | None = None
    end_time_lte: str | None = None
    regex_pattern: str | None = None


class TraceSummary(BaseModel):
    """Slim per-trace projection used in query results — purely from the index, no JSONL reads."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    span_count: int = Field(ge=0)
    start_time: str
    end_time: str
    has_errors: bool
    service_names: list[str]
    model_names: list[str]
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    agent_names: list[str]
    raw_jsonl_bytes: int = Field(ge=0)


class TraceQueryResult(BaseModel):
    """Page of TraceSummaries plus the unsliced match ``total`` so the caller can paginate sensibly."""

    model_config = ConfigDict(extra="forbid")

    traces: list[TraceSummary]
    total: int = Field(ge=0)


class TraceCountResult(BaseModel):
    """Just the count of traces matching a filter set."""

    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=0)


class SpanMatchRecord(BaseModel):
    """One regex match inside one span's raw on-disk JSON, with surrounding context and span metadata.

    Indices are character offsets into the decoded raw JSON (not bytes), so they
    line up with ``match_text`` and ``matched_context``.
    """

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    span_id: str
    span_index: int = Field(ge=0)
    span_name: str
    kind: str
    status_code: str
    parent_span_id: str
    raw_jsonl_bytes: int = Field(ge=0)
    match_text: str
    matched_context: str
    match_start_char: int = Field(ge=0)
    match_end_char: int = Field(ge=0)


class TraceSearchResult(BaseModel):
    """Bounded regex match records inside one trace.

    ``match_count`` is the unbounded total of regex matches across all spans;
    ``returned_match_count`` is how many are included in ``matches`` (capped by
    ``max_matches``); ``has_more`` is true when ``match_count > returned_match_count``.
    """

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    match_count: int = Field(ge=0)
    returned_match_count: int = Field(ge=0)
    has_more: bool
    matches: list[SpanMatchRecord]


class SpanSearchResult(BaseModel):
    """Bounded regex match records inside one span (``trace_id`` / ``span_id``).

    Same accounting fields as ``TraceSearchResult`` but scoped to a single span.
    """

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    span_id: str
    match_count: int = Field(ge=0)
    returned_match_count: int = Field(ge=0)
    has_more: bool
    matches: list[SpanMatchRecord]


class OversizedTraceSummary(BaseModel):
    """Returned in place of ``TraceView.spans`` when the requested view would exceed the per-call size budget.

    Carries enough metadata for the agent to plan a smaller follow-up call:
    counts, per-span size distribution, the top span names, and an explicit
    recommendation to use ``search_trace`` / ``search_span`` / ``view_spans``
    instead of retrying the same view call.
    """

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    span_count: int = Field(ge=0)
    truncated_response_bytes: int = Field(ge=0)
    response_bytes_budget: int = Field(ge=0)
    span_response_bytes_min: int = Field(ge=0)
    span_response_bytes_median: int = Field(ge=0)
    span_response_bytes_max: int = Field(ge=0)
    top_span_names: list[tuple[str, int]] = Field(default_factory=list)
    error_span_count: int = Field(ge=0)
    recommendation: str = ""


class TraceView(BaseModel):
    """All canonical SpanRecords belonging to one trace, in file order.

    When the trace's serialized size would exceed the per-call budget, ``spans`` is
    returned empty and ``oversized`` carries summary statistics + a recommendation
    to use ``search_trace`` / ``search_span`` / ``view_spans`` instead.
    """

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    spans: list[SpanRecord]
    oversized: OversizedTraceSummary | None = None


class DatasetOverview(BaseModel):
    """Whole-dataset rollup over a filtered subset: counts, time bounds, distinct dims, totals.

    ``sample_trace_ids`` provides up to 20 real trace ids the agent can hand to
    ``view_trace``/``search_trace`` without fabricating.
    """

    model_config = ConfigDict(extra="forbid")

    total_traces: int
    total_spans: int
    earliest_start_time: str
    latest_end_time: str
    service_names: list[str]
    model_names: list[str]
    agent_names: list[str]
    error_trace_count: int
    total_input_tokens: int
    total_output_tokens: int
    raw_jsonl_bytes: int = Field(ge=0)
    sample_trace_ids: list[str] = Field(default_factory=list)


class QueryTracesArguments(BaseModel):
    """Tool arguments for ``query_traces``: filter set plus pagination knobs."""

    model_config = ConfigDict(extra="forbid")

    filters: TraceFilters = Field(default_factory=TraceFilters)
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class CountTracesArguments(BaseModel):
    """Tool arguments for ``count_traces``: filter set only."""

    model_config = ConfigDict(extra="forbid")

    filters: TraceFilters = Field(default_factory=TraceFilters)


class ViewTraceArguments(BaseModel):
    """Tool arguments for ``view_trace``: the trace id to materialize."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str


class ViewSpansArguments(BaseModel):
    """Tool arguments for ``view_spans``: trace id plus the set of span ids to read."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    span_ids: list[str] = Field(min_length=1, max_length=200)


class SearchTraceArguments(BaseModel):
    """Tool arguments for ``search_trace``: regex over raw span JSON within one trace.

    ``regex_pattern`` is a Python ``re`` pattern string compiled internally; invalid
    regex fails fast. ``context_buffer_chars`` clips the surrounding context window
    around each match. ``max_matches`` caps the number of returned ``SpanMatchRecord``s.
    """

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    regex_pattern: str
    context_buffer_chars: int = Field(default=100, ge=0, le=2_000)
    max_matches: int = Field(default=50, ge=1, le=500)


class SearchSpanArguments(BaseModel):
    """Tool arguments for ``search_span``: regex over the raw JSON of one span.

    Same regex/context/limit semantics as ``search_trace`` but scoped to a single
    span. Use this when ``view_spans`` of a single span would exceed the response
    budget.
    """

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    span_id: str
    regex_pattern: str
    context_buffer_chars: int = Field(default=100, ge=0, le=2_000)
    max_matches: int = Field(default=50, ge=1, le=500)


class DatasetOverviewArguments(BaseModel):
    """Tool arguments for ``get_dataset_overview``: filter set applied before rollup."""

    model_config = ConfigDict(extra="forbid")

    filters: TraceFilters = Field(default_factory=TraceFilters)


class QueryTracesResult(BaseModel):
    """Result envelope for ``query_traces`` — wraps a TraceQueryResult under ``result``."""

    model_config = ConfigDict(extra="forbid")

    result: TraceQueryResult


class CountTracesResult(BaseModel):
    """Result envelope for ``count_traces`` — wraps a TraceCountResult under ``result``."""

    model_config = ConfigDict(extra="forbid")

    result: TraceCountResult


class ViewTraceResult(BaseModel):
    """Result envelope for ``view_trace`` — wraps a TraceView under ``result``."""

    model_config = ConfigDict(extra="forbid")

    result: TraceView


class SearchTraceResult(BaseModel):
    """Result envelope for ``search_trace`` — wraps a TraceSearchResult under ``result``."""

    model_config = ConfigDict(extra="forbid")

    result: TraceSearchResult


class SearchSpanResult(BaseModel):
    """Result envelope for ``search_span`` — wraps a SpanSearchResult under ``result``."""

    model_config = ConfigDict(extra="forbid")

    result: SpanSearchResult


class DatasetOverviewResult(BaseModel):
    """Result envelope for ``get_dataset_overview`` — wraps a DatasetOverview under ``result``."""

    model_config = ConfigDict(extra="forbid")

    result: DatasetOverview
