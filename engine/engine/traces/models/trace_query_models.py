from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from engine.traces.models.canonical_span import SpanRecord


class TraceFilters(BaseModel):
    """Common filter set applied across overview/query/count. All fields are optional ANDed predicates."""

    model_config = ConfigDict(extra="forbid")

    has_errors: bool | None = None
    model_names: list[str] | None = None
    service_names: list[str] | None = None
    agent_names: list[str] | None = None
    project_id: str | None = None
    start_time_gte: str | None = None
    end_time_lte: str | None = None


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


class TraceQueryResult(BaseModel):
    """Page of TraceSummaries plus the unsliced match ``total`` so the caller can paginate sensibly."""

    model_config = ConfigDict(extra="forbid")

    traces: list[TraceSummary]
    total: int = Field(ge=0)


class TraceCountResult(BaseModel):
    """Just the count of traces matching a filter set."""

    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=0)


class TraceSearchResult(BaseModel):
    """Substring-search hits within one trace; ``matches`` are raw JSON span lines."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    match_count: int = Field(ge=0)
    matches: list[str]


class TraceView(BaseModel):
    """All canonical SpanRecords belonging to one trace, in file order."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    spans: list[SpanRecord]


class DatasetOverview(BaseModel):
    """Whole-dataset rollup over a filtered subset: counts, time bounds, distinct dims, totals."""

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


class QueryTracesArguments(BaseModel):
    """Tool arguments for ``query_traces``: filter set plus pagination knobs."""

    model_config = ConfigDict(extra="forbid")

    filters: TraceFilters
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class CountTracesArguments(BaseModel):
    """Tool arguments for ``count_traces``: filter set only."""

    model_config = ConfigDict(extra="forbid")

    filters: TraceFilters


class ViewTraceArguments(BaseModel):
    """Tool arguments for ``view_trace``: the trace id to materialize."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str


class SearchTraceArguments(BaseModel):
    """Tool arguments for ``search_trace``: trace id plus literal substring pattern."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    pattern: str


class DatasetOverviewArguments(BaseModel):
    """Tool arguments for ``get_dataset_overview``: filter set applied before rollup."""

    model_config = ConfigDict(extra="forbid")

    filters: TraceFilters


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


class DatasetOverviewResult(BaseModel):
    """Result envelope for ``get_dataset_overview`` — wraps a DatasetOverview under ``result``."""

    model_config = ConfigDict(extra="forbid")

    result: DatasetOverview
