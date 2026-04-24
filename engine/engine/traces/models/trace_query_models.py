from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from engine.traces.models.canonical_span import SpanRecord


class TraceFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    has_errors: bool | None = None
    model_names: list[str] | None = None
    service_names: list[str] | None = None
    agent_names: list[str] | None = None
    project_id: str | None = None
    start_time_gte: str | None = None
    end_time_lte: str | None = None


class TraceSummary(BaseModel):
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
    model_config = ConfigDict(extra="forbid")

    traces: list[TraceSummary]
    total: int = Field(ge=0)


class TraceCountResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=0)


class TraceSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    match_count: int = Field(ge=0)
    matches: list[str]


class TraceView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    spans: list[SpanRecord]


class DatasetOverview(BaseModel):
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
    model_config = ConfigDict(extra="forbid")

    filters: TraceFilters
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class CountTracesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filters: TraceFilters


class ViewTraceArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str


class SearchTraceArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    pattern: str


class DatasetOverviewArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filters: TraceFilters


class QueryTracesResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: TraceQueryResult


class CountTracesResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: TraceCountResult


class ViewTraceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: TraceView


class SearchTraceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: TraceSearchResult


class DatasetOverviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: DatasetOverview
