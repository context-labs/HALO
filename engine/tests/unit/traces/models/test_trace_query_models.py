from __future__ import annotations

from engine.traces.models.trace_query_models import (
    CountTracesArguments,
    CountTracesResult,
    DatasetOverview,
    DatasetOverviewArguments,
    DatasetOverviewResult,
    QueryTracesArguments,
    QueryTracesResult,
    SearchTraceArguments,
    SearchTraceResult,
    TraceCountResult,
    TraceFilters,
    TraceQueryResult,
    TraceSearchResult,
    TraceSummary,
    TraceView,
    ViewTraceArguments,
    ViewTraceResult,
)


def test_filters_all_optional() -> None:
    f = TraceFilters()
    assert f.has_errors is None
    assert f.model_names is None
    assert f.service_names is None


def test_query_arguments_defaults() -> None:
    args = QueryTracesArguments(filters=TraceFilters())
    assert args.limit == 50
    assert args.offset == 0


def test_trace_summary_roundtrip() -> None:
    s = TraceSummary(
        trace_id="t",
        span_count=2,
        start_time="a",
        end_time="b",
        has_errors=False,
        service_names=["svc"],
        model_names=["m"],
        total_input_tokens=1,
        total_output_tokens=2,
        agent_names=["a"],
    )
    assert TraceSummary.model_validate_json(s.model_dump_json()) == s


def test_count_result() -> None:
    r = TraceCountResult(total=7)
    assert r.total == 7


def test_search_result_holds_matches() -> None:
    r = TraceSearchResult(trace_id="t", match_count=2, matches=["hit1", "hit2"])
    assert r.match_count == 2


def test_view_has_span_list() -> None:
    v = TraceView(trace_id="t", spans=[])
    assert v.trace_id == "t"


def test_dataset_overview() -> None:
    ov = DatasetOverview(
        total_traces=3,
        total_spans=6,
        earliest_start_time="a",
        latest_end_time="b",
        service_names=["svc"],
        model_names=["m"],
        agent_names=["a"],
        error_trace_count=1,
        total_input_tokens=330,
        total_output_tokens=100,
    )
    assert ov.total_traces == 3


def test_result_wrappers_tool_boundary() -> None:
    assert QueryTracesResult(result=TraceQueryResult(traces=[], total=0)).result.total == 0
    assert ViewTraceResult(result=TraceView(trace_id="t", spans=[])).result.trace_id == "t"
    assert CountTracesResult(result=TraceCountResult(total=0)).result.total == 0
    assert SearchTraceResult(result=TraceSearchResult(trace_id="t", match_count=0, matches=[])).result.match_count == 0
    ov = DatasetOverview(
        total_traces=0, total_spans=0, earliest_start_time="", latest_end_time="",
        service_names=[], model_names=[], agent_names=[], error_trace_count=0,
        total_input_tokens=0, total_output_tokens=0,
    )
    assert DatasetOverviewResult(result=ov).result.total_traces == 0
    assert DatasetOverviewArguments(filters=TraceFilters()).filters.has_errors is None
    assert SearchTraceArguments(trace_id="t", pattern="x").pattern == "x"
    assert ViewTraceArguments(trace_id="t").trace_id == "t"
    assert CountTracesArguments(filters=TraceFilters()).filters.has_errors is None
