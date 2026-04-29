from __future__ import annotations

import pytest

from engine.traces.models.trace_query_models import (
    CountTracesArguments,
    CountTracesResult,
    DatasetOverview,
    DatasetOverviewArguments,
    DatasetOverviewResult,
    OversizedTraceSummary,
    QueryTracesArguments,
    QueryTracesResult,
    SearchSpanArguments,
    SearchSpanResult,
    SearchTraceArguments,
    SearchTraceResult,
    SpanMatchRecord,
    SpanSearchResult,
    TraceCountResult,
    TraceFilters,
    TraceQueryResult,
    TraceSearchResult,
    TraceSummary,
    TraceView,
    ViewSpansArguments,
    ViewTraceArguments,
    ViewTraceResult,
)


def _match() -> SpanMatchRecord:
    return SpanMatchRecord(
        trace_id="t",
        span_id="s",
        span_index=0,
        span_name="root",
        kind="SPAN_KIND_INTERNAL",
        status_code="STATUS_CODE_OK",
        parent_span_id="",
        raw_jsonl_bytes=512,
        match_text="hit",
        matched_context="...hit...",
        match_start_char=10,
        match_end_char=13,
    )


def test_filters_all_optional_and_include_regex_pattern() -> None:
    f = TraceFilters()
    assert f.has_errors is None
    assert f.model_names is None
    assert f.service_names is None
    # regex_pattern is just another filter on TraceFilters.
    assert f.regex_pattern is None
    assert TraceFilters(regex_pattern="x").regex_pattern == "x"


def test_query_arguments_defaults() -> None:
    args = QueryTracesArguments()
    assert args.limit == 50
    assert args.offset == 0
    assert args.filters == TraceFilters()
    # No content_regex_pattern: regex lives in filters now.
    assert "content_regex_pattern" not in QueryTracesArguments.model_fields


def test_count_arguments_no_content_regex_pattern() -> None:
    assert "content_regex_pattern" not in CountTracesArguments.model_fields


def test_dataset_overview_arguments_no_content_regex_pattern() -> None:
    assert "content_regex_pattern" not in DatasetOverviewArguments.model_fields


def test_search_trace_arguments_uses_regex_pattern() -> None:
    args = SearchTraceArguments(trace_id="t", regex_pattern="STATUS_CODE_ERROR")
    assert args.regex_pattern == "STATUS_CODE_ERROR"
    assert args.context_buffer_chars == 100
    assert args.max_matches == 50


def test_search_span_arguments_defaults() -> None:
    args = SearchSpanArguments(trace_id="t", span_id="s", regex_pattern=".+")
    assert args.regex_pattern == ".+"
    assert args.context_buffer_chars == 100
    assert args.max_matches == 50


def test_search_arguments_validation_bounds() -> None:
    with pytest.raises(ValueError):
        SearchTraceArguments(trace_id="t", regex_pattern="x", max_matches=0)
    with pytest.raises(ValueError):
        SearchTraceArguments(trace_id="t", regex_pattern="x", max_matches=501)
    with pytest.raises(ValueError):
        SearchTraceArguments(trace_id="t", regex_pattern="x", context_buffer_chars=-1)
    with pytest.raises(ValueError):
        SearchTraceArguments(trace_id="t", regex_pattern="x", context_buffer_chars=2_001)


def test_view_spans_arguments_min_and_max_lengths() -> None:
    ViewSpansArguments(trace_id="t", span_ids=["s-0"])
    with pytest.raises(ValueError):
        ViewSpansArguments(trace_id="t", span_ids=[])
    with pytest.raises(ValueError):
        ViewSpansArguments(trace_id="t", span_ids=[f"s-{i}" for i in range(201)])


def test_trace_summary_requires_raw_jsonl_bytes_and_round_trips() -> None:
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
        raw_jsonl_bytes=4321,
    )
    assert TraceSummary.model_validate_json(s.model_dump_json()) == s
    with pytest.raises(ValueError):
        TraceSummary(  # missing raw_jsonl_bytes
            trace_id="t",
            span_count=0,
            start_time="",
            end_time="",
            has_errors=False,
            service_names=[],
            model_names=[],
            total_input_tokens=0,
            total_output_tokens=0,
            agent_names=[],
        )


def test_count_result() -> None:
    r = TraceCountResult(total=7)
    assert r.total == 7


def test_search_result_holds_match_records_and_accounting() -> None:
    r = TraceSearchResult(
        trace_id="t",
        match_count=2,
        returned_match_count=2,
        has_more=False,
        matches=[_match(), _match()],
    )
    assert r.match_count == 2
    assert r.returned_match_count == 2
    assert r.has_more is False
    assert all(isinstance(m, SpanMatchRecord) for m in r.matches)


def test_span_search_result_has_same_shape_with_span_id() -> None:
    r = SpanSearchResult(
        trace_id="t",
        span_id="s",
        match_count=3,
        returned_match_count=2,
        has_more=True,
        matches=[_match(), _match()],
    )
    assert r.has_more is True
    assert r.span_id == "s"


def test_view_has_span_list() -> None:
    v = TraceView(trace_id="t", spans=[])
    assert v.trace_id == "t"
    assert v.oversized is None


def test_oversized_summary_uses_response_bytes_naming() -> None:
    s = OversizedTraceSummary(
        trace_id="t",
        span_count=10,
        truncated_response_bytes=200_000,
        response_bytes_budget=150_000,
        span_response_bytes_min=10,
        span_response_bytes_median=100,
        span_response_bytes_max=80_000,
        top_span_names=[("a", 3), ("b", 2)],
        error_span_count=1,
        recommendation="...",
    )
    assert s.response_bytes_budget == 150_000
    assert s.span_response_bytes_max == 80_000
    assert s.top_span_names[0] == ("a", 3)


def test_dataset_overview_includes_raw_jsonl_bytes() -> None:
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
        raw_jsonl_bytes=12_345,
    )
    assert ov.total_traces == 3
    assert ov.raw_jsonl_bytes == 12_345


def test_result_wrappers_tool_boundary() -> None:
    assert QueryTracesResult(result=TraceQueryResult(traces=[], total=0)).result.total == 0
    assert ViewTraceResult(result=TraceView(trace_id="t", spans=[])).result.trace_id == "t"
    assert CountTracesResult(result=TraceCountResult(total=0)).result.total == 0
    search_envelope = SearchTraceResult(
        result=TraceSearchResult(
            trace_id="t",
            match_count=0,
            returned_match_count=0,
            has_more=False,
            matches=[],
        )
    )
    assert search_envelope.result.match_count == 0
    span_search_envelope = SearchSpanResult(
        result=SpanSearchResult(
            trace_id="t",
            span_id="s",
            match_count=0,
            returned_match_count=0,
            has_more=False,
            matches=[],
        )
    )
    assert span_search_envelope.result.span_id == "s"
    ov = DatasetOverview(
        total_traces=0,
        total_spans=0,
        earliest_start_time="",
        latest_end_time="",
        service_names=[],
        model_names=[],
        agent_names=[],
        error_trace_count=0,
        total_input_tokens=0,
        total_output_tokens=0,
        raw_jsonl_bytes=0,
    )
    assert DatasetOverviewResult(result=ov).result.total_traces == 0
    assert DatasetOverviewArguments().filters.has_errors is None
    assert ViewTraceArguments(trace_id="t").trace_id == "t"
    assert CountTracesArguments().filters.has_errors is None
