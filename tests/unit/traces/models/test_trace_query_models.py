"""Tests for the bits of ``trace_query_models`` that capture our own design
decisions (custom Field constraints + defaults).

Pydantic's own validation (required-field checks, JSON round-tripping, type
coercion) is covered by Pydantic; basedpyright catches missing or misnamed
fields on the construction side. Both are out of scope here.
"""

from __future__ import annotations

import pytest

from engine.traces.models.trace_query_models import (
    QueryTracesArguments,
    SearchSpanArguments,
    SearchTraceArguments,
    TraceFilters,
    ViewSpansArguments,
    canonical_timestamp,
)


def test_query_arguments_pagination_defaults() -> None:
    args = QueryTracesArguments()
    assert args.limit == 50
    assert args.offset == 0


def test_search_argument_defaults() -> None:
    trace = SearchTraceArguments(trace_id="t", regex_pattern="x")
    span = SearchSpanArguments(trace_id="t", span_id="s", regex_pattern="x")
    for args in (trace, span):
        assert args.context_buffer_chars == 100
        assert args.max_matches == 50


def test_search_argument_bounds() -> None:
    with pytest.raises(ValueError):
        SearchTraceArguments(trace_id="t", regex_pattern="x", max_matches=0)
    with pytest.raises(ValueError):
        SearchTraceArguments(trace_id="t", regex_pattern="x", max_matches=501)
    with pytest.raises(ValueError):
        SearchTraceArguments(trace_id="t", regex_pattern="x", context_buffer_chars=-1)
    with pytest.raises(ValueError):
        SearchTraceArguments(trace_id="t", regex_pattern="x", context_buffer_chars=2_001)


def test_view_spans_arguments_span_id_list_bounds() -> None:
    ViewSpansArguments(trace_id="t", span_ids=["s-0"])
    with pytest.raises(ValueError):
        ViewSpansArguments(trace_id="t", span_ids=[])
    with pytest.raises(ValueError):
        ViewSpansArguments(trace_id="t", span_ids=[f"s-{i}" for i in range(201)])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-04-23T05:00:00.123456789Z", "2026-04-23T05:00:00.123456789Z"),
        ("2026-04-23T05:00:00Z", "2026-04-23T05:00:00.000000000Z"),
        ("2026-04-23T05:00:00.5Z", "2026-04-23T05:00:00.500000000Z"),
        ("2026-04-23T05:00Z", "2026-04-23T05:00:00.000000000Z"),
        ("2026-04-23T05:00:00", "2026-04-23T05:00:00.000000000Z"),
        ("2026-04-23 05:00:00+00:00", "2026-04-23T05:00:00.000000000Z"),
        ("2026-04-23T07:00:00.25+02:00", "2026-04-23T05:00:00.250000000Z"),
        ("2026-04-23T00:30:00-0500", "2026-04-23T05:30:00.000000000Z"),
        ("2026-04-23T23:30:00-01:00", "2026-04-24T00:30:00.000000000Z"),
        ("2026-04-23", "2026-04-23T00:00:00.000000000Z"),
        ("0999-01-01T00:00:00Z", "0999-01-01T00:00:00.000000000Z"),
    ],
)
def test_canonical_timestamp_normalizes_to_stored_span_format(raw: str, expected: str) -> None:
    assert canonical_timestamp(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "yesterday",
        "2026-13-01T00:00:00Z",
        "2026-04-23T25:00:00Z",
        "2026-04-23T05:00:00+24:00",
        "2026-04-23T05:00:00+05:60",
        "2026-04-23T05:00:00+00:99",
        "2026-04-23T05:00:0\u0661Z",
        "9999-12-31T23:00:00-05:00",
        "1714000000",
    ],
)
def test_canonical_timestamp_rejects_unparseable_input(raw: str) -> None:
    with pytest.raises(ValueError, match="invalid ISO-8601 timestamp"):
        canonical_timestamp(raw)


def test_trace_filters_normalize_time_bounds() -> None:
    filters = TraceFilters(
        start_time_gte="2026-04-23T08:00:00+02:00", end_time_lte="2026-04-23T07:00:00Z"
    )
    assert filters.start_time_gte == "2026-04-23T06:00:00.000000000Z"
    assert filters.end_time_lte == "2026-04-23T07:00:00.000000000Z"
    assert TraceFilters().start_time_gte is None
    with pytest.raises(ValueError):
        TraceFilters(start_time_gte="not a time")
