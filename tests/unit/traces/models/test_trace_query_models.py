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
    ViewSpansArguments,
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
