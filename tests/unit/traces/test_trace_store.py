from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.models.trace_query_models import TraceFilters
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore


@pytest_asyncio.fixture
async def built_store(tmp_path: Path, fixtures_dir: Path) -> TraceStore:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    return TraceStore.load(trace_path=trace_path, index_path=index_path)


@pytest.mark.asyncio
async def test_load_sets_trace_count(built_store: TraceStore) -> None:
    assert built_store.trace_count == 3


@pytest.mark.asyncio
async def test_view_trace_returns_span_records(built_store: TraceStore) -> None:
    view = built_store.view_trace("t-bbbb")
    assert view.trace_id == "t-bbbb"
    assert len(view.spans) == 2
    assert view.spans[0].span_id == "s-bbbb-1"
    assert view.spans[1].attributes["llm.model_name"] == "gpt-5.4"
    assert view.oversized is None


@pytest.mark.asyncio
async def test_view_trace_unknown_raises(built_store: TraceStore) -> None:
    with pytest.raises(KeyError):
        built_store.view_trace("unknown")


@pytest.mark.asyncio
async def test_view_spans_returns_only_requested(built_store: TraceStore) -> None:
    view = built_store.view_spans("t-bbbb", ["s-bbbb-2"])
    assert view.trace_id == "t-bbbb"
    assert [s.span_id for s in view.spans] == ["s-bbbb-2"]


@pytest.mark.asyncio
async def test_view_spans_unknown_trace_raises(built_store: TraceStore) -> None:
    with pytest.raises(KeyError):
        built_store.view_spans("unknown", ["s"])


@pytest.mark.asyncio
async def test_query_filter_has_errors_includes_raw_jsonl_bytes(
    built_store: TraceStore,
) -> None:
    result = built_store.query_traces(
        filters=TraceFilters(has_errors=True),
        limit=10,
        offset=0,
    )
    assert result.total == 1
    assert len(result.traces) == 1
    only = result.traces[0]
    assert only.trace_id == "t-bbbb"
    assert only.raw_jsonl_bytes > 0


@pytest.mark.asyncio
async def test_query_filter_model_intersection(built_store: TraceStore) -> None:
    result = built_store.query_traces(
        filters=TraceFilters(model_names=["claude-haiku-4-5"]),
        limit=10,
        offset=0,
    )
    assert {t.trace_id for t in result.traces} == {"t-cccc"}


@pytest.mark.asyncio
async def test_query_traces_with_regex_pattern_narrows_to_matches(
    built_store: TraceStore,
) -> None:
    result = built_store.query_traces(
        filters=TraceFilters(regex_pattern="STATUS_CODE_ERROR"),
    )
    assert {t.trace_id for t in result.traces} == {"t-bbbb"}


@pytest.mark.asyncio
async def test_query_traces_combines_filters_and_regex_with_and_semantics(
    built_store: TraceStore,
) -> None:
    # Indexed filter selects t-aaaa and t-cccc; regex requires the anthropic span.
    result = built_store.query_traces(
        filters=TraceFilters(
            has_errors=False,
            regex_pattern="anthropic.messages.create",
        ),
    )
    assert {t.trace_id for t in result.traces} == {"t-aaaa", "t-cccc"}


@pytest.mark.asyncio
async def test_query_traces_invalid_regex_raises_clearly(built_store: TraceStore) -> None:
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        built_store.query_traces(filters=TraceFilters(regex_pattern="("))


@pytest.mark.asyncio
async def test_count_traces_with_and_without_filter(built_store: TraceStore) -> None:
    assert built_store.count_traces(TraceFilters()).total == 3
    assert built_store.count_traces(TraceFilters(has_errors=True)).total == 1


@pytest.mark.asyncio
async def test_count_traces_with_regex_pattern(built_store: TraceStore) -> None:
    assert built_store.count_traces(TraceFilters(regex_pattern="STATUS_CODE_ERROR")).total == 1
    assert built_store.count_traces(TraceFilters(regex_pattern="claude-haiku-4-5")).total == 1


@pytest.mark.asyncio
async def test_count_traces_invalid_regex_raises_clearly(built_store: TraceStore) -> None:
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        built_store.count_traces(TraceFilters(regex_pattern="["))


@pytest.mark.asyncio
async def test_overview_full_includes_raw_jsonl_bytes(
    built_store: TraceStore,
) -> None:
    ov = built_store.get_overview(TraceFilters())
    assert ov.total_traces == 3
    assert ov.total_spans == 6
    assert "agent-a" in ov.agent_names
    assert "gpt-5.4" in ov.model_names
    assert ov.error_trace_count == 1
    assert ov.total_input_tokens == 100 + 200 + 30
    assert ov.total_output_tokens == 50 + 40 + 10
    assert ov.raw_jsonl_bytes > 0


@pytest.mark.asyncio
async def test_overview_with_regex_pattern_narrows(
    built_store: TraceStore,
) -> None:
    ov = built_store.get_overview(TraceFilters(regex_pattern="STATUS_CODE_ERROR"))
    assert ov.total_traces == 1
    assert ov.error_trace_count == 1
    assert ov.sample_trace_ids == ["t-bbbb"]


@pytest.mark.asyncio
async def test_overview_invalid_regex_raises_clearly(built_store: TraceStore) -> None:
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        built_store.get_overview(TraceFilters(regex_pattern=")"))


@pytest.mark.asyncio
async def test_search_trace_returns_match_records(built_store: TraceStore) -> None:
    result = built_store.search_trace("t-bbbb", "tool failure")
    assert result.match_count == 1
    assert result.returned_match_count == 1
    assert result.has_more is False
    record = result.matches[0]
    assert record.trace_id == "t-bbbb"
    assert record.span_id == "s-bbbb-2"
    assert record.span_index == 1
    assert record.span_name == "openai.chat.completions.create"
    assert record.kind == "SPAN_KIND_CLIENT"
    assert record.status_code == "STATUS_CODE_ERROR"
    assert record.parent_span_id == "s-bbbb-1"
    assert record.raw_jsonl_bytes > 0
    assert record.match_text == "tool failure"
    assert "tool failure" in record.matched_context
    assert record.match_start_char >= 0
    assert record.match_end_char == record.match_start_char + len("tool failure")


@pytest.mark.asyncio
async def test_search_trace_no_match(built_store: TraceStore) -> None:
    result = built_store.search_trace("t-aaaa", "nonexistent-needle")
    assert result.match_count == 0
    assert result.returned_match_count == 0
    assert result.has_more is False
    assert result.matches == []


@pytest.mark.asyncio
async def test_search_trace_max_matches_caps_records_but_keeps_count(
    built_store: TraceStore,
) -> None:
    # ``inference`` appears many times across t-bbbb's spans (every span has
    # ``inference.export.schema_version`` plus several other keys).
    result = built_store.search_trace("t-bbbb", "inference", max_matches=2)
    assert result.returned_match_count == 2
    assert result.match_count > 2
    assert result.has_more is True
    assert len(result.matches) == 2


@pytest.mark.asyncio
async def test_search_trace_invalid_regex_raises_clearly(built_store: TraceStore) -> None:
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        built_store.search_trace("t-aaaa", "(")


@pytest.mark.asyncio
async def test_search_trace_unknown_raises(built_store: TraceStore) -> None:
    with pytest.raises(KeyError):
        built_store.search_trace("unknown", "x")


@pytest.mark.asyncio
async def test_search_span_returns_records_for_one_span(built_store: TraceStore) -> None:
    result = built_store.search_span("t-bbbb", "s-bbbb-2", "tool failure")
    assert result.match_count == 1
    assert result.span_id == "s-bbbb-2"
    record = result.matches[0]
    assert record.span_id == "s-bbbb-2"
    assert record.match_text == "tool failure"


@pytest.mark.asyncio
async def test_search_span_max_matches_caps_records(built_store: TraceStore) -> None:
    result = built_store.search_span("t-bbbb", "s-bbbb-2", "inference", max_matches=1)
    assert result.returned_match_count == 1
    assert result.match_count > 1
    assert result.has_more is True


@pytest.mark.asyncio
async def test_search_span_unknown_span_id_raises(built_store: TraceStore) -> None:
    with pytest.raises(KeyError):
        built_store.search_span("t-bbbb", "s-does-not-exist", "x")


@pytest.mark.asyncio
async def test_search_span_unknown_trace_raises(built_store: TraceStore) -> None:
    with pytest.raises(KeyError):
        built_store.search_span("unknown", "s", "x")


@pytest.mark.asyncio
async def test_search_span_invalid_regex_raises_clearly(built_store: TraceStore) -> None:
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        built_store.search_span("t-bbbb", "s-bbbb-2", "(")


@pytest.mark.asyncio
async def test_render_trace_under_budget(built_store: TraceStore) -> None:
    rendered = built_store.render_trace("t-aaaa", budget=4000)
    assert "t-aaaa" in rendered
    assert "s-aaaa-1" in rendered
    assert "s-aaaa-2" in rendered


@pytest.mark.asyncio
async def test_render_trace_truncates_when_over_budget(built_store: TraceStore) -> None:
    rendered = built_store.render_trace("t-aaaa", budget=200)
    assert rendered.endswith("... [truncated]")
    assert len(rendered) <= 200 + len("... [truncated]")


@pytest.mark.asyncio
async def test_paths_exposed_publicly(built_store: TraceStore) -> None:
    assert built_store.trace_path.is_file()
    assert built_store.index_path.is_file()
