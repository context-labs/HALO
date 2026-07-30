"""The dataset overview pre-aggregates error signatures.

The taxonomy exists so the agent starts failure-mode analysis from a
verified hypothesis set instead of rediscovering the grouping with
dozens of count/search calls (a real run spent ~80 LLM calls deriving
one GROUP BY). Groups are (span_name, status_message) over OTel ERROR
spans, largest first, with distinct-trace counts and example trace ids
the agent can hand straight to ``view_trace``.
"""

import asyncio
from pathlib import Path

import pytest

from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.models.trace_query_models import TraceFilters
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore

FIXTURE = Path(__file__).parents[2] / "fixtures" / "medium_traces.jsonl"


@pytest.fixture(name="store")
def store_fixture(tmp_path: Path) -> TraceStore:
    """Build the sidecar index in tmp (it is a gitignored local artifact)."""
    trace_path = tmp_path / "medium_traces.jsonl"
    trace_path.write_bytes(FIXTURE.read_bytes())
    index_path = asyncio.run(
        TraceIndexBuilder.ensure_index_exists(trace_path=trace_path, config=TraceIndexConfig())
    )
    return TraceStore.load(trace_path=trace_path, index_path=index_path)


def test_overview_carries_error_taxonomy_groups(store: TraceStore) -> None:
    overview = store.get_overview(TraceFilters())

    assert overview.error_taxonomy, "fixture has error traces; taxonomy must not be empty"
    by_key = {(g.span_name, g.status_message): g for g in overview.error_taxonomy}
    boom = by_key[("root", "boom")]
    assert boom.error_span_count == 46
    assert boom.trace_count == 46
    assert 1 <= len(boom.example_trace_ids) <= 3
    # Ordered largest-first by span count.
    counts = [g.error_span_count for g in overview.error_taxonomy]
    assert counts == sorted(counts, reverse=True)


def test_taxonomy_span_totals_match_overview_error_count(store: TraceStore) -> None:
    overview = store.get_overview(TraceFilters())

    assert (
        sum(g.error_span_count for g in overview.error_taxonomy) == overview.otel_error_span_count
    )


def test_taxonomy_example_ids_are_viewable(store: TraceStore) -> None:
    overview = store.get_overview(TraceFilters())

    example = overview.error_taxonomy[0].example_trace_ids[0]
    view = store.view_trace(example)
    assert view.trace_id == example


def test_empty_subset_has_empty_taxonomy(store: TraceStore) -> None:
    overview = store.get_overview(TraceFilters(service_names=["no-such-service"]))

    assert overview.error_taxonomy == []
