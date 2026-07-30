"""The dataset overview pre-aggregates error signatures.

The taxonomy exists so the agent starts failure-mode analysis from a
verified hypothesis set instead of rediscovering the grouping with
dozens of count/search calls (a real run spent ~80 LLM calls deriving
one GROUP BY). Groups are (span_name, status_message) over OTel ERROR
spans, largest first, with distinct-trace counts and example trace ids
the agent can hand straight to ``view_trace``.
"""

from pathlib import Path

from engine.traces.models.trace_query_models import TraceFilters
from engine.traces.trace_store import TraceStore

FIXTURE = Path(__file__).parents[2] / "fixtures" / "medium_traces.jsonl"


def _store() -> TraceStore:
    return TraceStore.load(FIXTURE, Path(f"{FIXTURE}.engine-index.jsonl"))


def test_overview_carries_error_taxonomy_groups() -> None:
    overview = _store().get_overview(TraceFilters())

    assert overview.error_taxonomy, "fixture has error traces; taxonomy must not be empty"
    by_key = {(g.span_name, g.status_message): g for g in overview.error_taxonomy}
    boom = by_key[("root", "boom")]
    assert boom.error_span_count == 46
    assert boom.trace_count == 46
    assert 1 <= len(boom.example_trace_ids) <= 3
    # Ordered largest-first by span count.
    counts = [g.error_span_count for g in overview.error_taxonomy]
    assert counts == sorted(counts, reverse=True)


def test_taxonomy_span_totals_match_overview_error_count() -> None:
    overview = _store().get_overview(TraceFilters())

    assert (
        sum(g.error_span_count for g in overview.error_taxonomy)
        == overview.otel_error_span_count
    )


def test_taxonomy_example_ids_are_viewable() -> None:
    store = _store()
    overview = store.get_overview(TraceFilters())

    example = overview.error_taxonomy[0].example_trace_ids[0]
    view = store.view_trace(example)
    assert view.trace_id == example


def test_empty_subset_has_empty_taxonomy() -> None:
    overview = _store().get_overview(TraceFilters(service_names=["no-such-service"]))

    assert overview.error_taxonomy == []
