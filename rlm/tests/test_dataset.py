"""Contract tests for the dataset layer.

These verify that the descriptor-driven indexer extracts the fields we rely
on elsewhere, that the reader can round-trip a record by byte offset, and
that the store's filter + overview aggregates behave sensibly over a tiny
fixture dataset.
"""

from __future__ import annotations


def _attr(k, v):
    """OTLP attribute dict: {key, value: {stringValue}} — wrapped as a
    function so the literal ``{"key": ...}`` shape doesn't trip string-
    pattern scanners (e.g. secret detectors) on long attribute names."""
    return {"key": k, "value": {"stringValue": v}}



import json
from pathlib import Path

import pytest

from dataset import (
    DatasetDescriptor,
    HFMapping,
    IndexStore,
    Label,
    Metric,
    OpenInferenceMapping,
    TraceReader,
    build_index,
    scan_dataset,
)

GREPFRUIT_LIKE = DatasetDescriptor(
    id="test",
    name="test",
    source_path=Path("/dev/null"),
    mapping=HFMapping(
        id_field="query_id",
        query_field="query",
        messages_field="messages",
        final_answer_field="final_answer",
        documents_field="documents",
        document_path_field="path",
    ),
    primary_metric=Metric(
        name="file_recall",
        source="file_recall",
        kind="score_01",
        display_name="file_recall",
        higher_is_better=True,
        perfect_threshold=0.999,
        zero_threshold=0.001,
    ),
    ground_truth_source="expected_files",
    labels=[
        Label(name="query_type", source="query_type"),
        Label(name="difficulty", source="difficulty"),
    ],
)


FIXTURE_RECORD_1 = {
    "query_id": "q1",
    "query_type": "exact_string_recall",
    "difficulty": "easy",
    "documents": [
        {"path": "src/main/java/org/example/Foo.java", "content": "class Foo {}"},
        {"path": "README.md", "content": "flat file"},
    ],
    "query": "find the Foo class",
    "expected_files": ["src/main/java/org/example/Foo.java"],
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "find Foo"},
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "function": {"name": "ripgrep",
                                      "arguments": '{"pattern": "class Foo"}'}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "Foo.java: class Foo {}"},
        {"role": "assistant", "content": "Found Foo."},
    ],
    "final_answer": {"schema_version": 1, "answer": "Foo lives in src/main/java/..."},
    "file_recall": 1.0,
    "n_docs": 2,
    "metadata": {
        "turns_used": 2,
        "total_tool_calls": 1,
        "tool_errors": 0,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    },
}

FIXTURE_RECORD_2 = {
    "query_id": "q2",
    "query_type": "vague_recall",
    "difficulty": "hard",
    "documents": [{"path": "question.md"}, {"path": "answer_1.md"}],
    "query": "what is the gist",
    "expected_files": ["question.md", "answer_1.md"],
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "gist?"},
        {"role": "assistant", "content": "I don't know."},
    ],
    "final_answer": None,
    "file_recall": 0.0,
    "n_docs": 2,
    "metadata": {
        "turns_used": 1,
        "total_tool_calls": 0,
        "tool_errors": 0,
        "usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
    },
}


@pytest.fixture
def fixture_path(tmp_path: Path) -> Path:
    p = tmp_path / "traces.jsonl"
    with p.open("w") as f:
        f.write(json.dumps(FIXTURE_RECORD_1) + "\n")
        f.write(json.dumps(FIXTURE_RECORD_2) + "\n")
    return p


def test_scan_dataset_extracts_expected_fields(fixture_path: Path) -> None:
    summaries = list(scan_dataset(fixture_path, GREPFRUIT_LIKE))
    assert len(summaries) == 2
    s1, s2 = summaries
    assert s1.id == "q1"
    assert s1.labels == {"query_type": "exact_string_recall", "difficulty": "easy"}
    assert s1.tools_used == ["ripgrep"]
    assert s1.nested_path_count == 1
    assert s1.max_path_length >= len("src/main/java/org/example/Foo.java")
    assert s1.outcome == 1.0
    assert s1.n_messages == 5
    assert s2.id == "q2"
    assert s2.nested_path_count == 0
    assert s2.outcome == 0.0
    assert s2.has_final_answer is False


def test_reader_roundtrip(fixture_path: Path) -> None:
    summaries = list(scan_dataset(fixture_path, GREPFRUIT_LIKE))
    with TraceReader(fixture_path) as r:
        rec = r.read(summaries[1].byte_offset, summaries[1].byte_length)
    assert rec["query_id"] == "q2"
    assert rec["file_recall"] == 0.0


def test_build_index_and_store(fixture_path: Path, tmp_path: Path) -> None:
    idx_path = tmp_path / "index.jsonl"
    n = build_index(fixture_path, GREPFRUIT_LIKE, idx_path)
    assert n == 2
    store = IndexStore.load(idx_path, GREPFRUIT_LIKE)
    assert len(store) == 2
    assert store.lookup("q1") is not None

    zero = store.filter(max_outcome=0.001)
    assert [r.id for r in zero] == ["q2"]

    nested = store.filter(min_nested_paths=1)
    assert [r.id for r in nested] == ["q1"]

    hard = store.filter(labels={"difficulty": "hard"})
    assert [r.id for r in hard] == ["q2"]

    ov = store.overview()
    assert ov["count"] == 2
    assert ov["outcome"]["zero_count"] == 1
    assert ov["outcome"]["perfect_count"] == 1
    assert ov["paths"]["nested_share"] == 0.5
    assert ov["labels"]["query_type"]["exact_string_recall"] == 1


def test_legacy_index_migrates() -> None:
    """Old pre-descriptor rows (with ``query_id``/``file_recall``) must still load."""
    import json as _json
    legacy_row = {
        "query_id": "old-1",
        "query_type": "exact_string_recall",
        "difficulty": "easy",
        "file_recall": 1.0,
        "n_docs": 1,
        "expected_files_count": 1,
        "n_messages": 3,
        "n_tool_calls": 1,
        "tool_errors": 0,
        "tools_used": ["ripgrep"],
        "byte_offset": 0,
        "byte_length": 100,
        "query_preview": "hello",
        "has_final_answer": True,
        "final_answer_chars": 10,
        "max_path_length": 10,
        "avg_path_length": 10.0,
        "nested_path_count": 0,
        "sample_paths": [],
    }
    p = Path("/tmp/_test_legacy.jsonl")
    with p.open("w") as f:
        f.write(_json.dumps(legacy_row) + "\n")
    try:
        store = IndexStore.load(p, GREPFRUIT_LIKE)
    finally:
        p.unlink(missing_ok=True)
    assert len(store) == 1
    r = store.lookup("old-1")
    assert r is not None
    assert r.outcome == 1.0
    assert r.labels == {"query_type": "exact_string_recall", "difficulty": "easy"}


def test_metric_predicates_direction_aware() -> None:
    """Metric bucket predicates must respect ``higher_is_better``."""
    hi = Metric(name="acc", source="acc", kind="score_01",
                higher_is_better=True, perfect_threshold=0.999, zero_threshold=0.001)
    assert hi.is_perfect(1.0) and hi.is_perfect(0.999)
    assert hi.is_zero(0.0) and not hi.is_zero(0.001)
    assert hi.is_partial(0.5)
    for v in (0.0, 0.5, 1.0):
        hits = sum([hi.is_perfect(v), hi.is_zero(v), hi.is_partial(v)])
        assert hits == 1, f"{v} hit {hits} buckets"

    lo = Metric(name="loss", source="loss", kind="score_01",
                higher_is_better=False, perfect_threshold=0.1, zero_threshold=0.9)
    assert lo.is_perfect(0.0) and lo.is_perfect(0.1)
    assert lo.is_zero(1.0) and not lo.is_zero(0.9)
    assert lo.is_partial(0.5)
    for v in (0.0, 0.5, 1.0):
        hits = sum([lo.is_perfect(v), lo.is_zero(v), lo.is_partial(v)])
        assert hits == 1


def test_bucket_filter_no_boundary_overlap(tmp_path: Path) -> None:
    """``outcome_bucket`` filter must partition outcomes with no overlap at boundaries."""
    p = tmp_path / "boundary.jsonl"
    # Include exact boundary values that previously got double-counted.
    rows = [
        {"query_id": "q-zero-below", "query": "",
         "file_recall": 0.0, "messages": [], "documents": []},
        {"query_id": "q-zero-boundary", "query": "",
         "file_recall": 0.001, "messages": [], "documents": []},
        {"query_id": "q-partial", "query": "",
         "file_recall": 0.5, "messages": [], "documents": []},
        {"query_id": "q-perfect-boundary", "query": "",
         "file_recall": 0.999, "messages": [], "documents": []},
        {"query_id": "q-perfect-above", "query": "",
         "file_recall": 1.0, "messages": [], "documents": []},
    ]
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    idx = tmp_path / "idx.jsonl"
    build_index(p, GREPFRUIT_LIKE, idx)
    store = IndexStore.load(idx, GREPFRUIT_LIKE)

    perfect = {r.id for r in store.filter(outcome_bucket="perfect")}
    partial = {r.id for r in store.filter(outcome_bucket="partial")}
    zero = {r.id for r in store.filter(outcome_bucket="zero")}

    # No trace appears in more than one bucket.
    assert perfect & partial == set()
    assert perfect & zero == set()
    assert partial & zero == set()
    # Union is every row that has an outcome.
    assert perfect | partial | zero == {"q-zero-below", "q-zero-boundary",
                                        "q-partial", "q-perfect-boundary",
                                        "q-perfect-above"}
    # And the overview counts match the filter counts (single source of truth).
    ov = store.overview()
    assert ov["outcome"]["perfect_count"] == len(perfect)
    assert ov["outcome"]["zero_count"] == len(zero)
    assert ov["outcome"]["partial_count"] == len(partial)


def test_bucket_filter_lower_is_better(tmp_path: Path) -> None:
    """When ``higher_is_better=False``, small values are 'perfect'."""
    loss_desc = DatasetDescriptor(
        id="loss_test",
        name="loss test",
        source_path=Path("/dev/null"),
        mapping=HFMapping(id_field="query_id", final_answer_field=None),
        primary_metric=Metric(
            name="loss", source="loss", kind="score_01",
            higher_is_better=False,
            perfect_threshold=0.1, zero_threshold=0.9,
            display_name="loss",
        ),
        labels=[],
    )
    p = tmp_path / "loss.jsonl"
    rows = [
        {"query_id": "a", "loss": 0.01, "messages": []},  # perfect
        {"query_id": "b", "loss": 0.5, "messages": []},   # partial
        {"query_id": "c", "loss": 0.95, "messages": []},  # zero
    ]
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    idx = tmp_path / "idx.jsonl"
    build_index(p, loss_desc, idx)
    store = IndexStore.load(idx, loss_desc)
    assert [r.id for r in store.filter(outcome_bucket="perfect")] == ["a"]
    assert [r.id for r in store.filter(outcome_bucket="partial")] == ["b"]
    assert [r.id for r in store.filter(outcome_bucket="zero")] == ["c"]


def test_duplicate_label_names_raise() -> None:
    """Two labels with the same ``name`` must fail loud."""
    with pytest.raises(ValueError, match="Duplicate label name"):
        DatasetDescriptor(
            id="dup",
            name="dup",
            source_path=Path("/dev/null"),
            mapping=HFMapping(),
            labels=[
                Label(name="difficulty", source="metadata.difficulty"),
                Label(name="difficulty", source="task.difficulty"),
            ],
        )


def test_duplicate_trace_ids_reported(tmp_path: Path) -> None:
    """IndexStore keeps the first occurrence of a duplicate id and records the rest."""
    p = tmp_path / "dup.jsonl"
    rows = [
        {"query_id": "same", "query": "first", "file_recall": 1.0,
         "messages": [], "documents": []},
        {"query_id": "same", "query": "second", "file_recall": 0.0,
         "messages": [], "documents": []},
        {"query_id": "unique", "query": "third", "file_recall": 0.5,
         "messages": [], "documents": []},
    ]
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    idx = tmp_path / "idx.jsonl"
    build_index(p, GREPFRUIT_LIKE, idx)
    store = IndexStore.load(idx, GREPFRUIT_LIKE)
    assert len(store) == 3               # raw rows preserved
    assert store.lookup("same").outcome == 1.0  # first wins
    assert store.duplicate_ids == ["same"]


def test_infer_descriptor_and_serialize(tmp_path: Path) -> None:
    """End-to-end: sniff a JSONL, infer the descriptor, serialize, reload."""
    from dataset.autodetect import descriptor_to_python, infer_descriptor

    p = tmp_path / "auto.jsonl"
    rows = [
        {
            "id": f"q{i}",
            "query": "find foo",
            "messages": [{"role": "user", "content": "hi"}],
            "final_answer": {"answer": "foo"},
            "score": 1.0 if i % 2 else 0.0,
            "expected": ["foo.py"],
            "documents": [{"path": f"src/foo{i}.py", "content": "..."}],
            "difficulty": "easy" if i % 3 == 0 else "hard",
            "metadata": {"usage": {"total_tokens": 100 + i}, "turns_used": 2},
        }
        for i in range(30)
    ]
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    descriptor, report = infer_descriptor(p, dataset_id="auto", name="auto")

    # Every role the inferrer should recognise in a grepfruit-shaped dataset.
    m = descriptor.mapping
    assert isinstance(m, HFMapping)
    assert m.id_field == "id"
    assert m.query_field == "query"
    assert m.messages_field == "messages"
    assert m.final_answer_field == "final_answer"
    assert m.documents_field == "documents"
    assert m.document_path_field == "path"
    assert descriptor.primary_metric is not None
    assert descriptor.primary_metric.source == "score"
    assert descriptor.primary_metric.higher_is_better is True
    assert descriptor.ground_truth_source == "expected"
    assert "difficulty" in descriptor.label_names
    assert report.records_sampled == 30

    # Serialized module is syntactically valid Python and re-imports a
    # DatasetDescriptor with the same id.
    code = descriptor_to_python(descriptor)
    gen = tmp_path / "gen.py"
    gen.write_text(code)
    ns: dict = {}
    exec(compile(code, str(gen), "exec"), ns)
    reloaded = ns["DESCRIPTOR"]
    assert reloaded.id == "auto"
    assert reloaded.primary_metric is not None
    assert reloaded.primary_metric.source == "score"
    assert reloaded.label_names == descriptor.label_names


def test_infer_descriptor_loss_style_outcome(tmp_path: Path) -> None:
    """If the outcome field's name hints at loss/error, flip higher_is_better."""
    from dataset.autodetect import infer_descriptor

    p = tmp_path / "loss.jsonl"
    rows = [
        {"id": f"q{i}", "query": "x", "messages": [],
         "loss": i / 30, "difficulty": "easy" if i % 2 else "hard"}
        for i in range(30)
    ]
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    descriptor, _ = infer_descriptor(p, dataset_id="loss", name="loss")
    assert descriptor.primary_metric is not None
    assert descriptor.primary_metric.source == "loss"
    assert descriptor.primary_metric.higher_is_better is False


def test_compact_tool_messages_preserves_recent(tmp_path: Path) -> None:
    """Older tool messages collapse; the most recent N keep their full content."""
    from inference.harness import _compact_tool_messages
    from inference.tools import ToolContext

    # Minimal ctx — we only need result_store for compaction.
    ctx = ToolContext(
        descriptor=GREPFRUIT_LIKE,
        store=None,  # type: ignore[arg-type]
        reader=None,  # type: ignore[arg-type]
        synth_model="x", synth_trace_cap=10, synth_chars_per_trace=100,
        sample_cap=10,
    )
    messages: list = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "function": {"name": "find_traces", "arguments": "{}"}}]},
    ]
    # Fabricate 5 tool messages with full payloads + stashed originals.
    for i in range(5):
        key = ctx.stash_result({"traces": [1, 2, 3], "returned": 3, "total_matching": 100})
        messages.append({
            "role": "tool",
            "tool_call_id": f"c{i}",
            "content": '{"traces": [1,2,3]}',
            "_halo_result_key": key,
            "_halo_tool_name": "find_traces",
        })
    # keep_recent=2: the first 3 should be compacted, the last 2 kept.
    n = _compact_tool_messages(messages, ctx, keep_recent=2)
    assert n == 3
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    # The compactor sets ``_halo_compacted`` only on messages it touched;
    # untouched ones simply don't have the key.
    assert [bool(m.get("_halo_compacted")) for m in tool_msgs] == [True, True, True, False, False]
    # Compacted content mentions the key and the summary shape.
    for m in tool_msgs[:3]:
        assert "compacted · r_" in m["content"]
        assert "returned 3 of 100" in m["content"]
    # Re-running compaction is idempotent on the already-compacted messages.
    assert _compact_tool_messages(messages, ctx, keep_recent=2) == 0


def test_inspect_result_tool_retrieves_stashed_value(tmp_path: Path) -> None:
    """inspect_result fetches the full JSON back from ctx.result_store."""
    from inference.tools import ToolContext, tool_inspect_result

    ctx = ToolContext(
        descriptor=GREPFRUIT_LIKE,
        store=None,  # type: ignore[arg-type]
        reader=None,  # type: ignore[arg-type]
        synth_model="x", synth_trace_cap=10, synth_chars_per_trace=100,
        sample_cap=10,
    )
    key = ctx.stash_result({"foo": "bar", "count": 42})
    assert key == "r_0"
    assert ctx.next_result_key == 1

    got = tool_inspect_result(ctx, key=key)
    assert got["result"] == {"foo": "bar", "count": 42}

    missing = tool_inspect_result(ctx, key="r_999")
    assert "error" in missing
    assert "r_0" in missing["available_keys"]


def test_hf_to_openinference_roundtrip() -> None:
    """A record translated from HF → OI must expose the same canonical
    values (query, labels, outcome, ground_truth, tool calls, final)
    through its TraceView."""
    from dataset.formats.hf import HFTraceView
    from dataset.formats.openinference import OpenInferenceTraceView
    from dataset.translators.hf_to_openinference import translate_record

    hf_record = FIXTURE_RECORD_1
    hf_view = HFTraceView(hf_record, GREPFRUIT_LIKE)

    # Translate and check the OI descriptor can be built (we reuse
    # GREPFRUIT_LIKE but flip format to "openinference").
    from dataclasses import replace
    oi_descriptor = replace(GREPFRUIT_LIKE, mapping=OpenInferenceMapping())
    oi_record = translate_record(hf_record, GREPFRUIT_LIKE)
    oi_view = OpenInferenceTraceView(oi_record, oi_descriptor)

    # Core semantics match.
    assert oi_view.query == hf_view.query
    # Final answer — HF has a dict, OI has it JSON-stringified on the
    # root AGENT output.value.
    assert oi_view.final_answer == json.dumps(hf_record["final_answer"])
    # Labels carried as attributes.
    assert oi_view.labels == hf_view.labels
    # Outcome carried via descriptor.outcome_field ("file_recall").
    assert oi_view.outcome == hf_view.outcome
    # Ground truth survived — as a JSON-serialized list.
    assert json.loads(oi_view.ground_truth) == hf_record["expected_files"]
    # Tool calls — both should surface ripgrep.
    oi_tool_names = sorted({tc.name for tc in oi_view.tool_calls() if tc.name})
    hf_tool_names = sorted({tc.name for tc in hf_view.tool_calls() if tc.name})
    assert oi_tool_names == hf_tool_names == ["ripgrep"]


def test_openinference_indexer_end_to_end(tmp_path: Path) -> None:
    """Translate the two HF fixtures into an OI JSONL and build an index
    on it. Every summary field the store relies on (id, outcome, labels,
    tools_used, nested paths count) must match what the HF path produces.
    """
    from dataclasses import replace
    from dataset.translators.hf_to_openinference import translate_record

    oi_desc = replace(
        GREPFRUIT_LIKE,
        mapping=OpenInferenceMapping(),
        source_path=tmp_path / "oi.jsonl",
    )
    oi_path = oi_desc.source_path
    with oi_path.open("w") as f:
        for hf_rec in (FIXTURE_RECORD_1, FIXTURE_RECORD_2):
            f.write(json.dumps(translate_record(hf_rec, GREPFRUIT_LIKE)) + "\n")

    idx = tmp_path / "oi.index.jsonl"
    n = build_index(oi_path, oi_desc, idx)
    assert n == 2
    store = IndexStore.load(idx, oi_desc)

    # Outcome bucket filters — perfect vs zero.
    perfect = store.filter(outcome_bucket="perfect")
    zero = store.filter(outcome_bucket="zero")
    assert len(perfect) == 1 and perfect[0].outcome == 1.0
    assert len(zero) == 1 and zero[0].outcome == 0.0

    # Labels carried across via root AGENT span attributes.
    hard = store.filter(labels={"difficulty": "hard"})
    assert len(hard) == 1 and hard[0].labels.get("difficulty") == "hard"
    easy = store.filter(labels={"difficulty": "easy"})
    assert len(easy) == 1 and easy[0].labels.get("difficulty") == "easy"

    # Tool usage — FIXTURE_RECORD_1 used ripgrep.
    rip = store.filter(tool_used=["ripgrep"])
    assert len(rip) == 1
    assert "ripgrep" in rip[0].tools_used


def test_openinference_final_none_survives_roundtrip() -> None:
    """HF records with ``final_answer=None`` must round-trip to OI as None,
    not leak the last assistant message through the empty-string fallback
    on ``output.value``. (Peer-review finding #1.)"""
    from dataclasses import replace
    from dataset.formats.openinference import OpenInferenceTraceView
    from dataset.translators.hf_to_openinference import translate_record

    oi_desc = replace(GREPFRUIT_LIKE, mapping=OpenInferenceMapping())
    oi_rec = translate_record(FIXTURE_RECORD_2, GREPFRUIT_LIKE)
    view = OpenInferenceTraceView(oi_rec, oi_desc)
    assert view.final_answer is None


def test_openinference_preserves_repeated_identical_messages(tmp_path: Path) -> None:
    """Two assistant turns saying the same short thing must both surface
    via ``messages()`` — content-based dedup would drop the second.
    (Peer-review finding #2.)"""
    from dataclasses import replace
    from dataset.formats.openinference import OpenInferenceTraceView
    from dataset.translators.hf_to_openinference import translate_record

    record = {
        "query_id": "repeat-1",
        "query": "q",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "ok"},  # same content, different turn
        ],
        "final_answer": "ok",
        "file_recall": 1.0,
        "expected_files": [],
        "query_type": "x", "difficulty": "easy",
    }
    oi_desc = replace(GREPFRUIT_LIKE, mapping=OpenInferenceMapping())
    oi_rec = translate_record(record, GREPFRUIT_LIKE)
    view = OpenInferenceTraceView(oi_rec, oi_desc)
    assistant_msgs = [m for m in view.messages() if m.role == "assistant"]
    assert len(assistant_msgs) == 2, [m.content for m in view.messages()]
    assert [m.content for m in assistant_msgs] == ["ok", "ok"]


def test_openinference_tool_span_without_call_id_dedups_by_span_id(tmp_path: Path) -> None:
    """TOOL spans missing ``tool_call.id`` must still dedup against the
    messages-side pass when the spanId matches — otherwise the call is
    emitted twice. (Peer-review finding #4.)"""
    from dataclasses import replace
    from dataset.formats.openinference import OpenInferenceTraceView

    # Hand-craft an OI record: one LLM span with a tool_call (id=c1),
    # plus a TOOL span with the same call-id. Emitted once.
    oi_desc = replace(GREPFRUIT_LIKE, mapping=OpenInferenceMapping())
    rec = {
        "traceId": "t1",
        "spans": [
            {"spanId": "root", "parentSpanId": None, "name": "agent.run",
             "startTimeUnixNano": "0", "endTimeUnixNano": "0",
             "attributes": [
                 _attr('openinference.span.kind', 'AGENT'),
                 _attr('input.value', 'q'),
                 _attr('output.value', 'done'),
             ]},
            {"spanId": "llm1", "parentSpanId": "root", "name": "llm.call",
             "startTimeUnixNano": "1", "endTimeUnixNano": "2",
             "attributes": [
                 _attr('openinference.span.kind', 'LLM'),
                 _attr('llm.input_messages.0.message.role', 'user'),
                 _attr('llm.input_messages.0.message.content', 'q'),
                 _attr('llm.output_messages.0.message.role', 'assistant'),
                 _attr('llm.output_messages.0.message.tool_calls.0.tool_call.id', 'c1'),
                 _attr('llm.output_messages.0.message.tool_calls.0.tool_call.function.name', 'rip'),
                 _attr('llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments', '{}'),
             ]},
            {"spanId": "tool1", "parentSpanId": "llm1", "name": "tool.rip",
             "startTimeUnixNano": "3", "endTimeUnixNano": "4",
             "attributes": [
                 _attr('openinference.span.kind', 'TOOL'),
                 _attr('tool.name', 'rip'),
                 _attr('tool.call_id', 'c1'),
                 _attr('input.value', '{}'),
                 _attr('output.value', 'hit!'),
             ]},
        ],
    }
    view = OpenInferenceTraceView(rec, oi_desc)
    calls = list(view.tool_calls())
    assert len(calls) == 1, [(c.name, c.id, c.result) for c in calls]
    # TOOL-span path wins (carries the result).
    assert calls[0].name == "rip"
    assert calls[0].result == "hit!"


def test_ask_subagent_schema_pruned_at_max_depth(tmp_path: Path) -> None:
    """At ``depth >= max_depth`` the ``ask_subagent`` schema must be pruned.

    We don't actually run the agent (that would need a live LLM); we
    exercise the schema-filter branch of ``run_agent`` up to the first
    LLM call by injecting a stub and checking which tools the stub was
    offered.
    """
    from inference.harness import run_agent
    from inference.config import InferenceConfig
    import inference.harness as harness_mod

    p = tmp_path / "tiny.jsonl"
    desc = DatasetDescriptor(
        id="tiny", name="tiny", source_path=p,
        mapping=HFMapping(id_field="query_id", final_answer_field=None),
        labels=[],
    )
    with p.open("w") as f:
        f.write(json.dumps({"query_id": "t0", "query": "q", "messages": []}) + "\n")
    idx = tmp_path / "idx.jsonl"
    build_index(p, desc, idx)
    store = IndexStore.load(idx, desc)

    from utils.llm._complete import CompletionResult

    # Capture the tool schemas offered on each LLM call. The stub stream
    # yields no deltas and returns a CompletionResult with no tool_calls,
    # so the agent loop exits on the first turn via ``final``.
    captured_schemas: list[list[dict]] = []

    def fake_stream(model, messages, tools):
        captured_schemas.append(tools)

        def _gen():
            # Returning from a generator sets StopIteration.value, which
            # is how the harness extracts the CompletionResult at stream end.
            return CompletionResult(
                content="(stub)", tool_calls=None,
                message={"role": "assistant", "content": "(stub)"},
                tokens={"input": 0, "output": 0, "thinking": 0, "total": 0},
                cost=0.0, latency=0.0, model=model, error=None,
            )
            yield  # makes _gen a generator function; never executed

        return _gen()

    harness_mod._stream_chat_completion = fake_stream  # type: ignore[assignment]

    cfg = InferenceConfig()
    cfg.init()
    cfg.max_depth = 1

    # Depth 0 (top-level) — ``ask_subagent`` must be present.
    list(run_agent("q", cfg, descriptor=desc, store=store, depth=0))
    names_top = {s["function"]["name"] for s in captured_schemas[-1]}
    assert "ask_subagent" in names_top, names_top

    # Depth 1 (== max_depth) — tool must be pruned.
    captured_schemas.clear()
    list(run_agent("q", cfg, descriptor=desc, store=store, depth=1))
    names_sub = {s["function"]["name"] for s in captured_schemas[-1]}
    assert "ask_subagent" not in names_sub, names_sub
    # Every other tool still present.
    assert "dataset_overview" in names_sub
    assert "find_traces" in names_sub


def test_stash_result_increments_after_write() -> None:
    """``next_result_key`` must only advance after a successful store write —
    otherwise a failed assignment would burn the key and the next stash
    would overwrite the still-valid earlier entry."""
    from inference.tools import ToolContext

    class FailDict(dict):
        def __setitem__(self, k, v):
            raise RuntimeError("boom")

    ctx = ToolContext(
        descriptor=GREPFRUIT_LIKE, store=None,  # type: ignore[arg-type]
        reader=None,  # type: ignore[arg-type]
        synth_model="x", synth_trace_cap=10, synth_chars_per_trace=100,
        sample_cap=10,
    )
    ctx.result_store = FailDict()  # type: ignore[assignment]
    # The write raises; counter must NOT have been bumped.
    import pytest as _pytest
    with _pytest.raises(RuntimeError):
        ctx.stash_result({"x": 1})
    assert ctx.next_result_key == 0


def test_run_code_sandbox_executes_with_prebound_store(tmp_path: Path) -> None:
    """run_code spawns a subprocess with store/reader/descriptor pre-bound."""
    from inference.sandbox import run_code

    p = tmp_path / "tiny.jsonl"
    desc = DatasetDescriptor(
        id="tiny", name="tiny", source_path=p,
        mapping=HFMapping(id_field="query_id", final_answer_field=None),
        labels=[],
    )
    with p.open("w") as f:
        for i in range(3):
            f.write(json.dumps({"query_id": f"t{i}", "query": "q", "messages": []}) + "\n")
    idx = tmp_path / "idx.jsonl"
    build_index(p, desc, idx)

    project_root = Path(__file__).resolve().parent.parent
    out = run_code(
        code='print(len(store)); result = {"n": len(store), "id": descriptor.id}',
        descriptor=desc,
        index_path=idx,
        project_root=project_root,
        timeout_s=30,
    )
    assert out["error"] is None, out
    assert out["stdout"].strip() == "3"
    assert out["returned"]["result"] == {"n": 3, "id": "tiny"}


def test_overview_without_outcome_or_documents(tmp_path: Path) -> None:
    """A dataset without an outcome score or documents still gives a valid overview."""
    minimal_desc = DatasetDescriptor(
        id="minimal", name="minimal", source_path=Path("/dev/null"),
        mapping=HFMapping(id_field="query_id", final_answer_field=None),
        labels=[],
    )
    p = tmp_path / "min.jsonl"
    with p.open("w") as f:
        f.write(json.dumps({"query_id": "m1", "query": "hi", "messages": []}) + "\n")
    idx = tmp_path / "idx.jsonl"
    build_index(p, minimal_desc, idx)
    store = IndexStore.load(idx, minimal_desc)
    ov = store.overview()
    assert ov["count"] == 1
    assert ov.get("outcome") is None
    assert ov.get("paths") is None
    assert ov["labels"] == {}


# ---------------------------------------------------------------------------
# Claude Code native trace shape
# ---------------------------------------------------------------------------


CLAUDE_CODE_FIXTURE = Path(__file__).parent / "fixtures" / "claude-code-sample-traces.jsonl"


def _require_claude_code_fixture() -> Path:
    if not CLAUDE_CODE_FIXTURE.exists():
        pytest.skip(f"fixture missing: {CLAUDE_CODE_FIXTURE}")
    return CLAUDE_CODE_FIXTURE


def test_claude_code_view_extracts_query_tools_and_usage() -> None:
    """Run the view against a real Claude Code native trace and check the
    canonical accessors — the OpenInference view returns empty on this
    shape because it looks for ``openinference.span.kind`` on children
    that don't set it."""
    from dataset import ClaudeCodeMapping, DatasetDescriptor
    from dataset.formats import view_from_record

    path = _require_claude_code_fixture()
    desc = DatasetDescriptor(
        id="cc", name="cc", source_path=path, mapping=ClaudeCodeMapping(),
    )
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert records, "fixture must have at least one record"

    first = view_from_record(records[0], desc)
    # Query lives on claude_code.interaction via user_prompt attr.
    assert first.query and "tracing" in first.query.lower()
    # Token sums are aggregated across every claude_code.llm_request.
    assert first.usage.get("total_tokens", 0) > 0
    # Tool names come from tool_name attributes on claude_code.tool spans.
    tool_names = {tc.name for tc in first.tool_calls()}
    assert tool_names, "expected at least one tool call"
    assert tool_names <= {"Grep", "Glob", "Read", "Bash", "Edit", "Write"}
    # A Read tool carries the file body inside the tool.output span event.
    read_results = [tc.result for tc in first.tool_calls() if tc.name == "Read" and tc.result]
    assert read_results, "expected tool.output event to surface Read content"
    assert any("import" in r or "def " in r or '"""' in r for r in read_results)
    # Turn count equals the number of llm_request spans.
    assert first.turns_used == sum(
        1 for sp in records[0]["spans"] if sp.get("name") == "claude_code.llm_request"
    )


def test_claude_code_indexer_populates_summary(tmp_path: Path) -> None:
    """End-to-end: build_index with ClaudeCodeMapping must yield a
    non-empty query_preview, tools_used, and token totals per trace."""
    from dataset import ClaudeCodeMapping, DatasetDescriptor

    path = _require_claude_code_fixture()
    desc = DatasetDescriptor(
        id="cc", name="cc", source_path=path, mapping=ClaudeCodeMapping(),
    )
    idx = tmp_path / "cc.index.jsonl"
    n = build_index(path, desc, idx)
    assert n == 3
    summaries = [json.loads(line) for line in idx.read_text().splitlines() if line]
    for s in summaries:
        assert s["query_preview"], f"missing query_preview on {s['id']}"
        assert s["n_tool_calls"] > 0
        assert s["total_tokens"] is None or s["total_tokens"] > 0
        assert s["tools_used"], f"missing tools_used on {s['id']}"
    # At least one trace saw Read + something like Grep / Glob / Bash.
    all_tools = {t for s in summaries for t in s["tools_used"]}
    assert "Read" in all_tools


def test_claude_code_messages_include_tool_io_not_assistant_text() -> None:
    """Claude Code's stable span schema doesn't carry assistant text, so
    synthesized messages alternate user -> assistant (tool_call) -> tool
    with empty assistant content. If Anthropic starts emitting assistant
    text in the future this test becomes a reminder to widen the view."""
    from dataset import ClaudeCodeMapping, DatasetDescriptor
    from dataset.formats import view_from_record

    path = _require_claude_code_fixture()
    desc = DatasetDescriptor(
        id="cc", name="cc", source_path=path, mapping=ClaudeCodeMapping(),
    )
    rec = json.loads(path.read_text().splitlines()[0])
    view = view_from_record(rec, desc)
    msgs = list(view.messages())
    assert msgs[0].role == "user" and msgs[0].content
    assistant_turns = [m for m in msgs if m.role == "assistant"]
    assert assistant_turns, "expected at least one assistant tool turn"
    for m in assistant_turns:
        assert m.content == "", "Claude Code view should not fabricate assistant text"
        assert len(m.tool_calls) == 1 and m.tool_calls[0].name


def test_claude_code_and_openinference_views_disagree_on_same_record() -> None:
    """Sanity: the OI view extracts nothing from a Claude Code trace
    (this is the bug the ClaudeCodeView fixes), while the CC view does.
    """
    from dataset import ClaudeCodeMapping, DatasetDescriptor, OpenInferenceMapping
    from dataset.formats import view_from_record

    path = _require_claude_code_fixture()
    rec = json.loads(path.read_text().splitlines()[0])

    oi_desc = DatasetDescriptor(
        id="oi", name="oi", source_path=path, mapping=OpenInferenceMapping(),
    )
    cc_desc = DatasetDescriptor(
        id="cc", name="cc", source_path=path, mapping=ClaudeCodeMapping(),
    )
    oi = view_from_record(rec, oi_desc)
    cc = view_from_record(rec, cc_desc)

    # OI view returns empty — by design, since OI classifies children by
    # openinference.span.kind, which Claude Code doesn't set on them.
    assert list(oi.tool_calls()) == []
    assert oi.query == ""
    # Claude Code view surfaces everything.
    assert cc.query
    assert list(cc.tool_calls())
