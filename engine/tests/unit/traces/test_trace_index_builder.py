from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.models.trace_index_models import TraceIndexMeta, TraceIndexRow
from engine.traces.trace_index_builder import TraceIndexBuilder


def _write_meta(
    meta_path: Path, *, trace_path: Path, schema_version: int = 1, trace_count: int = 0
) -> None:
    """Pre-seed a meta sidecar with the current stat fingerprint so the reuse path is taken."""
    size, mtime_ns = TraceIndexBuilder._fingerprint_trace_file(trace_path)
    meta = TraceIndexMeta(
        schema_version=schema_version,
        trace_count=trace_count,
        source_size=size,
        source_mtime_ns=mtime_ns,
    )
    meta_path.write_text(meta.model_dump_json())


@pytest.mark.asyncio
async def test_ensure_index_exists_default_path_returned(tmp_path: Path) -> None:
    trace_path = tmp_path / "t.jsonl"
    trace_path.write_text("")
    default_index = Path(str(trace_path) + ".engine-index.jsonl")
    default_meta = Path(str(trace_path) + ".engine-index.meta.json")
    default_index.write_text("")
    _write_meta(default_meta, trace_path=trace_path)

    result_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=TraceIndexConfig(),
    )
    assert result_path == default_index


@pytest.mark.asyncio
async def test_ensure_index_exists_explicit_override(tmp_path: Path) -> None:
    trace_path = tmp_path / "t.jsonl"
    trace_path.write_text("")
    custom_index = tmp_path / "custom.idx.jsonl"
    custom_meta = Path(str(custom_index) + ".meta.json")
    custom_index.write_text("")
    _write_meta(custom_meta, trace_path=trace_path)

    result_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=TraceIndexConfig(index_path=custom_index),
    )
    assert result_path == custom_index


@pytest.mark.asyncio
async def test_build_index_from_tiny_fixture(tmp_path: Path, fixtures_dir: Path) -> None:
    src = fixtures_dir / "tiny_traces.jsonl"
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes(src.read_bytes())

    result_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=TraceIndexConfig(),
    )
    assert result_path.exists()
    meta_path = TraceIndexBuilder._meta_path_for(result_path)
    assert meta_path.exists()

    meta = TraceIndexMeta.model_validate_json(meta_path.read_text())
    assert meta.schema_version == 1
    assert meta.trace_count == 3
    expected_size, expected_mtime_ns = TraceIndexBuilder._fingerprint_trace_file(trace_path)
    assert meta.source_size == expected_size
    assert meta.source_mtime_ns == expected_mtime_ns

    rows = [
        TraceIndexRow.model_validate_json(line) for line in result_path.read_text().splitlines()
    ]
    rows_by_id = {r.trace_id: r for r in rows}
    assert set(rows_by_id) == {"t-aaaa", "t-bbbb", "t-cccc"}

    bb = rows_by_id["t-bbbb"]
    assert bb.has_errors is True
    assert "gpt-5.4" in bb.model_names
    assert bb.total_input_tokens == 200
    assert bb.total_output_tokens == 40

    with trace_path.open("rb") as fh:
        fh.seek(bb.byte_offsets[0])
        blob = fh.read(bb.byte_lengths[0])
    span = json.loads(blob)
    assert span["span_id"] == "s-bbbb-1"


@pytest.mark.asyncio
async def test_ensure_index_rebuilds_on_schema_mismatch(tmp_path: Path, fixtures_dir: Path) -> None:
    src = fixtures_dir / "tiny_traces.jsonl"
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes(src.read_bytes())
    index_path = Path(str(trace_path) + ".engine-index.jsonl")
    meta_path = TraceIndexBuilder._meta_path_for(index_path)
    index_path.write_text("")
    size, mtime_ns = TraceIndexBuilder._fingerprint_trace_file(trace_path)
    stale = TraceIndexMeta(
        schema_version=999,
        trace_count=0,
        source_size=size,
        source_mtime_ns=mtime_ns,
    )
    meta_path.write_text(stale.model_dump_json())

    result_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=TraceIndexConfig(schema_version=1),
    )
    assert result_path == index_path
    rebuilt_meta = TraceIndexMeta.model_validate_json(meta_path.read_text())
    assert rebuilt_meta.schema_version == 1
    assert rebuilt_meta.trace_count == 3


@pytest.mark.asyncio
async def test_ensure_index_rebuilds_when_trace_changes(tmp_path: Path, fixtures_dir: Path) -> None:
    src = fixtures_dir / "tiny_traces.jsonl"
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes(src.read_bytes())

    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=TraceIndexConfig(),
    )
    meta_path = TraceIndexBuilder._meta_path_for(index_path)
    original_meta = TraceIndexMeta.model_validate_json(meta_path.read_text())
    assert original_meta.trace_count == 3

    extra_span = (
        '{"trace_id":"t-dddd","span_id":"s-dddd-1","parent_span_id":"","trace_state":"",'
        '"name":"root","kind":"SPAN_KIND_INTERNAL",'
        '"start_time":"2026-04-23T08:00:00.000000000Z",'
        '"end_time":"2026-04-23T08:00:01.000000000Z",'
        '"status":{"code":"STATUS_CODE_OK","message":""},'
        '"resource":{"attributes":{"service.name":"agent-c"}},'
        '"scope":{"name":"@test/scope","version":"0.0.1"},'
        '"attributes":{"openinference.span.kind":"AGENT","inference.export.schema_version":1,'
        '"inference.project_id":"prj_test","inference.observation_kind":"AGENT",'
        '"inference.agent_name":"agent-c"}}\n'
    )
    with trace_path.open("ab") as fh:
        fh.write(extra_span.encode("utf-8"))

    rebuilt_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=TraceIndexConfig(),
    )
    assert rebuilt_path == index_path
    rebuilt_meta = TraceIndexMeta.model_validate_json(meta_path.read_text())
    assert rebuilt_meta.trace_count == 4
    assert rebuilt_meta.source_size > original_meta.source_size
    expected_size, expected_mtime_ns = TraceIndexBuilder._fingerprint_trace_file(trace_path)
    assert rebuilt_meta.source_size == expected_size
    assert rebuilt_meta.source_mtime_ns == expected_mtime_ns


@pytest.mark.asyncio
async def test_ensure_index_reuses_when_trace_unchanged(tmp_path: Path, fixtures_dir: Path) -> None:
    src = fixtures_dir / "tiny_traces.jsonl"
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes(src.read_bytes())

    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=TraceIndexConfig(),
    )
    meta_path = TraceIndexBuilder._meta_path_for(index_path)
    first_meta_mtime = meta_path.stat().st_mtime_ns
    first_index_mtime = index_path.stat().st_mtime_ns

    await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=TraceIndexConfig(),
    )
    assert meta_path.stat().st_mtime_ns == first_meta_mtime
    assert index_path.stat().st_mtime_ns == first_index_mtime


def test_row_accumulator_merge_in_combines_partial_state() -> None:
    from engine.traces.trace_index_builder import _RowAccumulator

    a = _RowAccumulator(trace_id="t-1")
    a.byte_offsets = [0, 100]
    a.byte_lengths = [80, 120]
    a.span_count = 2
    a.start_time = "2026-04-23T05:00:00.000000000Z"
    a.end_time = "2026-04-23T05:00:01.000000000Z"
    a.has_errors = False
    a.service_names = {"svc-a"}
    a.model_names = {"model-x"}
    a.agent_names = {"agent-a"}
    a.total_input_tokens = 100
    a.total_output_tokens = 50
    a.project_id = "prj_test"

    b = _RowAccumulator(trace_id="t-1")
    b.byte_offsets = [300, 500]
    b.byte_lengths = [60, 90]
    b.span_count = 2
    b.start_time = "2026-04-23T05:00:02.000000000Z"
    b.end_time = "2026-04-23T05:00:03.000000000Z"
    b.has_errors = True
    b.service_names = {"svc-b"}
    b.model_names = {"model-y"}
    b.agent_names = {"agent-b"}
    b.total_input_tokens = 30
    b.total_output_tokens = 20
    b.project_id = None

    a.merge_in(b)

    assert a.byte_offsets == [0, 100, 300, 500]
    assert a.byte_lengths == [80, 120, 60, 90]
    assert a.span_count == 4
    assert a.start_time == "2026-04-23T05:00:00.000000000Z"
    assert a.end_time == "2026-04-23T05:00:03.000000000Z"
    assert a.has_errors is True
    assert a.service_names == {"svc-a", "svc-b"}
    assert a.model_names == {"model-x", "model-y"}
    assert a.agent_names == {"agent-a", "agent-b"}
    assert a.total_input_tokens == 130
    assert a.total_output_tokens == 70
    assert a.project_id == "prj_test"


def test_index_line_offsets_returns_non_empty_line_byte_ranges(tmp_path: Path) -> None:
    from engine.traces.trace_index_builder import _index_line_offsets

    trace_path = tmp_path / "t.jsonl"
    payload = b'{"a":1}\n{"b":2}\n\n{"c":3}\n'
    trace_path.write_bytes(payload)

    offsets = _index_line_offsets(trace_path)

    assert offsets == [(0, 8), (8, 8), (17, 8)]


def test_index_line_offsets_empty_file_returns_empty_list(tmp_path: Path) -> None:
    from engine.traces.trace_index_builder import _index_line_offsets

    trace_path = tmp_path / "t.jsonl"
    trace_path.write_bytes(b"")
    assert _index_line_offsets(trace_path) == []


def test_index_line_offsets_handles_missing_trailing_newline(tmp_path: Path) -> None:
    from engine.traces.trace_index_builder import _index_line_offsets

    trace_path = tmp_path / "t.jsonl"
    trace_path.write_bytes(b'{"a":1}\n{"b":2}')

    offsets = _index_line_offsets(trace_path)
    assert offsets == [(0, 8), (8, 7)]


def test_split_into_chunks_even_division() -> None:
    from engine.traces.trace_index_builder import _split_into_chunks

    items = [(i, 1) for i in range(8)]
    chunks = _split_into_chunks(items, 4)
    assert chunks == [
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)],
        [(4, 1), (5, 1)],
        [(6, 1), (7, 1)],
    ]


def test_split_into_chunks_uneven_division_keeps_order() -> None:
    from engine.traces.trace_index_builder import _split_into_chunks

    items = [(i, 1) for i in range(10)]
    chunks = _split_into_chunks(items, 3)
    flat = [t for c in chunks for t in c]
    assert flat == items
    assert all(len(c) > 0 for c in chunks)
    assert sum(len(c) for c in chunks) == 10


def test_split_into_chunks_caps_n_at_list_length() -> None:
    from engine.traces.trace_index_builder import _split_into_chunks

    items = [(0, 1), (1, 1)]
    chunks = _split_into_chunks(items, 8)
    assert len(chunks) == 2
    assert chunks == [[(0, 1)], [(1, 1)]]


def test_split_into_chunks_empty_input_returns_empty_list() -> None:
    from engine.traces.trace_index_builder import _split_into_chunks

    assert _split_into_chunks([], 4) == []


def test_process_chunk_accumulates_spans_by_trace_id(tmp_path: Path, fixtures_dir: Path) -> None:
    from engine.traces.trace_index_builder import _index_line_offsets, _process_chunk

    src = fixtures_dir / "tiny_traces.jsonl"
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes(src.read_bytes())

    line_offsets = _index_line_offsets(trace_path)
    result = _process_chunk(trace_path, line_offsets)

    assert set(result.keys()) == {"t-aaaa", "t-bbbb", "t-cccc"}
    bb = result["t-bbbb"]
    assert bb.span_count == 2
    assert bb.has_errors is True
    assert "gpt-5.4" in bb.model_names
    assert bb.total_input_tokens == 200
    assert bb.total_output_tokens == 40


def test_process_chunk_partial_chunk_only_sees_its_lines(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    from engine.traces.trace_index_builder import _index_line_offsets, _process_chunk

    src = fixtures_dir / "tiny_traces.jsonl"
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes(src.read_bytes())

    line_offsets = _index_line_offsets(trace_path)
    first_two = _process_chunk(trace_path, line_offsets[:2])

    assert set(first_two.keys()) == {"t-aaaa"}
    assert first_two["t-aaaa"].span_count == 2


def test_merge_accumulators_preserves_chunk_order_for_byte_offsets() -> None:
    from engine.traces.trace_index_builder import _merge_accumulators, _RowAccumulator

    chunk0_a = _RowAccumulator(trace_id="t-1")
    chunk0_a.byte_offsets = [0]
    chunk0_a.byte_lengths = [50]
    chunk0_a.span_count = 1
    chunk0_a.start_time = "2026-04-23T05:00:00.000000000Z"
    chunk0_a.end_time = "2026-04-23T05:00:00.500000000Z"

    chunk1_a = _RowAccumulator(trace_id="t-1")
    chunk1_a.byte_offsets = [200]
    chunk1_a.byte_lengths = [60]
    chunk1_a.span_count = 1
    chunk1_a.start_time = "2026-04-23T05:00:01.000000000Z"
    chunk1_a.end_time = "2026-04-23T05:00:01.500000000Z"

    chunk1_b = _RowAccumulator(trace_id="t-2")
    chunk1_b.byte_offsets = [300]
    chunk1_b.byte_lengths = [40]
    chunk1_b.span_count = 1
    chunk1_b.start_time = "2026-04-23T05:00:02.000000000Z"
    chunk1_b.end_time = "2026-04-23T05:00:02.500000000Z"

    per_worker = [
        {"t-1": chunk0_a},
        {"t-1": chunk1_a, "t-2": chunk1_b},
    ]
    merged = _merge_accumulators(per_worker)

    assert set(merged.keys()) == {"t-1", "t-2"}
    assert merged["t-1"].byte_offsets == [0, 200]
    assert merged["t-1"].span_count == 2
    assert merged["t-2"].byte_offsets == [300]


def test_merge_accumulators_empty_input_returns_empty_dict() -> None:
    from engine.traces.trace_index_builder import _merge_accumulators

    assert _merge_accumulators([]) == {}
    assert _merge_accumulators([{}, {}]) == {}


@pytest.mark.asyncio
async def test_parallel_matches_inline_byte_for_byte(
    tmp_path: Path, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = fixtures_dir / "medium_traces.jsonl"

    parallel_dir = tmp_path / "parallel"
    parallel_dir.mkdir()
    parallel_trace = parallel_dir / "traces.jsonl"
    parallel_trace.write_bytes(src.read_bytes())
    parallel_index = await TraceIndexBuilder.ensure_index_exists(
        trace_path=parallel_trace,
        config=TraceIndexConfig(),
    )

    inline_dir = tmp_path / "inline"
    inline_dir.mkdir()
    inline_trace = inline_dir / "traces.jsonl"
    inline_trace.write_bytes(src.read_bytes())
    monkeypatch.setattr(TraceIndexBuilder, "SMALL_FILE_THRESHOLD", 10_000_000)
    inline_index = await TraceIndexBuilder.ensure_index_exists(
        trace_path=inline_trace,
        config=TraceIndexConfig(),
    )

    assert parallel_index.read_bytes() == inline_index.read_bytes()
