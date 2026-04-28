from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from engine.traces.models.canonical_span import SpanRecord
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.models.trace_index_models import TraceIndexMeta, TraceIndexRow


def _index_line_offsets(trace_path: Path) -> list[tuple[int, int]]:
    """Stage 1: sequentially scan the JSONL, return (byte_offset, byte_length) for every non-empty line.

    ``byte_length`` includes the trailing newline; workers strip it before parsing.
    Empty lines are filtered out so worker processes never see them.
    """
    offsets: list[tuple[int, int]] = []
    with trace_path.open("rb") as fh:
        position = 0
        for raw_line in fh:
            length = len(raw_line)
            if raw_line.rstrip(b"\n"):
                offsets.append((position, length))
            position += length
    return offsets


def _split_into_chunks(
    line_offsets: list[tuple[int, int]], n_workers: int
) -> list[list[tuple[int, int]]]:
    """Stage 2 prep: split into ``n_workers`` contiguous, order-preserving slices.

    Caps ``n_workers`` at ``len(line_offsets)`` so no empty chunks are dispatched.
    Returns ``[]`` for an empty input.
    """
    if not line_offsets:
        return []
    n = min(n_workers, len(line_offsets))
    base, remainder = divmod(len(line_offsets), n)
    chunks: list[list[tuple[int, int]]] = []
    start = 0
    for i in range(n):
        size = base + (1 if i < remainder else 0)
        chunks.append(line_offsets[start : start + size])
        start += size
    return chunks


def _merge_accumulators(
    per_worker: list[dict[str, _RowAccumulator]],
) -> dict[str, _RowAccumulator]:
    """Stage 3: merge per-worker partials by trace_id; chunk-order traversal preserves file order.

    Iterating ``per_worker`` in order — and ``Pool.map`` returns results in input
    order — is what guarantees ``byte_offsets`` within a trace stays sorted by
    file position. No explicit sort step is needed.
    """
    merged: dict[str, _RowAccumulator] = {}
    for worker_dict in per_worker:
        for trace_id, acc in worker_dict.items():
            existing = merged.get(trace_id)
            if existing is None:
                merged[trace_id] = acc
            else:
                existing.merge_in(acc)
    return merged


def _process_chunk(trace_path: Path, chunk: list[tuple[int, int]]) -> dict[str, _RowAccumulator]:
    """Stage 2 worker: read each (offset, length) from the file, parse, and accumulate locally.

    Top-level so it pickles cleanly for ``multiprocessing.Pool.map``. Each worker
    opens the file independently and seeks to its tuples — the OS page cache makes
    repeated reads of nearby bytes essentially free after the stage-1 scan.
    """
    rows: dict[str, _RowAccumulator] = {}
    with trace_path.open("rb") as fh:
        for byte_offset, byte_length in chunk:
            fh.seek(byte_offset)
            raw = fh.read(byte_length)
            stripped = raw.rstrip(b"\n")
            if not stripped:
                continue
            span = SpanRecord.model_validate_json(stripped)
            acc = rows.setdefault(span.trace_id, _RowAccumulator(trace_id=span.trace_id))
            acc.absorb(span=span, byte_offset=byte_offset, byte_length=len(stripped))
    return rows


# TODO: Switch all dataclasses to pydantic
@dataclass
class _RowAccumulator:
    """Mutable per-trace_id rollup used during a single index-building pass; converts to TraceIndexRow at the end."""

    trace_id: str
    byte_offsets: list[int] = field(default_factory=list)
    byte_lengths: list[int] = field(default_factory=list)
    span_count: int = 0
    start_time: str = ""
    end_time: str = ""
    has_errors: bool = False
    service_names: set[str] = field(default_factory=set)
    model_names: set[str] = field(default_factory=set)
    agent_names: set[str] = field(default_factory=set)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    project_id: str | None = None

    # TODO: Use LLM for smart parsing
    def absorb(self, *, span: SpanRecord, byte_offset: int, byte_length: int) -> None:
        """Fold one span into the accumulator: record its byte slice and update rollup fields."""
        self.byte_offsets.append(byte_offset)
        self.byte_lengths.append(byte_length)
        self.span_count += 1

        if not self.start_time or span.start_time < self.start_time:
            self.start_time = span.start_time
        if not self.end_time or span.end_time > self.end_time:
            self.end_time = span.end_time

        if span.status.code == "STATUS_CODE_ERROR":
            self.has_errors = True

        svc = span.resource.attributes.get("service.name")
        if isinstance(svc, str):
            self.service_names.add(svc)

        model = span.attributes.get("inference.llm.model_name") or span.attributes.get(
            "llm.model_name"
        )
        if isinstance(model, str) and model:
            self.model_names.add(model)

        agent = span.attributes.get("inference.agent_name")
        if isinstance(agent, str) and agent:
            self.agent_names.add(agent)

        input_tokens = span.attributes.get("inference.llm.input_tokens")
        if isinstance(input_tokens, int):
            self.total_input_tokens += input_tokens
        output_tokens = span.attributes.get("inference.llm.output_tokens")
        if isinstance(output_tokens, int):
            self.total_output_tokens += output_tokens

        proj = span.attributes.get("inference.project_id")
        if isinstance(proj, str) and self.project_id is None:
            self.project_id = proj

    def merge_in(self, other: _RowAccumulator) -> None:
        """Fold ``other`` into ``self`` for the same trace_id; caller iterates partials in file order."""
        self.byte_offsets.extend(other.byte_offsets)
        self.byte_lengths.extend(other.byte_lengths)
        self.span_count += other.span_count

        if not self.start_time or (other.start_time and other.start_time < self.start_time):
            self.start_time = other.start_time
        if not self.end_time or other.end_time > self.end_time:
            self.end_time = other.end_time

        if other.has_errors:
            self.has_errors = True

        self.service_names |= other.service_names
        self.model_names |= other.model_names
        self.agent_names |= other.agent_names

        self.total_input_tokens += other.total_input_tokens
        self.total_output_tokens += other.total_output_tokens

        if self.project_id is None:
            self.project_id = other.project_id

    def finalize(self) -> TraceIndexRow:
        """Snapshot the accumulated state into the immutable TraceIndexRow that gets written to the sidecar."""
        return TraceIndexRow(
            trace_id=self.trace_id,
            byte_offsets=self.byte_offsets,
            byte_lengths=self.byte_lengths,
            span_count=self.span_count,
            start_time=self.start_time,
            end_time=self.end_time,
            has_errors=self.has_errors,
            service_names=sorted(self.service_names),
            model_names=sorted(self.model_names),
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            project_id=self.project_id,
            agent_names=sorted(self.agent_names),
        )


class TraceIndexBuilder:
    """Sidecar index creator for the flat OTel JSONL trace input.

    Index is sidecar-style next to the trace file. ``ensure_index_exists`` reuses
    an existing index when its stored stat fingerprint (size + mtime_ns) still
    matches the trace file, and rebuilds it otherwise. Schema-version mismatches
    still fail fast. ``build_index`` is the actual scan + write path.
    """

    SMALL_FILE_THRESHOLD = 1000

    @classmethod
    async def ensure_index_exists(
        cls,
        trace_path: Path,
        config: TraceIndexConfig,
    ) -> Path:
        """Return a usable index path, rebuilding when missing or stale.

        The sidecar is a derived cache: any mismatch — missing files, schema
        version drift, or a different ``source_size``/``source_mtime_ns`` — is
        treated as staleness and triggers a rebuild. ``build_index`` itself
        fails fast on requested versions it does not know how to write.
        """
        index_path = config.index_path or Path(str(trace_path) + ".engine-index.jsonl")
        meta_path = cls._meta_path_for(index_path)

        current_size, current_mtime_ns = cls._fingerprint_trace_file(trace_path)

        if index_path.exists() and meta_path.exists():
            existing = TraceIndexMeta.model_validate_json(meta_path.read_text())
            if (
                existing.schema_version == config.schema_version
                and existing.source_size == current_size
                and existing.source_mtime_ns == current_mtime_ns
            ):
                return index_path

        await cls.build_index(
            trace_path=trace_path,
            index_path=index_path,
            meta_path=meta_path,
            schema_version=config.schema_version,
            source_size=current_size,
            source_mtime_ns=current_mtime_ns,
        )
        return index_path

    @staticmethod
    def _fingerprint_trace_file(trace_path: Path) -> tuple[int, int]:
        """Return ``(size_bytes, mtime_ns)`` for the trace file via a single stat."""
        st = trace_path.stat()
        return st.st_size, st.st_mtime_ns

    @staticmethod
    def _meta_path_for(index_path: Path) -> Path:
        """Convention: ``<trace>.engine-index.jsonl`` ↔ ``<trace>.engine-index.meta.json``."""
        name = index_path.name
        if name.endswith(".engine-index.jsonl"):
            return index_path.with_name(name[: -len(".jsonl")] + ".meta.json")
        return index_path.with_name(name + ".meta.json")

    @classmethod
    async def build_index(
        cls,
        trace_path: Path,
        index_path: Path,
        meta_path: Path,
        schema_version: int,
        source_size: int | None = None,
        source_mtime_ns: int | None = None,
    ) -> None:
        """Two-pass parallel scan over the JSONL, grouping by trace_id and writing the sidecars atomically.

        Stage 1 (sequential) records ``(byte_offset, byte_length)`` per non-empty
        line. Stage 2 splits that list into N=cpu_count worker chunks dispatched
        via ``multiprocessing.Pool.map`` to parallel pydantic parse + accumulate.
        Stage 3 merges per-worker partials by ``trace_id`` in chunk order — which
        equals file order, preserving today's byte-exact output. Below
        ``SMALL_FILE_THRESHOLD`` non-empty lines we run inline (no Pool) to avoid
        fork+pickle overhead dominating on small files.
        """
        if schema_version != 1:
            raise ValueError(f"unsupported trace index schema_version={schema_version}")

        if source_size is None or source_mtime_ns is None:
            source_size, source_mtime_ns = cls._fingerprint_trace_file(trace_path)

        rows = await asyncio.to_thread(cls._run_build, trace_path)

        tmp_index = index_path.with_suffix(index_path.suffix + ".tmp")
        tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")

        tmp_index.write_text(
            "\n".join(row.model_dump_json() for row in rows) + ("\n" if rows else "")
        )
        tmp_meta.write_text(
            TraceIndexMeta(
                schema_version=schema_version,
                trace_count=len(rows),
                source_size=source_size,
                source_mtime_ns=source_mtime_ns,
            ).model_dump_json()
        )

        tmp_index.replace(index_path)
        tmp_meta.replace(meta_path)

    @classmethod
    def _run_build(cls, trace_path: Path) -> list[TraceIndexRow]:
        """Synchronous staged pipeline: index lines, parse in parallel, merge, finalize."""
        line_offsets = _index_line_offsets(trace_path)
        if not line_offsets:
            return []

        if len(line_offsets) < cls.SMALL_FILE_THRESHOLD:
            merged = _process_chunk(trace_path, line_offsets)
        else:
            n_workers = os.cpu_count() or 1
            chunks = _split_into_chunks(line_offsets, n_workers)
            ctx = mp.get_context("forkserver")
            with ctx.Pool(processes=len(chunks)) as pool:
                per_worker = pool.map(partial(_process_chunk, trace_path), chunks)
            merged = _merge_accumulators(per_worker)

        return [acc.finalize() for acc in merged.values()]
