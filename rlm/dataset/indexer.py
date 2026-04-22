"""Build a compact summary index over a trace JSONL, driven by a ``DatasetDescriptor``.

The source JSONL can be hundreds of GB. We scan it once, extract lightweight
per-trace metadata (labels, outcome, message counts, path stats, etc.) along
with the byte-offset and length so the full record can be fetched on demand
via ``TraceReader``.

The summary index is itself a small JSONL (tens of MB for ~1M traces) that
fits entirely in RAM.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataset._fastjson import dumps as _fast_dumps
from dataset._fastjson import loads as _fast_loads
from dataset.descriptor import DatasetDescriptor
from dataset.formats import view_from_record


@dataclass
class TraceSummary:
    """Per-trace metadata. Shape is the same for every dataset; fields that
    don't apply to a given dataset get zero/empty defaults."""

    id: str
    byte_offset: int
    byte_length: int
    query_preview: str

    # Descriptor-driven dims
    labels: dict[str, str] = field(default_factory=dict)
    outcome: float | None = None
    ground_truth_count: int = 0

    # Universal
    has_final_answer: bool = False
    final_answer_chars: int = 0
    n_messages: int = 0
    n_tool_calls: int = 0
    tool_errors: int = 0
    turns_used: int | None = None
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tools_used: list[str] = field(default_factory=list)

    # Only meaningful when descriptor.has_documents
    n_docs: int = 0
    max_path_length: int = 0
    avg_path_length: float = 0.0
    nested_path_count: int = 0
    sample_paths: list[str] = field(default_factory=list)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_summary(
    record: dict[str, Any], desc: DatasetDescriptor, offset: int, length: int
) -> TraceSummary:
    """Extract a TraceSummary for one record via a format-appropriate TraceView."""
    view = view_from_record(record, desc)

    # Walk messages once for n_messages + tools_used.
    messages = list(view.messages())
    tools_used: list[str] = []
    for m in messages:
        for tc in m.tool_calls:
            if tc.name and tc.name not in tools_used:
                tools_used.append(tc.name)

    # Path stats.
    docs = view.documents()
    n_docs = len(docs)
    path_lens = [len(d.path) for d in docs if d.path]
    nested = sum(1 for d in docs if "/" in d.path)
    sample_paths = [d.path for d in docs[:3]]
    max_path_length = max(path_lens) if path_lens else 0
    avg_path_length = round(sum(path_lens) / len(path_lens), 1) if path_lens else 0.0

    # Ground truth count (length if list, 1 if scalar, 0 otherwise).
    gt = view.ground_truth
    if isinstance(gt, list):
        ground_truth_count = len(gt)
    elif gt is None:
        ground_truth_count = 0
    else:
        ground_truth_count = 1

    # Final answer size.
    final = view.final_answer
    has_final_answer = False
    final_answer_chars = 0
    if isinstance(final, dict):
        has_final_answer = True
        ans = final.get("answer") or ""
        final_answer_chars = len(ans) if isinstance(ans, str) else len(json.dumps(ans))
    elif isinstance(final, str) and final:
        has_final_answer = True
        final_answer_chars = len(final)

    usage = view.usage

    return TraceSummary(
        id=view.id,
        byte_offset=offset,
        byte_length=length,
        query_preview=view.query[:240],
        labels=view.labels,
        outcome=view.outcome,
        ground_truth_count=ground_truth_count,
        has_final_answer=has_final_answer,
        final_answer_chars=final_answer_chars,
        n_messages=len(messages),
        n_tool_calls=view.tool_calls_total,
        tool_errors=view.tool_errors,
        turns_used=view.turns_used,
        total_tokens=usage.get("total_tokens"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        tools_used=tools_used,
        n_docs=n_docs,
        max_path_length=max_path_length,
        avg_path_length=avg_path_length,
        nested_path_count=nested,
        sample_paths=sample_paths,
    )


def scan_dataset(
    jsonl_path: Path,
    descriptor: DatasetDescriptor,
    max_records: int | None = None,
) -> Iterator[TraceSummary]:
    """Stream a trace JSONL from disk, yielding one TraceSummary per record."""
    offset = 0
    count = 0
    with open(jsonl_path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                return
            length = len(line)
            line_offset = offset
            offset += length
            try:
                record = _fast_loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            yield _extract_summary(record, descriptor, line_offset, length)
            count += 1
            if max_records is not None and count >= max_records:
                return


def build_index(
    jsonl_path: Path,
    descriptor: DatasetDescriptor,
    index_path: Path,
    *,
    max_records: int | None = None,
    progress_every: int = 2000,
    workers: int = 0,
) -> int:
    """Build a summary index by streaming the source JSONL once.

    Returns the number of indexed records.
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if workers and workers > 1:
        return _build_index_parallel(
            jsonl_path, descriptor, index_path,
            workers=workers, max_records=max_records, progress_every=progress_every,
        )

    count = 0
    with open(index_path, "wb") as out:
        for summary in scan_dataset(jsonl_path, descriptor, max_records=max_records):
            # ``vars(summary)`` is a direct reference to the dataclass dict;
            # ``dataclasses.asdict`` does a recursive deep-copy that we don't
            # need here (we immediately serialize with orjson). This is
            # measurably cheaper per-row on million-row indexes.
            out.write(_fast_dumps(vars(summary)))
            out.write(b"\n")
            count += 1
            if progress_every and count % progress_every == 0:
                print(f"  indexed {count:,} traces...", flush=True)
    return count


# ---------------------------------------------------------------------------
# Parallel indexer (producer reads lines, workers parse + extract).
# ---------------------------------------------------------------------------


_WORKER_DESCRIPTOR: DatasetDescriptor | None = None


def _worker_init(desc: DatasetDescriptor) -> None:
    global _WORKER_DESCRIPTOR
    _WORKER_DESCRIPTOR = desc


def _parse_one(task: tuple[int, int, bytes]) -> dict[str, Any] | None:
    offset, length, line = task
    try:
        record = _fast_loads(line)
    except Exception:
        return None
    if not isinstance(record, dict) or _WORKER_DESCRIPTOR is None:
        return None
    # vars(dataclass) returns ``__dict__`` which is already a plain dict,
    # same shape as asdict() for flat dataclasses, and ~5× cheaper.
    return vars(_extract_summary(record, _WORKER_DESCRIPTOR, offset, length))


def _line_producer(
    path: Path, max_records: int | None
) -> Iterator[tuple[int, int, bytes]]:
    offset = 0
    count = 0
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                return
            yield offset, len(line), line
            offset += len(line)
            count += 1
            if max_records is not None and count >= max_records:
                return


def _build_index_parallel(
    jsonl_path: Path,
    descriptor: DatasetDescriptor,
    index_path: Path,
    *,
    workers: int,
    max_records: int | None,
    progress_every: int,
) -> int:
    from multiprocessing import get_context

    ctx = get_context("fork")
    count = 0
    with ctx.Pool(workers, initializer=_worker_init, initargs=(descriptor,)) as pool, \
            open(index_path, "wb") as out:
        for summary_dict in pool.imap(
            _parse_one,
            _line_producer(jsonl_path, max_records),
            chunksize=64,
        ):
            if summary_dict is None:
                continue
            out.write(_fast_dumps(summary_dict))
            out.write(b"\n")
            count += 1
            if progress_every and count % progress_every == 0:
                print(f"  indexed {count:,} traces...", flush=True)
    return count
