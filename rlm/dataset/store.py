"""In-memory store for the trace summary index.

The store is descriptor-driven: filters work over generic ``labels`` and a
named ``outcome`` score rather than hard-coded grepfruit columns. Overviews
adapt their reported dimensions to whatever the descriptor defines.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import fields
from pathlib import Path
from typing import Any

from dataset._fastjson import loads as _fast_loads
from dataset.descriptor import BucketName, DatasetDescriptor
from dataset.indexer import TraceSummary
from loguru import logger

_TRACE_FIELDS = {f.name for f in fields(TraceSummary)}


def _as_list(value: str | Iterable[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _migrate_legacy_row(raw: dict[str, Any]) -> dict[str, Any]:
    """Bring old (pre-descriptor) index rows up to the current schema.

    Old grepfruit rows carried ``query_id``, ``query_type`` and ``difficulty``
    as top-level keys and ``file_recall`` as the outcome. We map them into
    ``id``, ``labels``, and ``outcome`` so they load cleanly.
    """
    if "id" not in raw and "query_id" in raw:
        raw["id"] = raw["query_id"]
    if "outcome" not in raw and "file_recall" in raw:
        raw["outcome"] = raw["file_recall"]
    if "ground_truth_count" not in raw and "expected_files_count" in raw:
        raw["ground_truth_count"] = raw["expected_files_count"]
    if "labels" not in raw:
        labels = {}
        for legacy in ("query_type", "difficulty"):
            if legacy in raw and raw[legacy] is not None:
                labels[legacy] = str(raw[legacy])
        raw["labels"] = labels
    return raw


class IndexStore:
    """Load and query a summary index produced by ``indexer.build_index``."""

    def __init__(self, rows: list[TraceSummary], descriptor: DatasetDescriptor):
        self.rows = rows
        self.descriptor = descriptor
        # First-write-wins so lookups are deterministic; duplicates are
        # surfaced via ``duplicate_ids`` so the caller can log / display them.
        self._by_id: dict[str, TraceSummary] = {}
        self.duplicate_ids: list[str] = []
        for r in rows:
            if r.id in self._by_id:
                self.duplicate_ids.append(r.id)
                continue
            self._by_id[r.id] = r
        if self.duplicate_ids:
            sample = ", ".join(self.duplicate_ids[:5])
            extra = f" (+{len(self.duplicate_ids) - 5} more)" if len(self.duplicate_ids) > 5 else ""
            logger.warning(
                "dataset {!r}: {} duplicate trace id(s) found in index; first "
                "occurrence wins for lookups (examples: {}{})",
                descriptor.id, len(self.duplicate_ids), sample, extra,
            )

    @classmethod
    def load(cls, index_path: Path, descriptor: DatasetDescriptor) -> IndexStore:
        rows: list[TraceSummary] = []
        # Binary mode + orjson is ~3-5x faster than the stdlib json on the
        # 50K+ summary rows a real dataset produces.
        with open(index_path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = _fast_loads(line)
                raw = _migrate_legacy_row(raw)
                data: dict[str, Any] = {k: raw.get(k) for k in _TRACE_FIELDS}
                data["tools_used"] = raw.get("tools_used") or []
                data["sample_paths"] = raw.get("sample_paths") or []
                data["labels"] = raw.get("labels") or {}
                rows.append(TraceSummary(**data))
        return cls(rows, descriptor)

    def __len__(self) -> int:
        return len(self.rows)

    def lookup(self, trace_id: str) -> TraceSummary | None:
        return self._by_id.get(trace_id)

    def filter(
        self,
        *,
        labels: dict[str, str | Iterable[str]] | None = None,
        tool_used: str | Iterable[str] | None = None,
        min_outcome: float | None = None,
        max_outcome: float | None = None,
        outcome_bucket: BucketName | None = None,
        min_messages: int | None = None,
        max_messages: int | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
        min_tool_errors: int | None = None,
        has_final_answer: bool | None = None,
        min_max_path_length: int | None = None,
        min_nested_paths: int | None = None,
        id_in: Iterable[str] | None = None,
    ) -> list[TraceSummary]:
        # Normalise label filters into a {name: set(values)} shape.
        norm_labels: dict[str, set[str]] | None = None
        if labels:
            norm_labels = {}
            for k, v in labels.items():
                vs = _as_list(v) or []
                norm_labels[k] = set(vs)

        tool = _as_list(tool_used)
        id_set = set(id_in) if id_in is not None else None

        # Bucket filter is direction-aware via the primary-metric
        # predicates; using them avoids inclusive min/max overlap at
        # threshold boundaries.
        bucket_predicate = None
        if outcome_bucket is not None and self.descriptor.primary_metric is not None:
            pm = self.descriptor.primary_metric
            predicates = {
                "perfect": pm.is_perfect,
                "zero": pm.is_zero,
                "partial": pm.is_partial,
            }
            bucket_predicate = predicates.get(outcome_bucket)

        out: list[TraceSummary] = []
        for r in self.rows:
            if norm_labels is not None:
                skip = False
                for k, allowed in norm_labels.items():
                    if not allowed:
                        continue
                    if r.labels.get(k) not in allowed:
                        skip = True
                        break
                if skip:
                    continue
            if tool is not None and not any(t in r.tools_used for t in tool):
                continue
            if min_outcome is not None and (r.outcome is None or r.outcome < min_outcome):
                continue
            if max_outcome is not None and (r.outcome is None or r.outcome > max_outcome):
                continue
            if bucket_predicate is not None and (r.outcome is None or not bucket_predicate(r.outcome)):
                continue
            if min_messages is not None and r.n_messages < min_messages:
                continue
            if max_messages is not None and r.n_messages > max_messages:
                continue
            if min_tokens is not None and (r.total_tokens is None or r.total_tokens < min_tokens):
                continue
            if max_tokens is not None and (r.total_tokens is None or r.total_tokens > max_tokens):
                continue
            if min_tool_errors is not None and r.tool_errors < min_tool_errors:
                continue
            if has_final_answer is not None and r.has_final_answer != has_final_answer:
                continue
            if min_max_path_length is not None and r.max_path_length < min_max_path_length:
                continue
            if min_nested_paths is not None and r.nested_path_count < min_nested_paths:
                continue
            if id_set is not None and r.id not in id_set:
                continue
            out.append(r)
        return out

    def sample(
        self, rows: list[TraceSummary], n: int, seed: int | None = None
    ) -> list[TraceSummary]:
        if len(rows) <= n:
            return list(rows)
        rng = random.Random(seed)
        return rng.sample(rows, n)

    def overview(self, rows: list[TraceSummary] | None = None) -> dict[str, Any]:
        rs = rows if rows is not None else self.rows
        n = len(rs)
        if n == 0:
            return {"count": 0}

        # Label distributions (descriptor-driven).
        label_dists: dict[str, dict[str, int]] = {}
        for name in self.descriptor.label_names:
            c: Counter[str] = Counter()
            for r in rs:
                v = r.labels.get(name)
                if v is not None:
                    c[v] += 1
            label_dists[name] = dict(c.most_common())

        tool_names: Counter[str] = Counter()
        for r in rs:
            tool_names.update(r.tools_used)

        outcomes = [r.outcome for r in rs if r.outcome is not None]
        msgs = [r.n_messages for r in rs]
        tokens = [r.total_tokens for r in rs if r.total_tokens is not None]
        turns = [r.turns_used for r in rs if r.turns_used is not None]
        tool_errs = sum(r.tool_errors for r in rs)
        max_paths = [r.max_path_length for r in rs]
        nested_counts = [r.nested_path_count for r in rs]

        def _hist(values: list[float | int], buckets: list[tuple[float, float]]) -> dict[str, int]:
            out: dict[str, int] = {}
            for lo, hi in buckets:
                label = f"[{lo:g},{hi:g})" if hi != float("inf") else f">={lo:g}"
                out[label] = sum(1 for v in values if lo <= v < hi)
            return out

        outcome_summary: dict[str, Any] | None = None
        if self.descriptor.primary_metric is not None and outcomes:
            pm = self.descriptor.primary_metric
            # Bucket via the direction-aware predicates. This is the same
            # source of truth used by sample_by_outcome and the UI filter
            # so boundaries never disagree.
            perfect = sum(1 for v in outcomes if pm.is_perfect(v))
            zero = sum(1 for v in outcomes if pm.is_zero(v))
            # Partial is the exhaustive complement — computed via subtraction
            # so the three counts always sum to len(outcomes).
            partial = len(outcomes) - perfect - zero
            outcome_summary = {
                "display_name": pm.label,
                "higher_is_better": pm.higher_is_better,
                "mean": round(sum(outcomes) / len(outcomes), 4),
                "perfect_count": perfect,
                "zero_count": zero,
                "partial_count": partial,
                "buckets": _hist(
                    outcomes,
                    [
                        (0.0, 0.001),
                        (0.001, 0.25),
                        (0.25, 0.5),
                        (0.5, 0.75),
                        (0.75, 0.999),
                        (0.999, float("inf")),
                    ],
                ),
            }

        result: dict[str, Any] = {
            "count": n,
            "labels": label_dists,
            "tools_used_totals": dict(tool_names.most_common()),
            "messages": {
                "mean": round(sum(msgs) / n, 2),
                "min": min(msgs),
                "max": max(msgs),
                "p50": _percentile(msgs, 50),
                "p90": _percentile(msgs, 90),
                "p99": _percentile(msgs, 99),
            },
            "tokens": (
                {
                    "mean": round(sum(tokens) / len(tokens), 2),
                    "min": min(tokens),
                    "max": max(tokens),
                    "p50": _percentile(tokens, 50),
                    "p90": _percentile(tokens, 90),
                    "p99": _percentile(tokens, 99),
                }
                if tokens
                else None
            ),
            "turns_used": (
                {
                    "mean": round(sum(turns) / len(turns), 2),
                    "min": min(turns),
                    "max": max(turns),
                    "p50": _percentile(turns, 50),
                    "p90": _percentile(turns, 90),
                }
                if turns
                else None
            ),
            "outcome": outcome_summary,
            "tool_errors_total": tool_errs,
        }

        if self.descriptor.has_documents:
            result["paths"] = {
                "max_path_length": {
                    "mean": round(sum(max_paths) / n, 2),
                    "p50": _percentile(max_paths, 50),
                    "p90": _percentile(max_paths, 90),
                    "p99": _percentile(max_paths, 99),
                    "max": max(max_paths) if max_paths else 0,
                },
                "nested_per_trace": {
                    "mean": round(sum(nested_counts) / n, 2),
                    "p50": _percentile(nested_counts, 50),
                    "p90": _percentile(nested_counts, 90),
                    "max": max(nested_counts) if nested_counts else 0,
                },
                "nested_share": round(sum(1 for c in nested_counts if c > 0) / n, 3),
            }

        return result


def _percentile(values: list[float | int], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return round(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac, 2)
