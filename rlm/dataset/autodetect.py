"""Heuristic inference of a ``DatasetDescriptor`` from a trace JSONL.

Given a path to a JSONL, read a small sample of records and guess which
fields map to the canonical roles (id, query, messages, outcome, ground
truth, documents, categorical labels). The inference is intentionally
conservative — we pick a mapping only if we're confident; the caller is
free to hand-edit the generated descriptor afterwards.

Users invoke this via ``halo ingest <path>``; no one writes catalog
descriptors by hand unless they want to.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataset._fastjson import loads as _loads
from dataset.descriptor import (
    ClaudeCodeMapping,
    DatasetDescriptor,
    HFMapping,
    Label,
    Metric,
    OpenInferenceMapping,
)


def _sniff_otlp_format(records: list[dict[str, Any]]) -> str | None:
    """Peek at sampled records and return ``"claude_code"`` /
    ``"openinference"`` when the data looks like an OTLP span-tree JSONL,
    or ``None`` when it's not OTLP (HF-style flat records).

    The decision is cheap: any span whose ``name`` starts with
    ``claude_code.`` wins the Claude Code path (stable signature of the
    Claude Code CLI's native telemetry). If the records look like OTLP
    span trees (``spans`` array present) but no ``claude_code.*`` span
    is seen, fall back to OpenInference.
    """
    otlp_votes = 0
    for r in records[:8]:
        spans = r.get("spans") if isinstance(r, dict) else None
        if not isinstance(spans, list):
            continue
        otlp_votes += 1
        for span in spans:
            name = str((span or {}).get("name") or "")
            if name.startswith("claude_code."):
                return "claude_code"
    return "openinference" if otlp_votes >= 1 else None


# Name hints (ordered by preference). We check if the hint path exists on
# a majority of sampled records before accepting it.
_ID_HINTS = ["query_id", "trace_id", "task_id", "id", "uid", "uuid"]
_QUERY_HINTS = ["query", "question", "prompt", "input", "task"]
_MESSAGES_HINTS = ["messages", "conversation", "history", "turns"]
_FINAL_HINTS = ["final_answer", "answer", "output", "response", "completion", "result"]
_OUTCOME_HINTS = [
    "file_recall", "recall", "score", "accuracy", "success",
    "correct", "reward", "f1", "pass", "precision",
    "loss", "error_rate",
]
_GROUND_TRUTH_HINTS = [
    "expected_files", "expected", "ground_truth", "target", "gold",
    "answer_key", "reference",
]
_DOCUMENTS_HINTS = ["documents", "files", "docs", "corpus", "context"]
_DOC_PATH_KEYS = ["path", "file", "filename", "name"]
_LABEL_HINT_WORDS = [
    "type", "category", "difficulty", "class", "kind",
    "tag", "genre", "domain", "bucket", "variant",
]
_LOWER_BETTER_WORDS = ["loss", "error", "cost", "latency", "distance"]
_USAGE_HINTS = ["metadata.usage", "usage"]
_TURNS_HINTS = ["metadata.turns_used", "metadata.turns", "turns_used", "turns"]
_TOOLCALLS_HINTS = [
    "metadata.total_tool_calls", "metadata.tool_calls_total",
    "total_tool_calls", "tool_calls_total",
]
_TOOL_ERRORS_HINTS = ["metadata.tool_errors", "tool_errors"]

_MIN_SAMPLE = 20
_DEFAULT_SAMPLE = 200
_LABEL_CARDINALITY_MAX = 50


@dataclass
class InferenceReport:
    """What the inferrer decided, surfaced for CLI output + logging."""

    records_sampled: int
    fields: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _has_path(obj: Any, path: str) -> bool:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _get_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _first_present(records: list[dict[str, Any]], hints: list[str]) -> str | None:
    """Return the first hint whose path exists in ≥50% of records."""
    if not records:
        return None
    threshold = max(1, len(records) // 2)
    for hint in hints:
        present = sum(1 for r in records if _has_path(r, hint))
        if present >= threshold:
            return hint
    return None


def _sample_records(path: Path, n: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "rb") as f:
        for line in f:
            if len(records) >= n:
                break
            try:
                r = _loads(line)
            except Exception:
                continue
            if isinstance(r, dict):
                records.append(r)
    return records


def _looks_like_messages(v: Any) -> bool:
    return (
        isinstance(v, list)
        and bool(v)
        and isinstance(v[0], dict)
        and "role" in v[0]
    )


def _detect_messages_field(records: list[dict[str, Any]]) -> str | None:
    # Prefer the hinted names, but confirm by shape.
    for hint in _MESSAGES_HINTS:
        sample = [r.get(hint) for r in records if hint in r]
        if sample and any(_looks_like_messages(v) for v in sample):
            return hint
    # Fallback: any top-level list field whose first element has role+content.
    for r in records[:10]:
        for k, v in r.items():
            if _looks_like_messages(v):
                return k
    return None


def _looks_like_docs(v: Any) -> str | None:
    """Return the path-key name if ``v`` looks like a docs list; else None."""
    if not (isinstance(v, list) and v and isinstance(v[0], dict)):
        return None
    for key in _DOC_PATH_KEYS:
        if key in v[0]:
            return key
    return None


def _detect_documents(records: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    for hint in _DOCUMENTS_HINTS:
        sample = [r.get(hint) for r in records if hint in r]
        if not sample:
            continue
        path_keys = [_looks_like_docs(v) for v in sample if v is not None]
        if path_keys and any(pk for pk in path_keys):
            # Majority-vote the path key name.
            counts = Counter([pk for pk in path_keys if pk])
            return hint, counts.most_common(1)[0][0]
    return None, None


def _detect_outcome(records: list[dict[str, Any]]) -> str | None:
    # Try hinted names first (top-level and nested metadata.*).
    candidate_paths = list(_OUTCOME_HINTS) + [f"metadata.{h}" for h in _OUTCOME_HINTS]
    for path in candidate_paths:
        vals = [_get_path(r, path) for r in records if _has_path(r, path)]
        nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(nums) < len(records) * 0.5:
            continue
        if all(0 <= v <= 1 for v in nums):
            return path
    return None


def _detect_ground_truth(records: list[dict[str, Any]]) -> str | None:
    for hint in _GROUND_TRUTH_HINTS:
        hits = [r.get(hint) for r in records if hint in r]
        if len(hits) >= len(records) * 0.5 and any(h is not None for h in hits):
            return hint
    return None


def _detect_label_fields(
    records: list[dict[str, Any]],
    reserved: set[str],
) -> list[str]:
    """Top-level scalar string fields with low cardinality."""
    card: dict[str, Counter[str]] = {}
    for r in records:
        for k, v in r.items():
            if k in reserved:
                continue
            if not isinstance(v, str):
                continue
            card.setdefault(k, Counter())[v] += 1
    # Discard fields that appear like free-text (no repeats across sample).
    filtered: list[tuple[str, int, int]] = []
    for k, c in card.items():
        if len(c) > _LABEL_CARDINALITY_MAX:
            continue
        if len(c) <= 1:
            continue  # singleton — not informative
        # Prefer names whose last segment contains a known hint.
        hinted = any(p in k.lower() for p in _LABEL_HINT_WORDS)
        filtered.append((k, len(c), 1 if hinted else 0))
    filtered.sort(key=lambda t: (-t[2], t[1]))
    return [k for k, _, _ in filtered[:6]]


def _slug_from_path(path: Path) -> str:
    stem = path.stem
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return slug or "dataset"


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def infer_descriptor(
    jsonl_path: Path,
    *,
    dataset_id: str | None = None,
    name: str | None = None,
    source_model: str | None = None,
    description: str | None = None,
    sample_size: int = _DEFAULT_SAMPLE,
) -> tuple[DatasetDescriptor, InferenceReport]:
    """Sniff ``jsonl_path`` and return a best-guess descriptor + a report.

    The report enumerates every field mapping the inferrer picked and any
    notes/warnings. The descriptor is ready to hand to ``build_index``.
    """
    jsonl_path = jsonl_path.resolve()
    records = _sample_records(jsonl_path, max(sample_size, _MIN_SAMPLE))
    report = InferenceReport(records_sampled=len(records))
    if len(records) < _MIN_SAMPLE:
        report.notes.append(
            f"only {len(records)} records available (wanted {_MIN_SAMPLE}+); "
            f"inferences may be unreliable"
        )
    if not records:
        raise ValueError(f"No parseable JSON records in {jsonl_path}")

    did = dataset_id or _slug_from_path(jsonl_path)
    disp_name = name or jsonl_path.stem

    # OTLP span-tree fast path: skip the HF-style field detection entirely.
    # OpenInference / Claude Code span-tree files have canonical attribute
    # names, so the mapping itself is the whole descriptor.
    otlp_fmt = _sniff_otlp_format(records)
    if otlp_fmt is not None:
        mapping: OpenInferenceMapping | ClaudeCodeMapping = (
            ClaudeCodeMapping() if otlp_fmt == "claude_code"
            else OpenInferenceMapping()
        )
        descriptor = DatasetDescriptor(
            id=did,
            name=disp_name,
            source_path=jsonl_path,
            mapping=mapping,
            source_model=source_model,
            description=description,
            primary_metric=None,
            ground_truth_source=None,
            labels=[],
        )
        report.fields = {"format": otlp_fmt, "mapping": type(mapping).__name__}
        report.notes.append(f"detected OTLP span-tree format: {otlp_fmt}")
        return descriptor, report

    id_field = _first_present(records, _ID_HINTS) or "id"
    query_field = _first_present(records, _QUERY_HINTS) or "query"
    messages_field = _detect_messages_field(records)
    final_answer_field = _first_present(records, _FINAL_HINTS)
    outcome_field = _detect_outcome(records)
    ground_truth_field = _detect_ground_truth(records)
    documents_field, document_path_field = _detect_documents(records)
    usage_field = _first_present(records, _USAGE_HINTS) or "metadata.usage"
    turns_field = _first_present(records, _TURNS_HINTS) or "metadata.turns_used"
    tool_calls_field = _first_present(records, _TOOLCALLS_HINTS) or "metadata.total_tool_calls"
    tool_errors_field = _first_present(records, _TOOL_ERRORS_HINTS) or "metadata.tool_errors"

    reserved: set[str] = set()
    for f in (
        id_field, query_field, messages_field, final_answer_field,
        outcome_field, ground_truth_field, documents_field,
    ):
        if f:
            reserved.add(f.split(".")[0])
    reserved.add("metadata")

    label_fields = _detect_label_fields(records, reserved)

    primary_metric: Metric | None = None
    if outcome_field:
        lower_better = any(p in outcome_field.lower() for p in _LOWER_BETTER_WORDS)
        display = outcome_field.split(".")[-1]
        primary_metric = Metric(
            name=display,
            source=outcome_field,
            kind="score_01",
            display_name=display,
            higher_is_better=not lower_better,
            perfect_threshold=0.999 if not lower_better else 0.001,
            zero_threshold=0.001 if not lower_better else 0.999,
        )
    else:
        report.notes.append("no outcome field detected — sample_by_outcome will be disabled")

    if messages_field is None:
        messages_field = "messages"
        report.notes.append("no messages field detected — defaulting to 'messages' (may be empty)")

    descriptor = DatasetDescriptor(
        id=did,
        name=disp_name,
        source_path=jsonl_path,
        mapping=HFMapping(
            id_field=id_field,
            query_field=query_field,
            messages_field=messages_field,
            final_answer_field=final_answer_field,
            documents_field=documents_field,
            document_path_field=document_path_field,
            usage_field=usage_field,
            turns_field=turns_field,
            tool_calls_total_field=tool_calls_field,
            tool_errors_field=tool_errors_field,
        ),
        source_model=source_model,
        description=description,
        primary_metric=primary_metric,
        ground_truth_source=ground_truth_field,
        labels=[Label(name=f.split(".")[-1], source=f) for f in label_fields],
    )

    report.fields = {
        "id_field": id_field,
        "query_field": query_field,
        "messages_field": messages_field,
        "final_answer_field": final_answer_field,
        "outcome_field": outcome_field,
        "ground_truth_field": ground_truth_field,
        "documents_field": documents_field,
        "document_path_field": document_path_field,
        "usage_field": usage_field,
        "turns_field": turns_field,
        "tool_calls_total_field": tool_calls_field,
        "tool_errors_field": tool_errors_field,
        "label_fields": label_fields,
        "primary_metric_higher_is_better": primary_metric.higher_is_better if primary_metric else None,
    }
    return descriptor, report


def descriptor_to_python(descriptor: DatasetDescriptor) -> str:
    """Serialise a :class:`DatasetDescriptor` as a Python module suitable
    for ``catalog/``. Emits either the HF or OpenInference shape based
    on the descriptor's ``mapping`` type."""
    d = descriptor
    seed_list = "[" + ", ".join(repr(x) for x in d.seed_questions) + "]"
    labels_list = "[\n        " + ",\n        ".join(
        f"Label(name={lbl.name!r}, source={lbl.source!r})"
        for lbl in d.labels
    ) + (",\n    ]" if d.labels else "]")

    pm = d.primary_metric
    if pm is not None:
        metric_block = (
            f"    primary_metric=Metric(\n"
            f"        name={pm.name!r},\n"
            f"        source={pm.source!r},\n"
            f"        kind={pm.kind!r},\n"
            f"        display_name={pm.display_name!r},\n"
            f"        higher_is_better={pm.higher_is_better!r},\n"
            f"        perfect_threshold={pm.perfect_threshold!r},\n"
            f"        zero_threshold={pm.zero_threshold!r},\n"
            f"    ),\n"
        )
    else:
        metric_block = "    primary_metric=None,\n"

    if isinstance(d.mapping, HFMapping):
        m = d.mapping
        mapping_block = (
            f"    mapping=HFMapping(\n"
            f"        id_field={m.id_field!r},\n"
            f"        query_field={m.query_field!r},\n"
            f"        messages_field={m.messages_field!r},\n"
            f"        final_answer_field={m.final_answer_field!r},\n"
            f"        documents_field={m.documents_field!r},\n"
            f"        document_path_field={m.document_path_field!r},\n"
            f"        usage_field={m.usage_field!r},\n"
            f"        turns_field={m.turns_field!r},\n"
            f"        tool_calls_total_field={m.tool_calls_total_field!r},\n"
            f"        tool_errors_field={m.tool_errors_field!r},\n"
            f"    ),\n"
        )
        imports = "from dataset import DatasetDescriptor, HFMapping, Label, Metric"
    elif isinstance(d.mapping, ClaudeCodeMapping):
        m = d.mapping
        mapping_block = (
            f"    mapping=ClaudeCodeMapping(\n"
            f"        id_attribute={m.id_attribute!r},\n"
            f"    ),\n"
        )
        imports = "from dataset import ClaudeCodeMapping, DatasetDescriptor, Label, Metric"
    else:
        assert isinstance(d.mapping, OpenInferenceMapping)
        m = d.mapping
        mapping_block = (
            f"    mapping=OpenInferenceMapping(\n"
            f"        id_attribute={m.id_attribute!r},\n"
            f"    ),\n"
        )
        imports = "from dataset import DatasetDescriptor, Label, Metric, OpenInferenceMapping"

    return f'''"""{d.name} — auto-generated by ``halo ingest``.

Hand-edit field mappings, labels, or the primary metric to refine the
inference. Re-run ``uv run halo index --dataset {d.id}`` after changes.
Seed questions populate the UI preset list.
"""

from __future__ import annotations

from pathlib import Path

{imports}

DESCRIPTOR = DatasetDescriptor(
    id={d.id!r},
    name={d.name!r},
    source_path=Path({str(d.source_path)!r}),
{mapping_block}    source_model={d.source_model!r},
    description={d.description!r},
{metric_block}    ground_truth_source={d.ground_truth_source!r},
    labels={labels_list},
    seed_questions={seed_list},
)
'''
