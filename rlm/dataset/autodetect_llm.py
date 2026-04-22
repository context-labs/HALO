"""LLM-assisted :class:`DatasetDescriptor` inference.

For rich, non-obvious trace layouts the regex-based heuristic in
:mod:`dataset.autodetect` misses non-standard field names, can't pick
good display names, can't infer multi-metric schemas, and can't suggest
seed questions. This module feeds a sample of records to Opus 4.7 (1M
context) and asks it to design the descriptor end-to-end.

Public API::

    descriptor = infer_descriptor_with_llm(path)

Usage from the CLI::

    uv run halo ingest --smart /path/to/traces.jsonl

The default model is ``claude-opus-4-7[1m]`` because descriptor design
benefits from strong reasoning + a huge context (so we can hand it
dozens of raw records at once instead of tiny summaries).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dataset._fastjson import loads as _fast_loads
from dataset.descriptor import (
    DatasetDescriptor,
    HFMapping,
    Label,
    Metric,
    OpenInferenceMapping,
)

_DEFAULT_MODEL = "claude-opus-4-7"
_DEFAULT_SAMPLE = 40


def _sample_records(path: Path, n: int) -> list[dict[str, Any]]:
    """Read up to ``n`` records from the head of ``path``."""
    out: list[dict[str, Any]] = []
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = _fast_loads(line)
            except Exception:
                continue
            if isinstance(rec, dict):
                out.append(rec)
            if len(out) >= n:
                break
    return out


def _detect_format(records: list[dict[str, Any]]) -> str:
    """Return ``"openinference"`` if records look like OTLP span trees,
    otherwise ``"hf"``."""
    oi_votes = 0
    for r in records[:8]:
        if "spans" in r and isinstance(r.get("spans"), list):
            oi_votes += 1
        if "traceId" in r and "spans" in r:
            oi_votes += 1
    return "openinference" if oi_votes >= 4 else "hf"


def _prompt(records: list[dict[str, Any]], fmt: str, filename: str) -> str:
    """Build the prompt shown to the LLM."""
    preview = json.dumps(records, indent=2, default=str)[:180_000]
    return f"""You are helping design a ``DatasetDescriptor`` for an agent-trace
dataset that halo will analyze. The dataset's on-disk format is
``{fmt}``. Source filename: ``{filename}``.

Here are the first {len(records)} records of the dataset:

```json
{preview}
```

Return a JSON object with the following shape (no prose around it):

```json
{{
  "format": "hf" | "openinference",
  "id": "<url-safe slug based on the filename>",
  "name": "<short human-readable display name>",
  "description": "<1-2 sentence summary of what the agent is doing>",
  "source_model": "<model that produced these traces, if identifiable, else null>",

  "mapping": {{
    // For hf:
    "id_field": "<dotted path to the trace id>",
    "query_field": "<dotted path to the user question>",
    "messages_field": "<dotted path to the conversation list>",
    "final_answer_field": "<dotted path or null>",
    "documents_field": "<dotted path or null>",
    "document_path_field": "<key inside each doc, usually 'path' or null>",
    "usage_field": "<dotted path to token usage dict, e.g. 'metadata.usage'>",
    "turns_field": "<dotted path to turns count>",
    "tool_calls_total_field": "<dotted path>",
    "tool_errors_field": "<dotted path>"

    // For openinference: leave id_attribute null unless a non-standard
    // trace-id attribute is present on the root span
    // "id_attribute": "<attribute key or null>"
  }},

  "primary_metric": null | {{
    "name": "<short metric id, e.g. 'file_recall'>",
    "source": "<field path (hf) or attribute key (oi)>",
    "kind": "score_01" | "binary" | "int" | "seconds" | "dollars",
    "display_name": "<human label shown in UI>",
    "higher_is_better": true | false,
    "perfect_threshold": <float>,
    "zero_threshold": <float>
  }},

  "ground_truth_source": "<field path or attribute key, or null>",

  "labels": [
    {{"name": "<short id>", "source": "<path or attr key>", "display_name": "<optional>"}}
    // ... up to ~6 labels. Pick the low-cardinality categorical fields
    // that an analyst would realistically want to group by. SKIP fields
    // that are unique per-record (ids, timestamps, free text).
  ],

  "seed_questions": [
    // 6-10 natural-language questions an analyst would ask about THIS
    // specific dataset. Favour questions that reveal failure patterns,
    // compare slices, or expose tool-choice issues. Be specific — the
    // analyst should be able to run each one against the tool surface.
  ]
}}
```

Be conservative: if a field isn't obvious from the sample, set it to
null rather than guessing. Prefer ``higher_is_better: false`` when the
metric name contains "loss", "error", "cost", or "latency". For
``score_01`` metrics, use ``perfect_threshold`` close to 1.0 and
``zero_threshold`` close to 0.0 (flip for loss-style metrics).
"""


def _json_from_response(text: str) -> dict[str, Any]:
    """Extract the JSON object from the LLM response. Tolerates prose
    or code fences around it."""
    # Fenced block first.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    # Otherwise grab the widest brace-balanced object.
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("no JSON object in LLM response")
    return json.loads(text[first : last + 1])


def _build_descriptor(spec: dict[str, Any], source_path: Path) -> DatasetDescriptor:
    """Build a :class:`DatasetDescriptor` from the LLM's JSON spec.

    Tolerates missing / malformed fields: the dataset id falls back to
    the source filename stem, individual labels are skipped if they
    lack a ``source``, and a primary_metric entry with no ``source`` is
    dropped rather than raised.
    """
    fmt = spec.get("format", "hf")
    mraw = spec.get("mapping") or {}
    if fmt == "openinference":
        mapping: OpenInferenceMapping | HFMapping = OpenInferenceMapping(
            id_attribute=mraw.get("id_attribute"),
        )
    else:
        mapping = HFMapping(
            id_field=mraw.get("id_field") or "id",
            query_field=mraw.get("query_field") or "query",
            messages_field=mraw.get("messages_field") or "messages",
            final_answer_field=mraw.get("final_answer_field"),
            documents_field=mraw.get("documents_field"),
            document_path_field=mraw.get("document_path_field") or "path",
            usage_field=mraw.get("usage_field") or "metadata.usage",
            turns_field=mraw.get("turns_field") or "metadata.turns_used",
            tool_calls_total_field=mraw.get("tool_calls_total_field")
                or "metadata.total_tool_calls",
            tool_errors_field=mraw.get("tool_errors_field")
                or "metadata.tool_errors",
        )

    pm_raw = spec.get("primary_metric")
    primary_metric: Metric | None = None
    if isinstance(pm_raw, dict) and pm_raw.get("source"):
        primary_metric = Metric(
            name=pm_raw.get("name") or pm_raw["source"].split(".")[-1],
            source=pm_raw["source"],
            kind=pm_raw.get("kind") or "score_01",
            display_name=pm_raw.get("display_name"),
            higher_is_better=bool(pm_raw.get("higher_is_better", True)),
            perfect_threshold=float(pm_raw.get("perfect_threshold", 0.999)),
            zero_threshold=float(pm_raw.get("zero_threshold", 0.001)),
        )

    labels: list[Label] = []
    for lbl in spec.get("labels") or []:
        if not isinstance(lbl, dict):
            continue
        src = lbl.get("source")
        if not src:
            continue
        labels.append(Label(
            name=lbl.get("name") or str(src).split(".")[-1],
            source=str(src),
            display_name=lbl.get("display_name"),
        ))

    return DatasetDescriptor(
        id=spec.get("id") or source_path.stem,
        name=spec.get("name") or source_path.stem,
        source_path=source_path,
        mapping=mapping,
        source_model=spec.get("source_model"),
        description=spec.get("description"),
        primary_metric=primary_metric,
        ground_truth_source=spec.get("ground_truth_source"),
        labels=labels,
        seed_questions=list(spec.get("seed_questions") or []),
    )


def infer_descriptor_with_llm(
    jsonl_path: Path,
    *,
    model: str = _DEFAULT_MODEL,
    sample_size: int = _DEFAULT_SAMPLE,
) -> tuple[DatasetDescriptor, dict[str, Any]]:
    """Design a :class:`DatasetDescriptor` end-to-end via an LLM.

    Returns the constructed descriptor and the raw spec the model
    produced (for logging or manual hand-tuning).
    """
    from utils.llm import complete

    jsonl_path = jsonl_path.resolve()
    records = _sample_records(jsonl_path, sample_size)
    if not records:
        raise ValueError(f"no parseable records in {jsonl_path}")
    fmt = _detect_format(records)

    prompt = _prompt(records, fmt, jsonl_path.name)
    result = complete(
        model,
        [{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    if result.error:
        raise RuntimeError(f"LLM descriptor inference failed: {result.error}")

    spec = _json_from_response(result.content or "")
    descriptor = _build_descriptor(spec, jsonl_path)
    return descriptor, spec
