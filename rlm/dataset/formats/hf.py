"""TraceView over the legacy Catalyst flat-JSONL shape.

Reads via the dotted paths in :class:`dataset.descriptor.HFMapping`.
Source references on :class:`Metric` / :class:`Label` / the descriptor's
``ground_truth_source`` are also treated as dotted field paths here.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from dataset.descriptor import (
    DatasetDescriptor,
    HFMapping,
    get_nested,
)
from dataset.formats.trace_view import (
    DocumentView,
    MessageView,
    ToolCallView,
    TraceView,
)


def _coerce_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _coerce_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class HFTraceView(TraceView):
    def __init__(self, record: dict, descriptor: DatasetDescriptor):
        assert isinstance(descriptor.mapping, HFMapping), (
            "HFTraceView requires a descriptor with HFMapping"
        )
        self._rec = record
        self._desc = descriptor
        self._m: HFMapping = descriptor.mapping

    # --- identity / semantics ---
    @property
    def id(self) -> str:
        return str(get_nested(self._rec, self._m.id_field) or "?")

    @property
    def query(self) -> str:
        q = get_nested(self._rec, self._m.query_field) or ""
        if not isinstance(q, str):
            q = json.dumps(q)
        return q

    @property
    def final_answer(self) -> Any | None:
        if not self._m.final_answer_field:
            return None
        return get_nested(self._rec, self._m.final_answer_field)

    @property
    def ground_truth(self) -> Any | None:
        if not self._desc.has_ground_truth:
            return None
        return get_nested(self._rec, self._desc.ground_truth_source)

    def metric_value(self, name: str) -> float | None:
        metric = self._desc.metric(name)
        if metric is None:
            return None
        return _coerce_float(get_nested(self._rec, metric.source))

    @property
    def outcome(self) -> float | None:
        if self._desc.primary_metric is None:
            return None
        return self.metric_value(self._desc.primary_metric.name)

    @property
    def labels(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for lbl in self._desc.labels:
            v = get_nested(self._rec, lbl.source)
            if v is None:
                continue
            out[lbl.name] = str(v)
        return out

    # --- trajectory ---
    def messages(self) -> Iterator[MessageView]:
        raw = get_nested(self._rec, self._m.messages_field) or []
        if not isinstance(raw, list):
            return
        for m in raw:
            if not isinstance(m, dict):
                continue
            role = m.get("role") or ""
            content = m.get("content")
            if content is None:
                content = ""
            elif not isinstance(content, str):
                content = json.dumps(content)
            tcs: list[ToolCallView] = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                tcs.append(ToolCallView(
                    id=tc.get("id"),
                    name=fn.get("name") or "",
                    arguments=fn.get("arguments") or "",
                ))
            yield MessageView(
                role=role,
                content=content,
                tool_calls=tcs,
                tool_call_id=m.get("tool_call_id"),
            )

    # --- auxiliary ---
    def documents(self) -> list[DocumentView]:
        if not self._desc.has_documents:
            return []
        docs = get_nested(self._rec, self._m.documents_field) or []
        if not isinstance(docs, list):
            return []
        out: list[DocumentView] = []
        key = self._m.document_path_field
        for d in docs:
            if not isinstance(d, dict):
                continue
            p = d.get(key) if key else None
            if not isinstance(p, str):
                continue
            out.append(DocumentView(path=p, content=d.get("content")))
        return out

    @property
    def turns_used(self) -> int | None:
        return _coerce_int(get_nested(self._rec, self._m.turns_field))

    @property
    def tool_errors(self) -> int:
        return int(_coerce_int(get_nested(self._rec, self._m.tool_errors_field)) or 0)

    @property
    def tool_calls_total(self) -> int:
        n = _coerce_int(get_nested(self._rec, self._m.tool_calls_total_field))
        if n is not None:
            return n
        return super().tool_calls_total

    @property
    def usage(self) -> dict[str, int]:
        u = get_nested(self._rec, self._m.usage_field) or {}
        if not isinstance(u, dict):
            return {}
        out: dict[str, int] = {}
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            v = _coerce_int(u.get(k))
            if v is not None:
                out[k] = v
        return out
