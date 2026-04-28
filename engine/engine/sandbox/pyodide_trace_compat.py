"""Stdlib-only TraceStore compat shim, injected into the Pyodide FS at sandbox boot.

Mirrors the public surface of :mod:`engine.traces.trace_store` and
:mod:`engine.traces.models.trace_query_models` that user analysis code may
touch — ``TraceStore``, ``TraceFilters``, ``query_traces``, ``count_traces``,
``view_trace``, ``search_trace``, ``get_overview``, ``render_trace``,
``trace_count`` — but reimplemented with only the Python standard library so
it imports cleanly inside Pyodide (no pydantic, no ``engine.*``).

Field semantics intentionally match the host module so REPL code written
against the host trace store keeps working under WASM. Parsing is pure
``json``: the host writes the index file with ``model_dump_json``, which
serializes attribute keys verbatim, so reading it back as plain dicts is
lossless for the fields this shim exposes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceFilters:
    """Optional ANDed predicates over the index. Mirrors the host TraceFilters."""

    has_errors: bool | None = None
    model_names: list[str] | None = None
    service_names: list[str] | None = None
    agent_names: list[str] | None = None
    project_id: str | None = None
    start_time_gte: str | None = None
    end_time_lte: str | None = None


@dataclass
class TraceSummary:
    trace_id: str
    span_count: int
    start_time: str
    end_time: str
    has_errors: bool
    service_names: list[str]
    model_names: list[str]
    total_input_tokens: int
    total_output_tokens: int
    agent_names: list[str]


@dataclass
class TraceQueryResult:
    traces: list[TraceSummary]
    total: int


@dataclass
class TraceCountResult:
    total: int


@dataclass
class TraceSearchResult:
    trace_id: str
    match_count: int
    matches: list[str]


@dataclass
class SpanStatus:
    code: str
    message: str = ""


@dataclass
class SpanResource:
    attributes: dict[str, Any]


@dataclass
class SpanScope:
    name: str
    version: str = ""


@dataclass
class SpanRecord:
    """Permissive span: declares the few fields render/search touch and stuffs the rest into ``extra``."""

    trace_id: str
    span_id: str
    name: str
    kind: str
    start_time: str
    end_time: str
    status: SpanStatus
    resource: SpanResource
    scope: SpanScope
    attributes: dict[str, Any]
    parent_span_id: str = ""
    trace_state: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpanRecord":
        status = data.get("status") or {}
        resource = data.get("resource") or {}
        scope = data.get("scope") or {}
        known = {
            "trace_id",
            "span_id",
            "parent_span_id",
            "trace_state",
            "name",
            "kind",
            "start_time",
            "end_time",
            "status",
            "resource",
            "scope",
            "attributes",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id", ""),
            trace_state=data.get("trace_state", ""),
            name=data["name"],
            kind=data["kind"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            status=SpanStatus(code=status.get("code", ""), message=status.get("message", "")),
            resource=SpanResource(attributes=resource.get("attributes", {})),
            scope=SpanScope(name=scope.get("name", ""), version=scope.get("version", "")),
            attributes=data.get("attributes", {}),
            extra=extra,
        )


@dataclass
class TraceView:
    trace_id: str
    spans: list[SpanRecord]


@dataclass
class DatasetOverview:
    total_traces: int
    total_spans: int
    earliest_start_time: str
    latest_end_time: str
    service_names: list[str]
    model_names: list[str]
    agent_names: list[str]
    error_trace_count: int
    total_input_tokens: int
    total_output_tokens: int


@dataclass
class _IndexRow:
    trace_id: str
    byte_offsets: list[int]
    byte_lengths: list[int]
    span_count: int
    start_time: str
    end_time: str
    has_errors: bool
    service_names: list[str]
    model_names: list[str]
    total_input_tokens: int
    total_output_tokens: int
    project_id: str | None
    agent_names: list[str]


def _row_from_dict(d: dict[str, Any]) -> _IndexRow:
    return _IndexRow(
        trace_id=d["trace_id"],
        byte_offsets=list(d["byte_offsets"]),
        byte_lengths=list(d["byte_lengths"]),
        span_count=int(d["span_count"]),
        start_time=d["start_time"],
        end_time=d["end_time"],
        has_errors=bool(d["has_errors"]),
        service_names=list(d.get("service_names", [])),
        model_names=list(d.get("model_names", [])),
        total_input_tokens=int(d.get("total_input_tokens", 0)),
        total_output_tokens=int(d.get("total_output_tokens", 0)),
        project_id=d.get("project_id"),
        agent_names=list(d.get("agent_names", [])),
    )


def _matches(row: _IndexRow, f: TraceFilters) -> bool:
    """ANDed predicate evaluation; mirrors the host TraceFilters semantics."""
    if f.has_errors is not None and row.has_errors != f.has_errors:
        return False
    if f.model_names is not None and not any(m in row.model_names for m in f.model_names):
        return False
    if f.service_names is not None and not any(s in row.service_names for s in f.service_names):
        return False
    if f.agent_names is not None and not any(a in row.agent_names for a in f.agent_names):
        return False
    if f.project_id is not None and row.project_id != f.project_id:
        return False
    if f.start_time_gte is not None and row.start_time < f.start_time_gte:
        return False
    if f.end_time_lte is not None and row.end_time > f.end_time_lte:
        return False
    return True


class TraceStore:
    """Pyodide-side TraceStore: read JSONL index + canonical traces with stdlib only."""

    def __init__(self, trace_path: Path, index_path: Path, rows: list[_IndexRow]) -> None:
        self._trace_path = trace_path
        self._index_path = index_path
        self._rows = rows
        self._rows_by_id: dict[str, _IndexRow] = {r.trace_id: r for r in rows}

    @classmethod
    def load(cls, trace_path: Path, index_path: Path) -> "TraceStore":
        raw = Path(index_path).read_text().splitlines()
        rows = [_row_from_dict(json.loads(line)) for line in raw if line]
        return cls(trace_path=Path(trace_path), index_path=Path(index_path), rows=rows)

    @property
    def trace_count(self) -> int:
        return len(self._rows)

    @property
    def trace_path(self) -> Path:
        return self._trace_path

    @property
    def index_path(self) -> Path:
        return self._index_path

    def view_trace(self, trace_id: str) -> TraceView:
        if trace_id not in self._rows_by_id:
            raise KeyError(trace_id)
        row = self._rows_by_id[trace_id]
        spans: list[SpanRecord] = []
        with self._trace_path.open("rb") as fh:
            for offset, length in zip(row.byte_offsets, row.byte_lengths, strict=True):
                fh.seek(offset)
                blob = fh.read(length)
                spans.append(SpanRecord.from_dict(json.loads(blob)))
        return TraceView(trace_id=trace_id, spans=spans)

    def query_traces(
        self, filters: TraceFilters, limit: int = 50, offset: int = 0
    ) -> TraceQueryResult:
        filtered = [row for row in self._rows if _matches(row, filters)]
        summaries = [
            TraceSummary(
                trace_id=row.trace_id,
                span_count=row.span_count,
                start_time=row.start_time,
                end_time=row.end_time,
                has_errors=row.has_errors,
                service_names=list(row.service_names),
                model_names=list(row.model_names),
                total_input_tokens=row.total_input_tokens,
                total_output_tokens=row.total_output_tokens,
                agent_names=list(row.agent_names),
            )
            for row in filtered[offset : offset + limit]
        ]
        return TraceQueryResult(traces=summaries, total=len(filtered))

    def count_traces(self, filters: TraceFilters) -> TraceCountResult:
        return TraceCountResult(total=sum(1 for r in self._rows if _matches(r, filters)))

    def get_overview(self, filters: TraceFilters) -> DatasetOverview:
        rows = [r for r in self._rows if _matches(r, filters)]
        if not rows:
            return DatasetOverview(
                total_traces=0,
                total_spans=0,
                earliest_start_time="",
                latest_end_time="",
                service_names=[],
                model_names=[],
                agent_names=[],
                error_trace_count=0,
                total_input_tokens=0,
                total_output_tokens=0,
            )

        services: set[str] = set()
        models: set[str] = set()
        agents: set[str] = set()
        for r in rows:
            services.update(r.service_names)
            models.update(r.model_names)
            agents.update(r.agent_names)

        return DatasetOverview(
            total_traces=len(rows),
            total_spans=sum(r.span_count for r in rows),
            earliest_start_time=min(r.start_time for r in rows),
            latest_end_time=max(r.end_time for r in rows),
            service_names=sorted(services),
            model_names=sorted(models),
            agent_names=sorted(agents),
            error_trace_count=sum(1 for r in rows if r.has_errors),
            total_input_tokens=sum(r.total_input_tokens for r in rows),
            total_output_tokens=sum(r.total_output_tokens for r in rows),
        )

    def search_trace(self, trace_id: str, pattern: str) -> TraceSearchResult:
        if trace_id not in self._rows_by_id:
            raise KeyError(trace_id)
        row = self._rows_by_id[trace_id]
        matches: list[str] = []
        with self._trace_path.open("rb") as fh:
            for offset, length in zip(row.byte_offsets, row.byte_lengths, strict=True):
                fh.seek(offset)
                blob = fh.read(length).decode("utf-8", errors="replace")
                if pattern in blob:
                    matches.append(blob)
        return TraceSearchResult(trace_id=trace_id, match_count=len(matches), matches=matches)

    def render_trace(self, trace_id: str, budget: int) -> str:
        view = self.view_trace(trace_id)
        lines: list[str] = [f"trace_id: {trace_id}", f"spans: {len(view.spans)}"]
        for s in view.spans:
            lines.append(
                f"- span_id={s.span_id} parent={s.parent_span_id or '∅'} "
                f"name={s.name} kind={s.kind} status={s.status.code}"
            )
            lines.append(f"  start={s.start_time} end={s.end_time}")
            model = s.attributes.get("inference.llm.model_name") or s.attributes.get(
                "llm.model_name"
            )
            if model:
                lines.append(f"  model={model}")
            in_tok = s.attributes.get("inference.llm.input_tokens")
            out_tok = s.attributes.get("inference.llm.output_tokens")
            if in_tok is not None or out_tok is not None:
                lines.append(f"  tokens: input={in_tok} output={out_tok}")

        rendered = "\n".join(lines)
        if len(rendered) > budget:
            return rendered[:budget] + "... [truncated]"
        return rendered


__all__ = [
    "DatasetOverview",
    "SpanRecord",
    "TraceCountResult",
    "TraceFilters",
    "TraceQueryResult",
    "TraceSearchResult",
    "TraceStore",
    "TraceSummary",
    "TraceView",
]
