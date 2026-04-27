from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from engine.traces.models.trace_index_models import TraceIndexRow

if TYPE_CHECKING:
    from engine.traces.models.trace_query_models import (
        DatasetOverview,
        TraceCountResult,
        TraceFilters,
        TraceQueryResult,
        TraceSearchResult,
        TraceView,
    )


class TraceStore:
    """Pure read/query/render API over a built index plus the canonical JSONL.

    Deliberately depends only on stdlib + Pydantic + ``engine.traces.models`` so the
    sandbox can import and instantiate it directly inside user code, with no agent
    SDK / async runtime / tool dependencies in the import graph.
    """

    def __init__(self, trace_path: Path, index_path: Path, rows: list[TraceIndexRow]) -> None:
        """Hold paths plus the in-memory index rows; prefer ``load`` for constructing from disk."""
        self._trace_path = trace_path
        self._index_path = index_path
        self._rows = rows
        self._rows_by_id: dict[str, TraceIndexRow] = {r.trace_id: r for r in rows}

    @classmethod
    def load(cls, trace_path: Path, index_path: Path) -> "TraceStore":
        """Read the sidecar index file line-by-line and construct a TraceStore."""
        raw = index_path.read_text().splitlines()
        rows = [TraceIndexRow.model_validate_json(line) for line in raw if line]
        return cls(trace_path=trace_path, index_path=index_path, rows=rows)

    @property
    def trace_count(self) -> int:
        """Total trace count in the loaded index (no filtering)."""
        return len(self._rows)

    @property
    def trace_path(self) -> Path:
        """The canonical JSONL path this store reads spans from."""
        return self._trace_path

    @property
    def index_path(self) -> Path:
        """The sidecar index path this store was loaded from."""
        return self._index_path

    def view_trace(self, trace_id: str) -> "TraceView":
        """Read all spans of one trace by seeking to each indexed byte offset and parsing as SpanRecord."""
        from engine.traces.models.canonical_span import SpanRecord
        from engine.traces.models.trace_query_models import TraceView

        if trace_id not in self._rows_by_id:
            raise KeyError(trace_id)
        row = self._rows_by_id[trace_id]

        with self._trace_path.open("rb") as fh:
            spans: list[SpanRecord] = []
            for offset, length in zip(row.byte_offsets, row.byte_lengths, strict=True):
                fh.seek(offset)
                blob = fh.read(length)
                spans.append(SpanRecord.model_validate_json(blob))
        return TraceView(trace_id=trace_id, spans=spans)

    def query_traces(
        self,
        filters: "TraceFilters",
        limit: int = 50,
        offset: int = 0,
    ) -> "TraceQueryResult":
        """Filter rows in memory, slice with ``offset:offset+limit``, and project each into a TraceSummary."""
        from engine.traces.models.trace_query_models import TraceQueryResult, TraceSummary

        filtered = [row for row in self._rows if _matches_filters(row, filters)]
        summaries = [
            TraceSummary(
                trace_id=row.trace_id,
                span_count=row.span_count,
                start_time=row.start_time,
                end_time=row.end_time,
                has_errors=row.has_errors,
                service_names=row.service_names,
                model_names=row.model_names,
                total_input_tokens=row.total_input_tokens,
                total_output_tokens=row.total_output_tokens,
                agent_names=row.agent_names,
            )
            for row in filtered[offset : offset + limit]
        ]
        return TraceQueryResult(traces=summaries, total=len(filtered))

    def count_traces(self, filters: "TraceFilters") -> "TraceCountResult":
        """Count matching rows without materializing summaries — cheaper than ``query_traces``."""
        from engine.traces.models.trace_query_models import TraceCountResult

        total = sum(1 for row in self._rows if _matches_filters(row, filters))
        return TraceCountResult(total=total)

    def get_overview(self, filters: "TraceFilters") -> "DatasetOverview":
        """Aggregate the filtered subset into a single DatasetOverview rollup row."""
        from engine.traces.models.trace_query_models import DatasetOverview

        rows = [r for r in self._rows if _matches_filters(r, filters)]
        if not rows:
            return DatasetOverview(
                total_traces=0, total_spans=0, earliest_start_time="",
                latest_end_time="", service_names=[], model_names=[], agent_names=[],
                error_trace_count=0, total_input_tokens=0, total_output_tokens=0,
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

    def search_trace(self, trace_id: str, pattern: str) -> "TraceSearchResult":
        """Substring search on raw JSON span text within one trace; returns matching span lines verbatim."""
        from engine.traces.models.trace_query_models import TraceSearchResult

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
        """Render a trace as plain text suitable for prompt/tool consumption, truncated to ``budget`` bytes."""
        view = self.view_trace(trace_id)
        lines: list[str] = [f"trace_id: {trace_id}", f"spans: {len(view.spans)}"]
        for s in view.spans:
            lines.append(
                f"- span_id={s.span_id} parent={s.parent_span_id or '∅'} "
                f"name={s.name} kind={s.kind} status={s.status.code}"
            )
            lines.append(f"  start={s.start_time} end={s.end_time}")
            model = s.attributes.get("inference.llm.model_name") or s.attributes.get("llm.model_name")
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


def _matches_filters(row: TraceIndexRow, filters: "TraceFilters") -> bool:
    """ANDed predicate: row passes only if every set filter matches."""
    if filters.has_errors is not None and row.has_errors != filters.has_errors:
        return False
    if filters.model_names is not None and not any(m in row.model_names for m in filters.model_names):
        return False
    if filters.service_names is not None and not any(s in row.service_names for s in filters.service_names):
        return False
    if filters.agent_names is not None and not any(a in row.agent_names for a in filters.agent_names):
        return False
    if filters.project_id is not None and row.project_id != filters.project_id:
        return False
    if filters.start_time_gte is not None and row.start_time < filters.start_time_gte:
        return False
    if filters.end_time_lte is not None and row.end_time > filters.end_time_lte:
        return False
    return True
