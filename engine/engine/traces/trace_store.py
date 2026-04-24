from __future__ import annotations

from pathlib import Path

from engine.traces.models.trace_index_models import TraceIndexRow


class TraceStore:
    def __init__(self, trace_path: Path, index_path: Path, rows: list[TraceIndexRow]) -> None:
        self._trace_path = trace_path
        self._index_path = index_path
        self._rows = rows
        self._rows_by_id: dict[str, TraceIndexRow] = {r.trace_id: r for r in rows}

    @classmethod
    def load(cls, trace_path: Path, index_path: Path) -> "TraceStore":
        raw = index_path.read_text().splitlines()
        rows = [TraceIndexRow.model_validate_json(line) for line in raw if line]
        return cls(trace_path=trace_path, index_path=index_path, rows=rows)

    @property
    def trace_count(self) -> int:
        return len(self._rows)

    def view_trace(self, trace_id: str) -> "TraceView":
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


def _matches_filters(row: TraceIndexRow, filters: "TraceFilters") -> bool:
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
