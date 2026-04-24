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
