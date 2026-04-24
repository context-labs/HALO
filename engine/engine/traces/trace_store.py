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
