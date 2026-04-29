from __future__ import annotations

from pathlib import Path

from engine.traces.models.trace_index_config import TraceIndexConfig


def test_defaults() -> None:
    cfg = TraceIndexConfig()
    assert cfg.index_path is None
    assert cfg.schema_version == 1


def test_explicit_index_path(tmp_path: Path) -> None:
    cfg = TraceIndexConfig(index_path=tmp_path / "idx.jsonl", schema_version=1)
    assert cfg.index_path == tmp_path / "idx.jsonl"
