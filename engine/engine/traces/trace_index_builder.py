from __future__ import annotations

from pathlib import Path

from engine.traces.models.trace_index_config import TraceIndexConfig


class TraceIndexBuilder:
    @classmethod
    async def ensure_index_exists(
        cls,
        trace_path: Path,
        config: TraceIndexConfig,
    ) -> Path:
        index_path = config.index_path or Path(str(trace_path) + ".engine-index.jsonl")
        meta_path = cls._meta_path_for(index_path)

        if index_path.exists() and meta_path.exists():
            return index_path

        await cls.build_index(
            trace_path=trace_path,
            index_path=index_path,
            meta_path=meta_path,
            schema_version=config.schema_version,
        )
        return index_path

    @staticmethod
    def _meta_path_for(index_path: Path) -> Path:
        name = index_path.name
        if name.endswith(".engine-index.jsonl"):
            return index_path.with_name(name[:-len(".jsonl")] + ".meta.json")
        return index_path.with_name(name + ".meta.json")

    @classmethod
    async def build_index(
        cls,
        trace_path: Path,
        index_path: Path,
        meta_path: Path,
        schema_version: int,
    ) -> None:
        # Real implementation lands in Task 2.2.
        raise NotImplementedError
