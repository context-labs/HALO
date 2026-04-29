from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class TraceIndexConfig(BaseModel):
    """Sidecar index location and schema version. Index is build-once: existing files are reused as-is."""

    model_config = ConfigDict(extra="forbid")

    index_path: Path | None = None
    schema_version: int = 1
