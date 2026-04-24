from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class TraceIndexConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_path: Path | None = None
    schema_version: int = 1
