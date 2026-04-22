"""Random-access reader for a grepfruit-style JSONL dataset.

Uses byte offsets recorded by ``indexer`` to seek to the start of a record
and read exactly its bytes, so we never scan the full file just to fetch one
trace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TraceReader:
    """Random-access reader that opens the source JSONL once and seeks per request."""

    def __init__(self, jsonl_path: Path):
        self.jsonl_path = Path(jsonl_path)
        self._file: Any = None

    def __enter__(self) -> TraceReader:
        self._file = open(self.jsonl_path, "rb")
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _ensure_open(self) -> Any:
        if self._file is None:
            self._file = open(self.jsonl_path, "rb")
        return self._file

    def read_raw(self, byte_offset: int, byte_length: int) -> bytes:
        """Read exactly one line from the file at the given byte offset."""
        f = self._ensure_open()
        f.seek(byte_offset)
        return f.read(byte_length)

    def read(self, byte_offset: int, byte_length: int) -> dict[str, Any]:
        """Read one record and parse it as JSON."""
        return json.loads(self.read_raw(byte_offset, byte_length))
