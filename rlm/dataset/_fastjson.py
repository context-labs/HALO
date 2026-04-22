"""Single import surface for a fast JSON parser.

orjson is 3-5x faster than stdlib ``json`` on the large trace records we
process and releases the GIL during parsing, so we always prefer it when
installed. The fallback keeps unit tests runnable on bare-Python environments.
"""

from __future__ import annotations

from typing import Any

try:
    import orjson

    def loads(data: bytes | str) -> Any:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return orjson.loads(data)

    def dumps(obj: Any) -> bytes:
        # OPT_NON_STR_KEYS is a small robustness shim — TraceSummary-like dicts
        # use str keys everywhere so this is a safety net, not a hot path need.
        return orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS)

    HAS_ORJSON = True

except ImportError:  # pragma: no cover - tested only when orjson is absent
    import json as _json

    def loads(data: bytes | str) -> Any:
        return _json.loads(data)

    def dumps(obj: Any) -> bytes:
        return _json.dumps(obj, ensure_ascii=False).encode("utf-8")

    HAS_ORJSON = False
