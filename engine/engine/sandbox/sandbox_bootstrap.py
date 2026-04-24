from __future__ import annotations


_TEMPLATE = '''\
from __future__ import annotations

import sys
from pathlib import Path

import numpy
import pandas

from engine.traces.trace_store import TraceStore

trace_store = TraceStore.load(
    trace_path=Path({trace_mount_path!r}),
    index_path=Path({index_mount_path!r}),
)

_USER_CODE = {user_code!r}

_globals = {{
    "trace_store": trace_store,
    "numpy": numpy,
    "pandas": pandas,
    "np": numpy,
    "pd": pandas,
    "Path": Path,
}}

exec(_USER_CODE, _globals, _globals)
'''


def render_bootstrap_script(
    *,
    user_code: str,
    trace_mount_path: str,
    index_mount_path: str,
) -> str:
    return _TEMPLATE.format(
        user_code=user_code,
        trace_mount_path=trace_mount_path,
        index_mount_path=index_mount_path,
    )
