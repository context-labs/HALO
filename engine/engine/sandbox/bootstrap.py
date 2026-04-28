from __future__ import annotations

_TEMPLATE = """\
from __future__ import annotations

import sys
from pathlib import Path

import numpy
import pandas

from engine.traces.trace_store import TraceStore

trace_store = TraceStore.load(
    trace_path=Path({trace_path!r}),
    index_path=Path({index_path!r}),
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
"""


def render_bootstrap_script(
    *,
    user_code: str,
    trace_path: str,
    index_path: str,
) -> str:
    """Render the wrapper script the sandbox runs: preloads ``trace_store``, numpy, pandas, then exec's user code.

    Trace and index paths are passed as the host paths because both Linux
    (bubblewrap) and macOS (sandbox-exec) bind those files at their original
    locations.
    """
    return _TEMPLATE.format(
        user_code=user_code,
        trace_path=trace_path,
        index_path=index_path,
    )
