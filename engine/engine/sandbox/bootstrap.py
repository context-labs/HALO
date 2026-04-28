from __future__ import annotations

# Bootstrap script template injected into Pyodide once per ``run_python``.
#
# Imports the stdlib-only TraceStore compat shim mounted at
# ``/halo/pyodide_trace_compat.py`` and constructs the ``user_globals`` dict
# the runner exposes to user code. Trace + index files are mounted at fixed
# virtual paths by ``Sandbox.run_python``.
_BOOTSTRAP_TEMPLATE = """\
import sys
sys.path.insert(0, "/halo")

import numpy
import pandas

from pyodide_trace_compat import TraceStore

trace_store = TraceStore.load(
    trace_path={trace_virtual_path!r},
    index_path={index_virtual_path!r},
)

user_globals = {{
    "trace_store": trace_store,
    "numpy": numpy,
    "pandas": pandas,
    "np": numpy,
    "pd": pandas,
}}
"""


def render_bootstrap_script(
    *,
    trace_virtual_path: str,
    index_virtual_path: str,
) -> str:
    """Render the Python bootstrap that imports the trace compat module and builds the user globals dict."""
    return _BOOTSTRAP_TEMPLATE.format(
        trace_virtual_path=trace_virtual_path,
        index_virtual_path=index_virtual_path,
    )
