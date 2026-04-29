"""Pyodide-side runtime: bootstrap helper and capture-aware exec wrapper.

This file is loaded **inside Pyodide**, not by the host engine. ``runner.js``
reads it at boot via ``import.meta.url`` and runs it through
``pyodide.runPython`` so the two callables below become live Python
functions in the WASM interpreter. Every per-execute round trip then goes
through them — no JS-side string templates, no per-call Python code
generation on the host.

* :func:`halo_bootstrap` runs once after mounts: imports numpy/pandas plus
  the real ``engine.traces.trace_store`` (the runner stages the host's
  ``engine`` package source into ``/halo/`` at boot, so the same module
  the host runs is what user code sees — no parallel shim), loads the
  index, and stashes the resulting user-facing globals (``trace_store``,
  ``numpy``, ``pandas``, ``np``, ``pd``) in module state for later execs.
* :func:`halo_execute` runs every user-code request: exec's the source
  against the prebuilt globals with stdout/stderr captured into
  ``StringIO`` so the host gets clean text back regardless of whether the
  code crashed.

Splitting them lets the host distinguish setup failures (malformed
index, missing dependency) from user-code failures (assertion errors,
imports the agent shouldn't have tried) — both surface through the same
``{exit_code, stdout, stderr}`` shape but the host knows which phase
they came from.
"""

from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path
from typing import Any

# Module-state pair owned by ``halo_bootstrap`` and read by ``halo_execute``.
# Kept private (single underscore) so user code can't bind references to
# them through the globals dict by accident.
_user_globals: dict[str, Any] = {}
_bootstrapped: bool = False


def halo_bootstrap(trace_path: str, index_path: str) -> dict[str, Any]:
    """Build the user-facing globals dict from mounted trace + index files.

    Runs once per sandbox session, before the first ``halo_execute``.
    Imports numpy/pandas plus the real ``engine.traces.trace_store`` (the
    runner stages the host's ``engine`` package source into ``/halo/`` at
    boot, which is why the import works in WASM), loads the index from
    the mounted virtual paths, and stashes the resulting ``trace_store``
    plus convenience aliases. Returns the standard capture envelope so a
    setup failure (malformed index, missing dependency) surfaces with a
    real traceback rather than a blank ``halo_execute`` failure later.
    """
    global _bootstrapped
    buf_stdout = io.StringIO()
    buf_stderr = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = buf_stdout, buf_stderr

    exit_code = 0
    try:
        # ``/halo`` is where the runner stages the engine package; putting
        # it on sys.path lets ``import engine.traces.trace_store`` resolve
        # the same module the host runs against — no parallel shim.
        if "/halo" not in sys.path:
            sys.path.insert(0, "/halo")

        import numpy
        import pandas

        from engine.traces.trace_store import TraceStore

        trace_store = TraceStore.load(
            trace_path=Path(trace_path),
            index_path=Path(index_path),
        )
        _user_globals.clear()
        _user_globals.update(
            {
                "trace_store": trace_store,
                "numpy": numpy,
                "pandas": pandas,
                "np": numpy,
                "pd": pandas,
            }
        )
        _bootstrapped = True
    except BaseException:
        exit_code = 1
        traceback.print_exc()
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    return {
        "exit_code": exit_code,
        "stdout": buf_stdout.getvalue(),
        "stderr": buf_stderr.getvalue(),
    }


def halo_execute(code: str) -> dict[str, Any]:
    """Exec user ``code`` against the bootstrap globals with captured I/O.

    The capture is the only reason this isn't a one-line ``exec``: user
    ``print`` calls and any traceback from an uncaught exception both
    need to come back to the host as fields on the response object, not
    interleaved on the runner's actual stdout where they'd corrupt the
    JSON-RPC stream.

    Bootstrapping is a precondition; calling without it raises a
    ``RuntimeError`` so the host gets a clear message instead of a
    bewildering ``NameError`` on ``trace_store``.
    """
    buf_stdout = io.StringIO()
    buf_stderr = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = buf_stdout, buf_stderr

    exit_code = 0
    try:
        if not _bootstrapped:
            raise RuntimeError(
                "sandbox not bootstrapped: halo_bootstrap must run before halo_execute"
            )
        exec(compile(code, "<sandbox>", "exec"), _user_globals, _user_globals)
    except BaseException:
        exit_code = 1
        traceback.print_exc()
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    return {
        "exit_code": exit_code,
        "stdout": buf_stdout.getvalue(),
        "stderr": buf_stderr.getvalue(),
    }
