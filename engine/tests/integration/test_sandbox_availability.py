from __future__ import annotations

import pytest

from engine.sandbox.sandbox import Sandbox


def test_sandbox_resolve_returns_working_sandbox() -> None:
    """End-to-end probe must succeed; failure here is a release blocker.

    A successful resolve means: Deno is on PATH, the Pyodide npm package
    is in the Deno cache, and the wheels (numpy/pandas/etc.) the runner
    preloads are pre-cached next to ``pyodide.asm.wasm``. If any of those
    are missing the sandbox can't run user code under HALO's locked-down
    permissions and ``run_code`` would silently disappear from the agent
    surface — exactly the kind of release blocker we want to catch in CI.
    """
    sandbox = Sandbox.resolve()
    if sandbox is None:
        pytest.fail("Pyodide sandbox unavailable in CI; this must work for release.")

    assert sandbox.deno_executable.is_file()
    assert sandbox.runner_path.is_file()
    assert sandbox.runtime_path.is_file()
    assert sandbox.trace_compat_path.is_file()
