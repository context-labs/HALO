from __future__ import annotations

from engine.sandbox.bootstrap import render_bootstrap_script


def test_bootstrap_script_includes_user_code() -> None:
    script = render_bootstrap_script(
        user_code="print(trace_store.trace_count)",
        trace_path="/host/path/traces.jsonl",
        index_path="/host/path/traces.jsonl.engine-index.jsonl",
    )
    assert "print(trace_store.trace_count)" in script
    assert "from engine.traces.trace_store import TraceStore" in script
    assert "/host/path/traces.jsonl" in script
    assert "/host/path/traces.jsonl.engine-index.jsonl" in script
    assert "import numpy" in script
    assert "import pandas" in script


def test_user_code_runs_in_isolated_namespace() -> None:
    script = render_bootstrap_script(
        user_code="x = 1",
        trace_path="/a",
        index_path="/b",
    )
    assert "exec(_USER_CODE" in script
