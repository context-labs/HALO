from __future__ import annotations

from pathlib import Path

import pytest

from engine.sandbox.sandbox_availability import SandboxBackend, resolve_sandbox_runtime
from engine.sandbox.sandbox_config import SandboxConfig
from engine.sandbox.sandbox_runner import SandboxRunner
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder


@pytest.mark.asyncio
async def test_sandbox_runs_real_python_against_trace_store(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """``sandbox-exec``-backed sandbox executes user code with a working ``trace_store``."""
    sandbox = resolve_sandbox_runtime()
    if sandbox is None:
        pytest.fail("macOS sandbox unavailable in CI; this must work for release.")
    assert sandbox.backend == SandboxBackend.MACOS_SANDBOX_EXEC

    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = tmp_path / "traces.jsonl.engine-index.jsonl"
    await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig(index_path=index_path)
    )

    runner = SandboxRunner(sandbox=sandbox)

    result = await runner.run_python(
        code="print('count=', trace_store.trace_count)",
        trace_path=trace_path,
        index_path=index_path,
        config=SandboxConfig(timeout_seconds=20.0),
    )

    assert result.exit_code == 0, (
        f"sandboxed run failed (exit={result.exit_code}, timed_out={result.timed_out}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "count= 3" in result.stdout
