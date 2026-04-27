from __future__ import annotations

from pathlib import Path

import pytest

from engine.sandbox.runtime_mounts import discover_python_runtime_mounts
from engine.sandbox.sandbox_availability import (
    SandboxBackend,
    render_unavailable_warning,
    resolve_sandbox_status,
)
from engine.sandbox.sandbox_config import SandboxConfig
from engine.sandbox.sandbox_runner import SandboxRunner
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder


@pytest.mark.asyncio
async def test_sandbox_runs_real_python_against_trace_store(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """Bubblewrap-backed sandbox executes user code with a working ``trace_store``."""
    status = resolve_sandbox_status()
    if not status.available:
        pytest.fail(
            "Linux sandbox unavailable in CI; this must work for release.\n"
            + render_unavailable_warning(status)
        )
    assert status.backend in (
        SandboxBackend.LINUX_BWRAP_SYSTEM,
        SandboxBackend.LINUX_BWRAP_PACKAGED,
    )

    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = tmp_path / "traces.jsonl.engine-index.jsonl"
    await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig(index_path=index_path)
    )

    runtime_mounts = discover_python_runtime_mounts()
    runner = SandboxRunner(sandbox_status=status, runtime_mounts=runtime_mounts)

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
