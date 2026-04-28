from __future__ import annotations

from pathlib import Path

import pytest

from engine.sandbox.linux_client import LinuxClient
from engine.sandbox.models import SandboxConfig
from engine.sandbox.sandbox import resolve_sandbox
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder


@pytest.mark.asyncio
async def test_sandbox_runs_real_python_against_trace_store(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """Bubblewrap-backed sandbox executes user code with a working ``trace_store``."""
    sandbox = resolve_sandbox(config=SandboxConfig(timeout_seconds=20.0))
    if sandbox is None:
        pytest.fail("Linux sandbox unavailable in CI; this must work for release.")
    assert isinstance(sandbox.client, LinuxClient)

    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = tmp_path / "traces.jsonl.engine-index.jsonl"
    await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig(index_path=index_path)
    )

    result = await sandbox.run_python(
        code="print('count=', trace_store.trace_count)",
        trace_path=trace_path,
        index_path=index_path,
    )

    assert result.exit_code == 0, (
        f"sandboxed run failed (exit={result.exit_code}, timed_out={result.timed_out}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "count= 3" in result.stdout
