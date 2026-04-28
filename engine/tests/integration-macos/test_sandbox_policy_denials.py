from __future__ import annotations

from pathlib import Path

import pytest

from engine.sandbox.sandbox_availability import resolve_sandbox_runtime
from engine.sandbox.sandbox_config import SandboxConfig
from engine.sandbox.sandbox_runner import SandboxRunner
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder


async def _ready(tmp_path: Path, fixtures_dir: Path) -> tuple[SandboxRunner, Path, Path]:
    sandbox = resolve_sandbox_runtime()
    if sandbox is None:
        pytest.fail("macOS sandbox unavailable in CI; this must work for release.")

    runner = SandboxRunner(sandbox=sandbox)

    trace_path = tmp_path / "t.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    return runner, trace_path, index_path


@pytest.mark.asyncio
async def test_cannot_write_outside_workspace(tmp_path: Path, fixtures_dir: Path) -> None:
    runner, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await runner.run_python(
        code="open('/etc/attack', 'w').write('no')",
        trace_path=trace_path,
        index_path=index_path,
        config=SandboxConfig(timeout_seconds=10.0),
    )
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_cannot_read_outside_allowed(tmp_path: Path, fixtures_dir: Path) -> None:
    runner, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await runner.run_python(
        code="print(open('/etc/master.passwd').read()[:10])",
        trace_path=trace_path,
        index_path=index_path,
        config=SandboxConfig(timeout_seconds=10.0),
    )
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_no_network(tmp_path: Path, fixtures_dir: Path) -> None:
    runner, trace_path, index_path = await _ready(tmp_path, fixtures_dir)
    result = await runner.run_python(
        code="import socket; s = socket.socket(); s.connect(('1.1.1.1', 80))",
        trace_path=trace_path,
        index_path=index_path,
        config=SandboxConfig(timeout_seconds=5.0),
    )
    assert result.exit_code != 0
