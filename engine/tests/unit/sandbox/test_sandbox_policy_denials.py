from __future__ import annotations

import platform
from pathlib import Path

import pytest

from engine.sandbox.sandbox_config import SandboxConfig
from engine.sandbox.sandbox_runner import SandboxRunner
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder

from .conftest import bwrap_can_sandbox


def _sandbox_ready(tmp_path: Path, fixtures_dir: Path) -> tuple[SandboxRunner, Path, Path]:
    system = platform.system()
    if system == "Linux" and not bwrap_can_sandbox():
        pytest.skip("bubblewrap unavailable or kernel disallows user-namespace sandboxing")
    if system not in ("Linux", "Darwin"):
        pytest.skip(f"unsupported platform {system}")
    venv = Path(__file__).resolve().parents[3] / ".sandbox-venv"
    if not (venv / "bin" / "python").exists():
        pytest.skip("sandbox venv not built")

    trace_path = tmp_path / "t.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    return SandboxRunner(sandbox_venv=venv), trace_path, tmp_path


@pytest.mark.asyncio
async def test_cannot_write_outside_workspace(tmp_path: Path, fixtures_dir: Path) -> None:
    runner, trace_path, _ = _sandbox_ready(tmp_path, fixtures_dir)
    idx = await TraceIndexBuilder.ensure_index_exists(trace_path=trace_path, config=TraceIndexConfig())
    result = await runner.run_python(
        code="open('/etc/attack', 'w').write('no')",
        trace_path=trace_path,
        index_path=idx,
        config=SandboxConfig(timeout_seconds=10.0),
    )
    assert result.exit_code != 0
    assert "PermissionError" in result.stderr or "Read-only" in result.stderr or "not permitted" in result.stderr


@pytest.mark.asyncio
async def test_cannot_read_outside_allowed(tmp_path: Path, fixtures_dir: Path) -> None:
    runner, trace_path, _ = _sandbox_ready(tmp_path, fixtures_dir)
    idx = await TraceIndexBuilder.ensure_index_exists(trace_path=trace_path, config=TraceIndexConfig())
    result = await runner.run_python(
        code="print(open('/etc/passwd').read()[:10])",
        trace_path=trace_path,
        index_path=idx,
        config=SandboxConfig(timeout_seconds=10.0),
    )
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_no_network(tmp_path: Path, fixtures_dir: Path) -> None:
    runner, trace_path, _ = _sandbox_ready(tmp_path, fixtures_dir)
    idx = await TraceIndexBuilder.ensure_index_exists(trace_path=trace_path, config=TraceIndexConfig())
    result = await runner.run_python(
        code=("import socket; s = socket.socket(); s.connect(('1.1.1.1', 80))"),
        trace_path=trace_path,
        index_path=idx,
        config=SandboxConfig(timeout_seconds=5.0),
    )
    assert result.exit_code != 0
