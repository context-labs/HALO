from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from engine.sandbox.runtime_mounts import PythonRuntimeMounts
from engine.sandbox.sandbox_availability import SandboxBackend, SandboxRuntime
from engine.sandbox.sandbox_config import CodeExecutionResult, RunCodeArguments, SandboxConfig
from engine.tools.run_code_tool import RunCodeTool
from engine.tools.tool_protocol import ToolContext
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore


def _sandbox(tmp_path: Path) -> SandboxRuntime:
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.write_text("")
    return SandboxRuntime(
        backend=SandboxBackend.LINUX_BWRAP_SYSTEM,
        executable=bwrap,
        runtime_mounts=PythonRuntimeMounts(
            python_executable=python,
            runtime_paths=(),
            library_paths=(),
        ),
    )


@pytest.mark.asyncio
async def test_run_code_tool_delegates_to_sandbox_runner(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    trace_path = tmp_path / "t.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    store = TraceStore.load(trace_path=trace_path, index_path=index_path)

    fake_runner = AsyncMock()
    fake_runner.run_python = AsyncMock(
        return_value=CodeExecutionResult(
            exit_code=0,
            stdout="ok",
            stderr="",
            timed_out=False,
        )
    )
    ctx = ToolContext.model_construct(trace_store=store, sandbox_runner=fake_runner)

    tool = RunCodeTool(
        sandbox_config=SandboxConfig(),
        sandbox=_sandbox(tmp_path),
    )
    result = await tool.run(ctx, RunCodeArguments(code="print('hello')"))
    assert result.exit_code == 0
    fake_runner.run_python.assert_awaited_once()
