from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from engine.sandbox.models import CodeExecutionResult, RunCodeArguments
from engine.sandbox.sandbox import Sandbox
from engine.tools.run_code_tool import RunCodeTool
from engine.tools.tool_protocol import ToolContext
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore


@pytest.mark.asyncio
async def test_run_code_tool_delegates_to_sandbox_in_tool_context(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    trace_path = tmp_path / "t.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    store = TraceStore.load(trace_path=trace_path, index_path=index_path)

    fake_sandbox = MagicMock(spec=Sandbox)
    fake_sandbox.run_python = AsyncMock(
        return_value=CodeExecutionResult(
            exit_code=0,
            stdout="ok",
            stderr="",
            timed_out=False,
        )
    )
    ctx = ToolContext.model_construct(trace_store=store, sandbox=fake_sandbox)

    tool = RunCodeTool()
    result = await tool.run(ctx, RunCodeArguments(code="print('hello')"))

    assert result.exit_code == 0
    fake_sandbox.run_python.assert_awaited_once_with(
        code="print('hello')",
        trace_path=store.trace_path,
        index_path=store.index_path,
    )


@pytest.mark.asyncio
async def test_run_code_tool_raises_when_sandbox_missing_from_tool_context(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    """If the tool factory ever registers ``run_code`` without populating
    ``ToolContext.sandbox``, ``run`` must fail loudly rather than silently
    do nothing."""
    trace_path = tmp_path / "t.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    store = TraceStore.load(trace_path=trace_path, index_path=index_path)

    ctx = ToolContext.model_construct(trace_store=store, sandbox=None)

    tool = RunCodeTool()
    with pytest.raises(RuntimeError, match="ToolContext.sandbox required"):
        await tool.run(ctx, RunCodeArguments(code="print('hello')"))
