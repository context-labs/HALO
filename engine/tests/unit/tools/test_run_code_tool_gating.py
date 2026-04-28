from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from engine.agents.agent_config import AgentConfig
from engine.agents.agent_context import AgentContext
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.engine_run_state import EngineRunState
from engine.engine_config import EngineConfig
from engine.model_config import ModelConfig
from engine.sandbox.sandbox import Sandbox
from engine.tools.subagent_tool_factory import _child_tools_for_depth


def _engine_config() -> EngineConfig:
    agent = AgentConfig(
        name="root",
        instructions="",
        model=ModelConfig(name="claude-sonnet-4-5"),
        maximum_turns=10,
    )
    return EngineConfig(
        root_agent=agent,
        subagent=agent,
        synthesis_model=ModelConfig(name="claude-haiku-4-5"),
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        maximum_depth=2,
    )


def _parent() -> AgentExecution:
    return AgentExecution(
        agent_id="parent-x",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )


def _parent_context() -> AgentContext:
    return AgentContext(
        items=[],
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        text_message_compaction_keep_last_messages=2,
        tool_call_compaction_keep_last_turns=2,
    )


def _semaphores() -> dict[int, asyncio.Semaphore]:
    return {depth: asyncio.Semaphore(1) for depth in range(1, 4)}


def _sandbox(tmp_path: Path) -> Sandbox:
    """Stub Sandbox: this test only checks tool registration, never invokes the sandbox."""
    deno = tmp_path / "deno"
    deno.write_text("")
    runner = tmp_path / "runner.js"
    runner.write_text("")
    runtime = tmp_path / "pyodide_runtime.py"
    runtime.write_text("")
    trace_compat = tmp_path / "pyodide_trace_compat.py"
    trace_compat.write_text("")
    deno_dir = tmp_path / "deno-cache"
    deno_dir.mkdir()
    return Sandbox(
        deno_executable=deno,
        runner_path=runner,
        runtime_path=runtime,
        trace_compat_path=trace_compat,
        deno_dir=deno_dir,
    )


def _run_state(*, sandbox: Sandbox | None) -> EngineRunState:
    run_state = MagicMock(spec=EngineRunState)
    run_state.config = _engine_config()
    run_state.output_bus = EngineOutputBus()
    run_state.trace_store = MagicMock()
    run_state.sandbox = sandbox
    return run_state


def test_run_code_registered_when_sandbox_available(tmp_path: Path) -> None:
    run_state = _run_state(sandbox=_sandbox(tmp_path))

    tools = _child_tools_for_depth(
        depth=0,
        run_state=run_state,
        semaphores_by_depth=_semaphores(),
        parent_execution=_parent(),
        parent_context=_parent_context(),
    )

    assert "run_code" in {t.name for t in tools}


def test_run_code_omitted_when_sandbox_unavailable() -> None:
    run_state = _run_state(sandbox=None)

    tools = _child_tools_for_depth(
        depth=0,
        run_state=run_state,
        semaphores_by_depth=_semaphores(),
        parent_execution=_parent(),
        parent_context=_parent_context(),
    )

    names = {t.name for t in tools}
    assert "run_code" not in names
    assert "get_dataset_overview" in names
    assert "synthesize_traces" in names
