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
from engine.sandbox.runtime_mounts import PythonRuntimeMounts
from engine.sandbox.sandbox_availability import (
    SandboxBackend,
    SandboxStatus,
    SandboxUnavailableReason,
)
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


def _runtime_mounts(tmp_path: Path) -> PythonRuntimeMounts:
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.write_text("")
    return PythonRuntimeMounts(
        python_executable=python,
        runtime_paths=(),
        library_paths=(),
    )


def _run_state_with_status(
    *, sandbox_status: SandboxStatus, runtime_mounts: PythonRuntimeMounts | None
) -> EngineRunState:
    """Build a MagicMock EngineRunState fixed to the given sandbox status."""
    run_state = MagicMock(spec=EngineRunState)
    run_state.config = _engine_config()
    run_state.output_bus = EngineOutputBus()
    run_state.trace_store = MagicMock()
    run_state.sandbox_status = sandbox_status
    run_state.runtime_mounts = runtime_mounts
    return run_state


def test_run_code_registered_when_sandbox_available(tmp_path: Path) -> None:
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")
    status = SandboxStatus.ok(backend=SandboxBackend.LINUX_BWRAP_SYSTEM, executable=bwrap)
    run_state = _run_state_with_status(
        sandbox_status=status, runtime_mounts=_runtime_mounts(tmp_path)
    )

    tools = _child_tools_for_depth(
        depth=0,
        run_state=run_state,
        semaphores_by_depth=_semaphores(),
        parent_execution=_parent(),
        parent_context=_parent_context(),
    )

    assert "run_code" in {t.name for t in tools}


def test_run_code_omitted_when_sandbox_unavailable(tmp_path: Path) -> None:
    status = SandboxStatus.unavailable(
        reason=SandboxUnavailableReason.NAMESPACE_DENIED,
        diagnostic="Operation not permitted",
        remediation="run with --privileged",
    )
    run_state = _run_state_with_status(sandbox_status=status, runtime_mounts=None)

    tools = _child_tools_for_depth(
        depth=0,
        run_state=run_state,
        semaphores_by_depth=_semaphores(),
        parent_execution=_parent(),
        parent_context=_parent_context(),
    )

    names = {t.name for t in tools}
    assert "run_code" not in names
    # The other leaf tools must still be present.
    assert "get_dataset_overview" in names
    assert "synthesize_traces" in names


def test_run_code_omitted_when_runtime_mounts_missing(tmp_path: Path) -> None:
    """Defensive: an ``available`` status without runtime mounts must still skip ``run_code``."""
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")
    status = SandboxStatus.ok(backend=SandboxBackend.LINUX_BWRAP_SYSTEM, executable=bwrap)
    run_state = _run_state_with_status(sandbox_status=status, runtime_mounts=None)

    tools = _child_tools_for_depth(
        depth=0,
        run_state=run_state,
        semaphores_by_depth=_semaphores(),
        parent_execution=_parent(),
        parent_context=_parent_context(),
    )

    assert "run_code" not in {t.name for t in tools}
