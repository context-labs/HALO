from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from openai import AsyncOpenAI

from engine.agents.agent_config import AgentConfig
from engine.agents.agent_context import AgentContext
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.engine_run_state import EngineRunState
from engine.code.code_repo import CodeRepo
from engine.engine_config import EngineConfig
from engine.model_config import ModelConfig
from engine.tools.subagent_tool_factory import _child_tools_for_depth

_CODE_TOOL_NAMES = {"view_repo_tree", "glob_files", "grep_files", "read_file"}


def _engine_config(*, maximum_depth: int = 2) -> EngineConfig:
    agent = AgentConfig(
        name="root",
        model=ModelConfig(name="claude-sonnet-4-5"),
        maximum_turns=10,
    )
    return EngineConfig(
        root_agent=agent,
        subagent=agent,
        synthesis_model=ModelConfig(name="claude-haiku-4-5"),
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        maximum_depth=maximum_depth,
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


def _semaphores(maximum_depth: int) -> dict[int, asyncio.Semaphore]:
    return {depth: asyncio.Semaphore(1) for depth in range(1, maximum_depth + 1)}


def _code_repo(tmp_path: Path) -> CodeRepo:
    (tmp_path / "main.py").write_text("x = 1\n")
    return CodeRepo.open(tmp_path)


def _run_state(*, code_repo: CodeRepo | None, maximum_depth: int = 2) -> EngineRunState:
    run_state = MagicMock(spec=EngineRunState)
    run_state.config = _engine_config(maximum_depth=maximum_depth)
    run_state.output_bus = EngineOutputBus()
    run_state.trace_store = MagicMock()
    run_state.sandbox = None
    run_state.code_repo = code_repo
    run_state.openai_client = AsyncOpenAI(api_key="test")
    return run_state


def _tool_names(*, run_state: EngineRunState, depth: int, maximum_depth: int = 2) -> set[str]:
    tools = _child_tools_for_depth(
        depth=depth,
        run_state=run_state,
        semaphores_by_depth=_semaphores(maximum_depth),
        parent_execution=_parent(),
        parent_context=_parent_context(),
    )
    return {t.name for t in tools}


def test_code_tools_registered_when_repo_configured(tmp_path: Path) -> None:
    run_state = _run_state(code_repo=_code_repo(tmp_path))
    assert _CODE_TOOL_NAMES <= _tool_names(run_state=run_state, depth=0)


def test_code_tools_omitted_when_no_repo() -> None:
    run_state = _run_state(code_repo=None)
    names = _tool_names(run_state=run_state, depth=0)
    assert names.isdisjoint(_CODE_TOOL_NAMES)
    # Trace tools are unaffected.
    assert "get_dataset_overview" in names


def test_code_tools_present_at_max_depth(tmp_path: Path) -> None:
    run_state = _run_state(code_repo=_code_repo(tmp_path), maximum_depth=2)
    names = _tool_names(run_state=run_state, depth=2, maximum_depth=2)
    assert _CODE_TOOL_NAMES <= names
    # call_subagent is gated out at max depth, but the code tools are not.
    assert "call_subagent" not in names


def test_code_tools_present_when_max_depth_zero(tmp_path: Path) -> None:
    run_state = _run_state(code_repo=_code_repo(tmp_path), maximum_depth=0)
    names = _tool_names(run_state=run_state, depth=0, maximum_depth=0)
    assert _CODE_TOOL_NAMES <= names
    assert "call_subagent" not in names
