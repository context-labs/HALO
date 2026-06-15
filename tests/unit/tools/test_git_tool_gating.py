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
from engine.engine_config import EngineConfig
from engine.git.git_repo import GitRepo
from engine.model_config import ModelConfig
from engine.tools.subagent_tool_factory import _child_tools_for_depth
from tests.unit.git.git_fixture import build_git_repo

_GIT_TOOL_NAMES = {"git_log", "git_show", "git_diff", "git_blame", "git_read_file"}


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


def _git_repo(tmp_path: Path) -> GitRepo:
    repo = GitRepo.open(build_git_repo(tmp_path))
    assert repo is not None
    return repo


def _run_state(*, git_repo: GitRepo | None, maximum_depth: int = 2) -> EngineRunState:
    run_state = MagicMock(spec=EngineRunState)
    run_state.config = _engine_config(maximum_depth=maximum_depth)
    run_state.output_bus = EngineOutputBus()
    run_state.trace_store = MagicMock()
    run_state.sandbox = None
    run_state.code_repo = None
    run_state.git_repo = git_repo
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


def test_git_tools_registered_when_repo_configured(tmp_path: Path) -> None:
    run_state = _run_state(git_repo=_git_repo(tmp_path))
    assert _GIT_TOOL_NAMES <= _tool_names(run_state=run_state, depth=0)


def test_git_tools_omitted_when_no_repo() -> None:
    run_state = _run_state(git_repo=None)
    names = _tool_names(run_state=run_state, depth=0)
    assert names.isdisjoint(_GIT_TOOL_NAMES)
    # Trace tools are unaffected.
    assert "get_dataset_overview" in names


def test_git_tools_present_at_max_depth(tmp_path: Path) -> None:
    run_state = _run_state(git_repo=_git_repo(tmp_path), maximum_depth=2)
    names = _tool_names(run_state=run_state, depth=2, maximum_depth=2)
    assert _GIT_TOOL_NAMES <= names
    # call_subagent is gated out at max depth, but the git tools are not.
    assert "call_subagent" not in names


def test_git_tools_present_when_max_depth_zero(tmp_path: Path) -> None:
    run_state = _run_state(git_repo=_git_repo(tmp_path), maximum_depth=0)
    names = _tool_names(run_state=run_state, depth=0, maximum_depth=0)
    assert _GIT_TOOL_NAMES <= names
    assert "call_subagent" not in names
