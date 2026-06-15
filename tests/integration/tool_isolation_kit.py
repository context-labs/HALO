"""Shared wiring helpers for the per-tool isolation tests.

Each ``test_<tool_name>_isolation.py`` calls ``wired_tools(...)`` to get the
production-shaped ``FunctionTool`` for the tool under test (built by
``_child_tools_for_depth``'s real ``make_ctx`` factory) and then invokes
``on_invoke_tool`` directly with raw JSON arguments. That exercises the full
SDK boundary — Pydantic schema parse on the way in, ``model_dump_json`` on the
way out — without spinning up a real agent loop.

To add a new tool: add an isolation test file calling these helpers, and
extend ``EXPECTED_TOOL_NAMES_WITH_SANDBOX`` in ``test_tool_inventory.py``.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
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
from engine.git.git_repo import _REPO_REDIRECT_GIT_ENV, GitRepo
from engine.model_config import ModelConfig
from engine.sandbox.sandbox import Sandbox
from engine.tools.subagent_tool_factory import _child_tools_for_depth
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore

LIVE_MODEL = os.environ.get("HALO_INTEGRATION_MODEL", "gpt-5.4-mini")
LIVE_TIMEOUT_SECONDS = float(os.environ.get("HALO_INTEGRATION_TIMEOUT", "60"))


def engine_config(*, maximum_depth: int = 1) -> EngineConfig:
    """Build a minimal ``EngineConfig`` aimed at the live integration model.

    ``maximum_depth=1`` is the default so ``call_subagent`` shows up in the
    depth-0 tool list. Bump it explicitly when a test needs grandchildren.
    """
    agent = AgentConfig(
        name="root",
        model=ModelConfig(name=LIVE_MODEL),
        maximum_turns=4,
    )
    return EngineConfig(
        root_agent=agent,
        subagent=agent.model_copy(update={"name": "sub", "maximum_turns": 3}),
        synthesis_model=ModelConfig(name=LIVE_MODEL),
        compaction_model=ModelConfig(name=LIVE_MODEL),
        maximum_depth=maximum_depth,
        maximum_parallel_subagents=1,
    )


def new_agent_context(cfg: EngineConfig) -> AgentContext:
    """Empty context bound to ``cfg``'s compaction settings — caller appends items as needed."""
    return AgentContext(
        items=[],
        compaction_model=cfg.compaction_model,
        text_message_compaction_keep_last_messages=cfg.text_message_compaction_keep_last_messages,
        tool_call_compaction_keep_last_turns=cfg.tool_call_compaction_keep_last_turns,
    )


def root_execution(cfg: EngineConfig) -> AgentExecution:
    """Synthetic depth-0 ``AgentExecution`` — stands in for the root parent of a real run."""
    return AgentExecution(
        agent_id="root-1",
        agent_name=cfg.root_agent.name,
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )


async def load_store(tmp_path: Path, fixtures_dir: Path) -> TraceStore:
    """Copy ``tiny_traces.jsonl`` into ``tmp_path``, build its index, return a loaded ``TraceStore``."""
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    return TraceStore.load(trace_path=trace_path, index_path=index_path)


def wired_tools(
    *,
    cfg: EngineConfig,
    store: TraceStore,
    agent_context: AgentContext,
    parent_execution: AgentExecution,
    sandbox: Sandbox | None = None,
    code_repo: CodeRepo | None = None,
    git_repo: GitRepo | None = None,
) -> dict[str, object]:
    """Build the production tool list for a depth-0 agent, indexed by tool name.

    Pass ``sandbox=`` to register ``run_code`` (use ``MagicMock(spec=Sandbox)``
    when the test does not actually invoke ``run_code``, or a real resolved
    ``Sandbox`` when it does). Pass ``code_repo=`` to register the code tools
    (``glob_files``/``grep_files``/``read_file``); pass ``git_repo=`` to register
    the git tools (``git_log``/``git_show``/``git_diff``/``git_blame``/
    ``git_read_file``).
    """
    run_state = EngineRunState(
        trace_store=store,
        output_bus=EngineOutputBus(),
        config=cfg,
        sandbox=sandbox,
        code_repo=code_repo,
        git_repo=git_repo,
        openai_client=AsyncOpenAI(
            base_url=cfg.model_provider.base_url,
            api_key=cfg.model_provider.api_key,
            default_headers=cfg.model_provider.default_headers,
        ),
    )
    run_state.register(parent_execution)
    semaphores = {d: asyncio.Semaphore(1) for d in range(1, cfg.maximum_depth + 1)}
    tools = _child_tools_for_depth(
        depth=0,
        run_state=run_state,
        semaphores_by_depth=semaphores,
        parent_execution=parent_execution,
        parent_context=agent_context,
    )
    return {t.name: t for t in tools}


def fake_sandbox() -> Sandbox:
    """A no-op ``Sandbox`` stand-in for tests that only need ``run_code`` to be *registered*.

    Tests that actually invoke ``run_code`` must use ``Sandbox.get()`` instead.
    """
    return MagicMock(spec=Sandbox)


def git_init_repo(tmp_path: Path, fixtures_dir: Path) -> GitRepo:
    """Copy ``tiny_repo`` into ``tmp_path``, git-init it with one commit, return an open ``GitRepo``.

    Identity and dates are pinned and global/system git config is bypassed so
    the result is independent of the host's git configuration.
    """
    root = tmp_path / "git_repo"
    shutil.copytree(fixtures_dir / "tiny_repo", root)
    # Strip ambient repo-redirect vars (e.g. GIT_DIR from a git pre-push hook) so
    # the build targets the tmp repo, not HALO's own.
    env = {k: v for k, v in os.environ.items() if k not in _REPO_REDIRECT_GIT_ENV}
    env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_AUTHOR_NAME": "Test Author",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test Author",
            "GIT_COMMITTER_EMAIL": "test@example.com",
            "GIT_AUTHOR_DATE": "2024-01-01T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2024-01-01T00:00:00+00:00",
        }
    )
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(root)],
        check=True,
        capture_output=True,
        env=env,
    )
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True, capture_output=True, env=env)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-q", "-m", "Import tiny_repo"],
        check=True,
        capture_output=True,
        env=env,
    )
    repo = GitRepo.open(root)
    assert repo is not None
    return repo
