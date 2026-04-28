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
        instructions="Answer briefly.",
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
) -> dict[str, object]:
    """Build the production tool list for a depth-0 agent, indexed by tool name.

    Pass ``sandbox=`` to register ``run_code`` (use ``MagicMock(spec=Sandbox)``
    when the test does not actually invoke ``run_code``, or a real resolved
    ``Sandbox`` when it does).
    """
    run_state = EngineRunState(
        trace_store=store,
        output_bus=EngineOutputBus(),
        config=cfg,
        sandbox=sandbox,
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

    Tests that actually invoke ``run_code`` must use ``resolve_sandbox`` instead.
    """
    return MagicMock(spec=Sandbox)
