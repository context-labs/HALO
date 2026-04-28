"""Isolated integration test for ``call_subagent`` (live LLM).

End-to-end live invocation of the registered SDK ``FunctionTool``: the parent's
``call_subagent`` dispatches to a real subagent loop driven by the OpenAI Agents
SDK, which calls a real LLM and at least one trace tool, then returns a
``SubagentToolResult`` to the parent. Skips when ``OPENAI_API_KEY`` is not set;
in CI ``task test:integration`` injects it via Infisical.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from agents.tool_context import ToolContext as SdkToolContext

from engine.tools.subagent_result import SubagentToolResult
from tests.integration.tool_isolation_kit import (
    LIVE_TIMEOUT_SECONDS,
    engine_config,
    load_store,
    new_agent_context,
    root_execution,
    wired_tools,
)


@pytest.mark.asyncio
async def test_call_subagent_through_sdk_adapter_live(tmp_path: Path, fixtures_dir: Path) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; live call_subagent requires real LLM access")

    cfg = engine_config(maximum_depth=1)
    store = await load_store(tmp_path, fixtures_dir)
    tools = wired_tools(
        cfg=cfg,
        store=store,
        agent_context=new_agent_context(cfg),
        parent_execution=root_execution(cfg),
    )

    sdk_ctx = SdkToolContext(
        context=None,
        tool_name="call_subagent",
        tool_call_id="parent-call-1",
        tool_arguments="{}",
    )
    raw_args = json.dumps(
        {
            "input": (
                "Use a tool to count how many traces in the dataset have errors. "
                "Reply with just the number."
            )
        }
    )

    async with asyncio.timeout(LIVE_TIMEOUT_SECONDS):
        raw = await tools["call_subagent"].on_invoke_tool(sdk_ctx, raw_args)

    result = SubagentToolResult.model_validate_json(raw)
    assert result.answer.strip(), "subagent returned empty answer"
    assert result.turns_used >= 1
    assert result.tool_calls_made >= 1, "expected the subagent to invoke at least one tool"
