"""Isolated integration test for ``synthesize_traces`` (live LLM).

Invokes the registered SDK ``FunctionTool`` against a real ``TraceStore``,
which forwards the rendered trace text to the synthesis model. Skips when
``OPENAI_API_KEY`` is not set; in CI, ``task test:integration`` injects it
via Infisical.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from agents.tool_context import ToolContext as SdkToolContext

from tests.integration.tool_isolation_kit import (
    LIVE_TIMEOUT_SECONDS,
    engine_config,
    load_store,
    new_agent_context,
    root_execution,
    wired_tools,
)


@pytest.mark.live
@pytest.mark.asyncio
async def test_synthesize_traces_through_sdk_adapter_live(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; live synthesize_traces requires real LLM access")

    cfg = engine_config()
    store = await load_store(tmp_path, fixtures_dir)
    tools = wired_tools(
        cfg=cfg,
        store=store,
        agent_context=new_agent_context(cfg),
        parent_execution=root_execution(cfg),
    )

    raw = await asyncio.wait_for(
        tools["synthesize_traces"].on_invoke_tool(
            MagicMock(spec=SdkToolContext),
            '{"trace_ids": ["t-bbbb"], "focus": "errors"}',
        ),
        LIVE_TIMEOUT_SECONDS,
    )
    payload = json.loads(raw)
    assert isinstance(payload["summary"], str)
    assert payload["summary"].strip(), "live synthesize_traces returned empty summary"
