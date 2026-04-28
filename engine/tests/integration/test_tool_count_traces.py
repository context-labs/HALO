"""Isolated integration test for ``count_traces``.

Invokes the registered SDK ``FunctionTool`` against a real ``TraceStore`` and
asserts the exact count returned for a model-name filter.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from agents.tool_context import ToolContext as SdkToolContext

from tests.integration.tool_isolation_kit import (
    engine_config,
    load_store,
    new_agent_context,
    root_execution,
    wired_tools,
)


@pytest.mark.asyncio
async def test_count_traces_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    cfg = engine_config()
    store = await load_store(tmp_path, fixtures_dir)
    tools = wired_tools(
        cfg=cfg,
        store=store,
        agent_context=new_agent_context(cfg),
        parent_execution=root_execution(cfg),
    )

    raw = await tools["count_traces"].on_invoke_tool(
        MagicMock(spec=SdkToolContext),
        '{"filters": {"model_names": ["gpt-5.4"]}}',
    )
    payload = json.loads(raw)["result"]
    assert payload["total"] == 1
