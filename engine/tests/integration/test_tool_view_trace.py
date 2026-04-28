"""Isolated integration test for ``view_trace``.

Invokes the registered SDK ``FunctionTool`` against a real ``TraceStore`` and
asserts the exact span ids and statuses returned for the error trace.
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
async def test_view_trace_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    cfg = engine_config()
    store = await load_store(tmp_path, fixtures_dir)
    tools = wired_tools(
        cfg=cfg,
        store=store,
        agent_context=new_agent_context(cfg),
        parent_execution=root_execution(cfg),
    )

    raw = await tools["view_trace"].on_invoke_tool(
        MagicMock(spec=SdkToolContext), '{"trace_id": "t-bbbb"}'
    )
    payload = json.loads(raw)["result"]

    assert payload["trace_id"] == "t-bbbb"
    assert [s["span_id"] for s in payload["spans"]] == ["s-bbbb-1", "s-bbbb-2"]
    assert [s["status"]["code"] for s in payload["spans"]] == [
        "STATUS_CODE_ERROR",
        "STATUS_CODE_ERROR",
    ]
