"""Isolated integration test for ``view_spans``.

Invokes the registered SDK ``FunctionTool`` against a real ``TraceStore`` and
asserts that the surgical-read path returns only the requested spans.
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
async def test_view_spans_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    cfg = engine_config()
    store = await load_store(tmp_path, fixtures_dir)
    tools = wired_tools(
        cfg=cfg,
        store=store,
        agent_context=new_agent_context(cfg),
        parent_execution=root_execution(cfg),
    )

    raw = await tools["view_spans"].on_invoke_tool(
        MagicMock(spec=SdkToolContext),
        '{"trace_id": "t-bbbb", "span_ids": ["s-bbbb-2"]}',
    )
    payload = json.loads(raw)["result"]
    assert payload["trace_id"] == "t-bbbb"
    assert [s["span_id"] for s in payload["spans"]] == ["s-bbbb-2"]
    assert payload["oversized"] is None
