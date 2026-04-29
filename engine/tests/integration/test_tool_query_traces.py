"""Isolated integration test for ``query_traces``.

Invokes the registered SDK ``FunctionTool`` against a real ``TraceStore`` and
asserts the exact paginated summary returned for the ``has_errors=True`` filter.
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
async def test_query_traces_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    cfg = engine_config()
    store = await load_store(tmp_path, fixtures_dir)
    tools = wired_tools(
        cfg=cfg,
        store=store,
        agent_context=new_agent_context(cfg),
        parent_execution=root_execution(cfg),
    )

    raw = await tools["query_traces"].on_invoke_tool(
        MagicMock(spec=SdkToolContext),
        '{"filters": {"has_errors": true}, "limit": 10, "offset": 0}',
    )
    payload = json.loads(raw)["result"]

    assert payload["total"] == 1
    assert len(payload["traces"]) == 1
    only = payload["traces"][0]
    assert only["trace_id"] == "t-bbbb"
    assert only["has_errors"] is True
    assert only["span_count"] == 2
    assert only["model_names"] == ["gpt-5.4"]
    assert only["total_input_tokens"] == 200
    assert only["total_output_tokens"] == 40
    assert only["raw_jsonl_bytes"] > 0
