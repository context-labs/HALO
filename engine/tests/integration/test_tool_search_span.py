"""Isolated integration test for ``search_span``.

Invokes the registered SDK ``FunctionTool`` against a real ``TraceStore`` and
asserts the ``SpanMatchRecord`` shape returned when scoping a regex to one span.
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
async def test_search_span_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    cfg = engine_config()
    store = await load_store(tmp_path, fixtures_dir)
    tools = wired_tools(
        cfg=cfg,
        store=store,
        agent_context=new_agent_context(cfg),
        parent_execution=root_execution(cfg),
    )

    raw = await tools["search_span"].on_invoke_tool(
        MagicMock(spec=SdkToolContext),
        '{"trace_id": "t-bbbb", "span_id": "s-bbbb-2", "regex_pattern": "tool failure"}',
    )
    payload = json.loads(raw)["result"]
    assert payload["trace_id"] == "t-bbbb"
    assert payload["span_id"] == "s-bbbb-2"
    assert payload["match_count"] == 1
    assert payload["returned_match_count"] == 1
    assert payload["has_more"] is False
    record = payload["matches"][0]
    assert record["span_id"] == "s-bbbb-2"
    assert record["match_text"] == "tool failure"


@pytest.mark.asyncio
async def test_search_span_max_matches_caps_records(tmp_path: Path, fixtures_dir: Path) -> None:
    cfg = engine_config()
    store = await load_store(tmp_path, fixtures_dir)
    tools = wired_tools(
        cfg=cfg,
        store=store,
        agent_context=new_agent_context(cfg),
        parent_execution=root_execution(cfg),
    )

    raw = await tools["search_span"].on_invoke_tool(
        MagicMock(spec=SdkToolContext),
        '{"trace_id": "t-bbbb", "span_id": "s-bbbb-2", "regex_pattern": "inference", "max_matches": 1}',
    )
    payload = json.loads(raw)["result"]
    assert payload["returned_match_count"] == 1
    assert payload["match_count"] > 1
    assert payload["has_more"] is True
