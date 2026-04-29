"""Isolated integration test for ``get_context_item``.

Verifies that ``_child_tools_for_depth`` plumbs the calling agent's
``AgentContext`` into the per-invocation ``ToolContext`` so the registered
tool can read the agent's own stored items by id — including original content
and compaction summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from agents.tool_context import ToolContext as SdkToolContext

from engine.agents.agent_context_items import AgentContextItem
from tests.integration.tool_isolation_kit import (
    engine_config,
    load_store,
    new_agent_context,
    root_execution,
    wired_tools,
)


@pytest.mark.asyncio
async def test_get_context_item_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    cfg = engine_config()
    store = await load_store(tmp_path, fixtures_dir)
    parent_context = new_agent_context(cfg)
    parent_context.append(
        AgentContextItem(
            item_id="ctx-42",
            role="user",
            content="stored content",
            is_compacted=True,
            compaction_summary="user said hi",
        )
    )

    tools = wired_tools(
        cfg=cfg,
        store=store,
        agent_context=parent_context,
        parent_execution=root_execution(cfg),
    )

    raw = await tools["get_context_item"].on_invoke_tool(
        MagicMock(spec=SdkToolContext), '{"item_id": "ctx-42"}'
    )
    payload = json.loads(raw)["item"]
    assert payload["item_id"] == "ctx-42"
    assert payload["role"] == "user"
    assert payload["content"] == "stored content"
    assert payload["is_compacted"] is True
    assert payload["compaction_summary"] == "user said hi"
