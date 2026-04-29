"""Isolated integration test for ``get_dataset_overview``.

Invokes the registered SDK ``FunctionTool`` (not ``tool.run`` directly) so the
JSON-arg parse and serialization at the SDK boundary are exercised end-to-end
against a real ``TraceStore`` loaded from ``tiny_traces.jsonl``.
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
async def test_get_dataset_overview_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    cfg = engine_config()
    store = await load_store(tmp_path, fixtures_dir)
    tools = wired_tools(
        cfg=cfg,
        store=store,
        agent_context=new_agent_context(cfg),
        parent_execution=root_execution(cfg),
    )

    raw = await tools["get_dataset_overview"].on_invoke_tool(
        MagicMock(spec=SdkToolContext), '{"filters": {}}'
    )
    payload = json.loads(raw)["result"]

    assert payload["total_traces"] == 3
    assert payload["total_spans"] == 6
    assert payload["error_trace_count"] == 1
    assert payload["service_names"] == ["agent-a", "agent-b"]
    assert payload["model_names"] == [
        "claude-haiku-4-5",
        "claude-sonnet-4-5",
        "gpt-5.4",
    ]
    assert payload["agent_names"] == ["agent-a", "agent-b"]
    assert payload["total_input_tokens"] == 330
    assert payload["total_output_tokens"] == 100
    assert payload["raw_jsonl_bytes"] > 0
