"""Inventory guard for the per-tool isolation suite.

Asserts that the set of tools registered by ``_child_tools_for_depth`` exactly
matches the set we have isolation tests for. When this fails after adding a
new tool: add a ``test_<tool_name>_isolation.py`` calling into
``tool_isolation_kit`` and update the expected set below.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.tool_isolation_kit import (
    engine_config,
    fake_sandbox,
    load_store,
    new_agent_context,
    root_execution,
    wired_tools,
)

EXPECTED_TOOL_NAMES = {
    "get_dataset_overview",
    "query_traces",
    "count_traces",
    "view_trace",
    "view_spans",
    "search_trace",
    "search_span",
    "get_context_item",
    "synthesize_traces",
    "run_code",
    "call_subagent",
}


@pytest.mark.asyncio
async def test_registered_tools_match_isolation_suite(tmp_path: Path, fixtures_dir: Path) -> None:
    cfg = engine_config(maximum_depth=1)
    store = await load_store(tmp_path, fixtures_dir)
    tools = wired_tools(
        cfg=cfg,
        store=store,
        agent_context=new_agent_context(cfg),
        parent_execution=root_execution(cfg),
        sandbox=fake_sandbox(),
    )
    assert set(tools.keys()) == EXPECTED_TOOL_NAMES
