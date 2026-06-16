"""Isolated integration tests for the code tools (``glob_files``/``grep_files``/``read_file``).

Invokes the registered SDK ``FunctionTool``s against a real ``CodeRepo`` over the
``tiny_repo`` fixture and asserts the exact JSON returned across the SDK boundary
(Pydantic parse on the way in, ``model_dump_json`` on the way out). Deterministic,
so no live LLM and no ``-m live`` marker — mirrors ``test_tool_query_traces.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from agents.tool_context import ToolContext as SdkToolContext

from engine.code.code_repo import CodeRepo
from tests.integration.tool_isolation_kit import (
    engine_config,
    load_store,
    new_agent_context,
    root_execution,
    wired_tools,
)


async def _code_tools(tmp_path: Path, fixtures_dir: Path) -> dict[str, object]:
    cfg = engine_config(maximum_depth=1)
    store = await load_store(tmp_path, fixtures_dir)
    repo = CodeRepo.open(fixtures_dir / "tiny_repo")
    return wired_tools(
        cfg=cfg,
        store=store,
        agent_context=new_agent_context(cfg),
        parent_execution=root_execution(cfg),
        code_repo=repo,
    )


@pytest.mark.asyncio
async def test_view_repo_tree_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    tools = await _code_tools(tmp_path, fixtures_dir)
    raw = await tools["view_repo_tree"].on_invoke_tool(MagicMock(spec=SdkToolContext), "{}")
    result = json.loads(raw)["result"]
    assert result["root"].endswith("tiny_repo")
    assert "agent/" in result["tree"]
    assert "config.py" in result["tree"]


@pytest.mark.asyncio
async def test_glob_files_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    tools = await _code_tools(tmp_path, fixtures_dir)
    raw = await tools["glob_files"].on_invoke_tool(
        MagicMock(spec=SdkToolContext), '{"pattern": "**/*.py"}'
    )
    result = json.loads(raw)["result"]
    assert [f["path"] for f in result["files"]] == ["agent/config.py", "agent/runner.py"]
    assert result["has_more"] is False


@pytest.mark.asyncio
async def test_grep_files_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    tools = await _code_tools(tmp_path, fixtures_dir)
    raw = await tools["grep_files"].on_invoke_tool(
        MagicMock(spec=SdkToolContext), '{"regex_pattern": "MAX_RETRIES"}'
    )
    result = json.loads(raw)["result"]
    hits = {(m["path"], m["line_number"]) for m in result["matches"]}
    # The constant is defined and referenced in config.py.
    assert ("agent/config.py", 1) in hits


@pytest.mark.asyncio
async def test_read_file_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    tools = await _code_tools(tmp_path, fixtures_dir)
    raw = await tools["read_file"].on_invoke_tool(
        MagicMock(spec=SdkToolContext), '{"path": "agent/config.py", "offset": 1, "limit": 2}'
    )
    result = json.loads(raw)["result"]
    assert result["content"] == "     1\tMAX_RETRIES = 3\n     2\tTIMEOUT_SECONDS = 30"
    assert result["start_line"] == 1
    assert result["end_line"] == 2
    assert result["total_line_count"] == 7


@pytest.mark.asyncio
async def test_read_file_confinement_error_through_sdk_adapter(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    tools = await _code_tools(tmp_path, fixtures_dir)
    # An escaping path raises ValueError from the tool; in a real run the SDK's
    # tool dispatcher converts that into a model-visible error result. Invoked
    # directly here, the exception propagates unchanged.
    with pytest.raises(ValueError, match="outside the repo root"):
        await tools["read_file"].on_invoke_tool(
            MagicMock(spec=SdkToolContext), '{"path": "../../../etc/hosts"}'
        )
