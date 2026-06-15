"""Isolated integration tests for the git tools (``git_log``/``git_show``/etc.).

Invokes the registered SDK ``FunctionTool``s against a real ``GitRepo`` (a
git-init'd copy of the ``tiny_repo`` fixture) and asserts the JSON returned
across the SDK boundary (Pydantic parse on the way in, ``model_dump_json`` on
the way out). Deterministic, so no live LLM and no ``-m live`` marker — mirrors
``test_tool_code_tools.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from agents.tool_context import ToolContext as SdkToolContext

from tests.integration.tool_isolation_kit import (
    engine_config,
    git_init_repo,
    load_store,
    new_agent_context,
    root_execution,
    wired_tools,
)


async def _git_tools(tmp_path: Path, fixtures_dir: Path) -> dict[str, object]:
    cfg = engine_config(maximum_depth=1)
    store = await load_store(tmp_path, fixtures_dir)
    repo = git_init_repo(tmp_path, fixtures_dir)
    return wired_tools(
        cfg=cfg,
        store=store,
        agent_context=new_agent_context(cfg),
        parent_execution=root_execution(cfg),
        git_repo=repo,
    )


@pytest.mark.asyncio
async def test_git_log_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    tools = await _git_tools(tmp_path, fixtures_dir)
    raw = await tools["git_log"].on_invoke_tool(MagicMock(spec=SdkToolContext), "{}")
    result = json.loads(raw)["result"]
    assert result["returned_count"] == 1
    assert result["has_more"] is False
    assert result["commits"][0]["subject"] == "Import tiny_repo"
    assert result["commits"][0]["author"] == "Test Author"


@pytest.mark.asyncio
async def test_git_show_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    tools = await _git_tools(tmp_path, fixtures_dir)
    raw = await tools["git_show"].on_invoke_tool(MagicMock(spec=SdkToolContext), '{"ref": "HEAD"}')
    result = json.loads(raw)["result"]
    assert result["commit"]["subject"] == "Import tiny_repo"
    assert "config.py" in result["body"]


@pytest.mark.asyncio
async def test_git_read_file_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    tools = await _git_tools(tmp_path, fixtures_dir)
    raw = await tools["git_read_file"].on_invoke_tool(
        MagicMock(spec=SdkToolContext),
        '{"ref": "HEAD", "path": "agent/config.py", "offset": 1, "limit": 2}',
    )
    result = json.loads(raw)["result"]
    assert result["content"] == "     1\tMAX_RETRIES = 3\n     2\tTIMEOUT_SECONDS = 30"
    assert result["total_line_count"] == 7


@pytest.mark.asyncio
async def test_git_blame_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    tools = await _git_tools(tmp_path, fixtures_dir)
    raw = await tools["git_blame"].on_invoke_tool(
        MagicMock(spec=SdkToolContext),
        '{"path": "agent/config.py", "start_line": 1, "end_line": 1}',
    )
    result = json.loads(raw)["result"]
    assert result["returned_count"] == 1
    assert result["lines"][0]["line_number"] == 1
    assert result["lines"][0]["line_text"] == "MAX_RETRIES = 3"
    assert result["lines"][0]["summary"] == "Import tiny_repo"


@pytest.mark.asyncio
async def test_git_read_file_confinement_error_through_sdk_adapter(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    tools = await _git_tools(tmp_path, fixtures_dir)
    with pytest.raises(ValueError, match="outside the repo root"):
        await tools["git_read_file"].on_invoke_tool(
            MagicMock(spec=SdkToolContext), '{"ref": "HEAD", "path": "../../../etc/hosts"}'
        )
