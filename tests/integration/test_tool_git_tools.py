"""Isolated integration tests for the git tools (``git_log``/``git_show``/etc.).

Invokes the registered SDK ``FunctionTool``s against a real ``GitRepo`` (a
git-init'd copy of the ``tiny_repo`` fixture, one pinned commit) and asserts the
exact JSON returned across the SDK boundary (Pydantic parse on the way in,
``model_dump_json`` on the way out). Deterministic, so no live LLM and no
``-m live`` marker — mirrors ``test_tool_code_tools.py``.
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

# The single commit ``git_init_repo`` makes (pinned identity + UTC date + the
# fixed ``tiny_repo`` contents), so its sha is deterministic across machines.
_COMMIT = {
    "full_sha": "ada076fe2628bb194239b947747326aeaf19af04",
    "short_sha": "ada076fe2628",
    "author": "Test Author",
    "authored_at": "2024-01-01T00:00:00+00:00",
    "subject": "Import tiny_repo",
}


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
    assert json.loads(raw)["result"] == {
        "commits": [_COMMIT],
        "returned_count": 1,
        "has_more": False,
    }


@pytest.mark.asyncio
async def test_git_show_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    tools = await _git_tools(tmp_path, fixtures_dir)
    raw = await tools["git_show"].on_invoke_tool(MagicMock(spec=SdkToolContext), '{"ref": "HEAD"}')
    assert json.loads(raw)["result"] == {
        "commit": _COMMIT,
        "body": (
            " README.md       |  3 +++\n"
            " agent/config.py |  7 +++++++\n"
            " agent/runner.py | 16 ++++++++++++++++\n"
            " 3 files changed, 26 insertions(+)"
        ),
        "truncated": False,
    }


@pytest.mark.asyncio
async def test_git_read_file_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    tools = await _git_tools(tmp_path, fixtures_dir)
    raw = await tools["git_read_file"].on_invoke_tool(
        MagicMock(spec=SdkToolContext),
        '{"ref": "HEAD", "path": "agent/config.py", "offset": 1, "limit": 2}',
    )
    assert json.loads(raw)["result"] == {
        "path": "agent/config.py",
        "content": "     1\tMAX_RETRIES = 3\n     2\tTIMEOUT_SECONDS = 30",
        "start_line": 1,
        "end_line": 2,
        "total_line_count": 7,
        "truncated": False,
    }


@pytest.mark.asyncio
async def test_git_blame_through_sdk_adapter(tmp_path: Path, fixtures_dir: Path) -> None:
    tools = await _git_tools(tmp_path, fixtures_dir)
    raw = await tools["git_blame"].on_invoke_tool(
        MagicMock(spec=SdkToolContext),
        '{"path": "agent/config.py", "start_line": 1, "end_line": 1}',
    )
    assert json.loads(raw)["result"] == {
        "path": "agent/config.py",
        "lines": [
            {
                "line_number": 1,
                "short_sha": "ada076fe2628",
                "author": "Test Author",
                "summary": "Import tiny_repo",
                "line_text": "MAX_RETRIES = 3",
            }
        ],
        "returned_count": 1,
        "truncated": False,
    }


@pytest.mark.asyncio
async def test_git_read_file_confinement_error_through_sdk_adapter(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    tools = await _git_tools(tmp_path, fixtures_dir)
    with pytest.raises(ValueError, match="outside the repo root"):
        await tools["git_read_file"].on_invoke_tool(
            MagicMock(spec=SdkToolContext), '{"ref": "HEAD", "path": "../../../etc/hosts"}'
        )
