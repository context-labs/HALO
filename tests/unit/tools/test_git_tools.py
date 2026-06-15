from __future__ import annotations

from pathlib import Path

import pytest

from engine.git.git_repo import GitRepo
from engine.git.models import (
    GitBlameArguments,
    GitDiffArguments,
    GitLogArguments,
    GitReadFileArguments,
    GitShowArguments,
)
from engine.tools.git_tools import (
    GitBlameTool,
    GitDiffTool,
    GitLogTool,
    GitReadFileTool,
    GitShowTool,
)
from engine.tools.tool_protocol import ToolContext
from tests.unit.git.git_fixture import (
    COMMIT_1_SUBJECT,
    COMMIT_2_SUBJECT,
    COMMIT_3_SUBJECT,
    PICKAXE_TOKEN,
    build_git_repo,
)


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    repo = GitRepo.open(build_git_repo(tmp_path))
    assert repo is not None
    return ToolContext.model_construct(git_repo=repo)


def _short_sha(ctx: ToolContext, subject: str) -> str:
    assert ctx.git_repo is not None
    commits = ctx.git_repo.log(
        max_commits=50,
        since=None,
        until=None,
        ref_range=None,
        path=None,
        pickaxe_string=None,
        pickaxe_regex=None,
    ).commits
    return {c.subject: c.short_sha for c in commits}[subject]


# --- happy paths -------------------------------------------------------------


@pytest.mark.asyncio
async def test_git_log_tool(ctx: ToolContext) -> None:
    tool = GitLogTool()
    result = await tool.run(ctx, GitLogArguments())
    assert [c.subject for c in result.result.commits] == [
        COMMIT_3_SUBJECT,
        COMMIT_2_SUBJECT,
        COMMIT_1_SUBJECT,
    ]
    assert result.result.has_more is False


@pytest.mark.asyncio
async def test_git_log_tool_pickaxe(ctx: ToolContext) -> None:
    tool = GitLogTool()
    result = await tool.run(ctx, GitLogArguments(pickaxe_string=PICKAXE_TOKEN))
    assert [c.subject for c in result.result.commits] == [COMMIT_2_SUBJECT]


@pytest.mark.asyncio
async def test_git_show_tool(ctx: ToolContext) -> None:
    tool = GitShowTool()
    sha = _short_sha(ctx, COMMIT_2_SUBJECT)
    result = await tool.run(ctx, GitShowArguments(ref=sha, include_patch=True))
    assert result.result.commit.subject == COMMIT_2_SUBJECT
    assert f"+{PICKAXE_TOKEN} = 1" in result.result.body


@pytest.mark.asyncio
async def test_git_diff_tool(ctx: ToolContext) -> None:
    tool = GitDiffTool()
    first = _short_sha(ctx, COMMIT_1_SUBJECT)
    last = _short_sha(ctx, COMMIT_3_SUBJECT)
    result = await tool.run(ctx, GitDiffArguments(from_ref=first, to_ref=last, stat_only=True))
    assert result.result.stat_only is True
    assert "config.py" in result.result.diff
    assert "runner.py" in result.result.diff


@pytest.mark.asyncio
async def test_git_blame_tool(ctx: ToolContext) -> None:
    tool = GitBlameTool()
    result = await tool.run(ctx, GitBlameArguments(path="config.py", start_line=1, end_line=3))
    assert result.result.returned_count == 3
    assert result.result.lines[0].line_text == "MAX_RETRIES = 3"
    assert result.result.lines[2].summary == COMMIT_2_SUBJECT


@pytest.mark.asyncio
async def test_git_read_file_tool(ctx: ToolContext) -> None:
    tool = GitReadFileTool()
    first = _short_sha(ctx, COMMIT_1_SUBJECT)
    result = await tool.run(ctx, GitReadFileArguments(ref=first, path="config.py"))
    assert result.result.content == "     1\tMAX_RETRIES = 3\n     2\tTIMEOUT_SECONDS = 30"
    assert result.result.total_line_count == 2


# --- require_git_repo --------------------------------------------------------


@pytest.mark.asyncio
async def test_git_log_requires_git_repo() -> None:
    tool = GitLogTool()
    with pytest.raises(RuntimeError, match="ToolContext.git_repo required"):
        await tool.run(ToolContext.model_construct(), GitLogArguments())


@pytest.mark.asyncio
async def test_git_show_requires_git_repo() -> None:
    tool = GitShowTool()
    with pytest.raises(RuntimeError, match="ToolContext.git_repo required"):
        await tool.run(ToolContext.model_construct(), GitShowArguments(ref="HEAD"))


@pytest.mark.asyncio
async def test_git_diff_requires_git_repo() -> None:
    tool = GitDiffTool()
    with pytest.raises(RuntimeError, match="ToolContext.git_repo required"):
        await tool.run(ToolContext.model_construct(), GitDiffArguments(from_ref="a", to_ref="b"))


@pytest.mark.asyncio
async def test_git_blame_requires_git_repo() -> None:
    tool = GitBlameTool()
    with pytest.raises(RuntimeError, match="ToolContext.git_repo required"):
        await tool.run(
            ToolContext.model_construct(),
            GitBlameArguments(path="x.py", start_line=1, end_line=1),
        )


@pytest.mark.asyncio
async def test_git_read_file_requires_git_repo() -> None:
    tool = GitReadFileTool()
    with pytest.raises(RuntimeError, match="ToolContext.git_repo required"):
        await tool.run(ToolContext.model_construct(), GitReadFileArguments(ref="HEAD", path="x.py"))


# --- argument-model validators -----------------------------------------------


def test_git_log_rejects_both_pickaxes() -> None:
    with pytest.raises(ValueError, match="at most one of pickaxe"):
        GitLogArguments(pickaxe_string="a", pickaxe_regex="b")


def test_git_blame_rejects_end_before_start() -> None:
    with pytest.raises(ValueError, match="end_line must be >= start_line"):
        GitBlameArguments(path="x.py", start_line=5, end_line=2)
