from __future__ import annotations

from pathlib import Path

import pytest

from engine.code.models import FileContent
from engine.git.git_repo import GitRepo
from engine.git.models import (
    BlameLine,
    GitBlame,
    GitBlameArguments,
    GitBlameResult,
    GitDiff,
    GitDiffArguments,
    GitDiffResult,
    GitLog,
    GitLogArguments,
    GitLogResult,
    GitReadFileArguments,
    GitReadFileResult,
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
    AUTHOR_NAME,
    COMMIT_1,
    COMMIT_2,
    COMMIT_3,
    PICKAXE_TOKEN,
    build_git_repo,
)


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    repo = GitRepo.open(build_git_repo(tmp_path))
    assert repo is not None
    return ToolContext.model_construct(git_repo=repo)


# --- happy paths (assert the whole envelope) ---------------------------------


@pytest.mark.asyncio
async def test_git_log_tool(ctx: ToolContext) -> None:
    result = await GitLogTool().run(ctx, GitLogArguments())
    assert result == GitLogResult(
        result=GitLog(commits=[COMMIT_3, COMMIT_2, COMMIT_1], returned_count=3, has_more=False)
    )


@pytest.mark.asyncio
async def test_git_log_tool_pickaxe(ctx: ToolContext) -> None:
    result = await GitLogTool().run(ctx, GitLogArguments(pickaxe_string=PICKAXE_TOKEN))
    assert result == GitLogResult(
        result=GitLog(commits=[COMMIT_2], returned_count=1, has_more=False)
    )


@pytest.mark.asyncio
async def test_git_show_tool(ctx: ToolContext) -> None:
    result = await GitShowTool().run(
        ctx, GitShowArguments(ref=COMMIT_2.short_sha, include_patch=True)
    )
    # Patch body carries opaque blob-index shas; assert the commit + added line.
    assert result.result.commit == COMMIT_2
    assert result.result.truncated is False
    assert f"+{PICKAXE_TOKEN} = 1" in result.result.body


@pytest.mark.asyncio
async def test_git_diff_tool(ctx: ToolContext) -> None:
    result = await GitDiffTool().run(
        ctx,
        GitDiffArguments(from_ref=COMMIT_1.short_sha, to_ref=COMMIT_3.short_sha, stat_only=True),
    )
    assert result == GitDiffResult(
        result=GitDiff(
            diff=" config.py | 1 +\n runner.py | 2 +-\n 2 files changed, 2 insertions(+), 1 deletion(-)",
            stat_only=True,
            truncated=False,
        )
    )


@pytest.mark.asyncio
async def test_git_blame_tool(ctx: ToolContext) -> None:
    result = await GitBlameTool().run(
        ctx, GitBlameArguments(path="config.py", start_line=1, end_line=3)
    )
    assert result == GitBlameResult(
        result=GitBlame(
            path="config.py",
            lines=[
                BlameLine(
                    line_number=1,
                    short_sha=COMMIT_1.short_sha,
                    author=AUTHOR_NAME,
                    summary=COMMIT_1.subject,
                    line_text="MAX_RETRIES = 3",
                ),
                BlameLine(
                    line_number=2,
                    short_sha=COMMIT_1.short_sha,
                    author=AUTHOR_NAME,
                    summary=COMMIT_1.subject,
                    line_text="TIMEOUT_SECONDS = 30",
                ),
                BlameLine(
                    line_number=3,
                    short_sha=COMMIT_2.short_sha,
                    author=AUTHOR_NAME,
                    summary=COMMIT_2.subject,
                    line_text=f"{PICKAXE_TOKEN} = 1",
                ),
            ],
            returned_count=3,
            truncated=False,
        )
    )


@pytest.mark.asyncio
async def test_git_read_file_tool(ctx: ToolContext) -> None:
    result = await GitReadFileTool().run(
        ctx, GitReadFileArguments(ref=COMMIT_1.short_sha, path="config.py")
    )
    assert result == GitReadFileResult(
        result=FileContent(
            path="config.py",
            content="     1\tMAX_RETRIES = 3\n     2\tTIMEOUT_SECONDS = 30",
            start_line=1,
            end_line=2,
            total_line_count=2,
            truncated=False,
        )
    )


# --- require_git_repo --------------------------------------------------------


@pytest.mark.asyncio
async def test_git_log_requires_git_repo() -> None:
    with pytest.raises(RuntimeError, match="ToolContext.git_repo required"):
        await GitLogTool().run(ToolContext.model_construct(), GitLogArguments())


@pytest.mark.asyncio
async def test_git_show_requires_git_repo() -> None:
    with pytest.raises(RuntimeError, match="ToolContext.git_repo required"):
        await GitShowTool().run(ToolContext.model_construct(), GitShowArguments(ref="HEAD"))


@pytest.mark.asyncio
async def test_git_diff_requires_git_repo() -> None:
    with pytest.raises(RuntimeError, match="ToolContext.git_repo required"):
        await GitDiffTool().run(
            ToolContext.model_construct(), GitDiffArguments(from_ref="a", to_ref="b")
        )


@pytest.mark.asyncio
async def test_git_blame_requires_git_repo() -> None:
    with pytest.raises(RuntimeError, match="ToolContext.git_repo required"):
        await GitBlameTool().run(
            ToolContext.model_construct(),
            GitBlameArguments(path="x.py", start_line=1, end_line=1),
        )


@pytest.mark.asyncio
async def test_git_read_file_requires_git_repo() -> None:
    with pytest.raises(RuntimeError, match="ToolContext.git_repo required"):
        await GitReadFileTool().run(
            ToolContext.model_construct(), GitReadFileArguments(ref="HEAD", path="x.py")
        )


# --- argument-model validators -----------------------------------------------


def test_git_log_rejects_both_pickaxes() -> None:
    with pytest.raises(ValueError, match="at most one of pickaxe"):
        GitLogArguments(pickaxe_string="a", pickaxe_regex="b")


def test_git_blame_rejects_end_before_start() -> None:
    with pytest.raises(ValueError, match="end_line must be >= start_line"):
        GitBlameArguments(path="x.py", start_line=5, end_line=2)
