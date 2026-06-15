from __future__ import annotations

import asyncio

from engine.git.models import (
    GitBlameArguments,
    GitBlameResult,
    GitDiffArguments,
    GitDiffResult,
    GitLogArguments,
    GitLogResult,
    GitReadFileArguments,
    GitReadFileResult,
    GitShowArguments,
    GitShowResult,
)
from engine.tools.tool_protocol import ToolContext


class GitLogTool:
    """Tool wrapper around ``GitRepo.log``: list commits (read-only).

    Stateless: the live ``GitRepo`` comes through ``tool_context.git_repo`` (wired
    by the per-run ``make_ctx`` factory). Registration is gated upstream on a git
    work tree, so ``tool_context.git_repo`` is always populated here.
    """

    name = "git_log"
    description = (
        "List commits, newest first (read-only), as bounded `CommitSummary` records "
        "(full/short sha, author, ISO-8601 `authored_at`, subject). `since`/`until` window "
        "by date; `pickaxe_string` finds commits that changed an exact string and "
        "`pickaxe_regex` a POSIX extended-regex (not PCRE — `\\w` won't work) — at most one. "
        "`ref_range` ('A..B' or a branch/sha) and `path` narrow further. If `has_more`, "
        "tighten the window rather than raising `max_commits`."
    )
    arguments_model = GitLogArguments
    result_model = GitLogResult

    async def run(self, tool_context: ToolContext, arguments: GitLogArguments) -> GitLogResult:
        """List commits, off the event loop."""
        repo = tool_context.require_git_repo()
        result = await asyncio.to_thread(
            repo.log,
            max_commits=arguments.max_commits,
            since=arguments.since,
            until=arguments.until,
            ref_range=arguments.ref_range,
            path=arguments.path,
            pickaxe_string=arguments.pickaxe_string,
            pickaxe_regex=arguments.pickaxe_regex,
        )
        return GitLogResult(result=result)


class GitShowTool:
    """Tool wrapper around ``GitRepo.show``: inspect one commit (read-only)."""

    name = "git_show"
    description = (
        "Show one commit (read-only): its metadata plus a `--stat` file summary, or the "
        "(size-capped) patch when `include_patch=true`. `path` limits to one file."
    )
    arguments_model = GitShowArguments
    result_model = GitShowResult

    async def run(self, tool_context: ToolContext, arguments: GitShowArguments) -> GitShowResult:
        """Show one commit, off the event loop."""
        repo = tool_context.require_git_repo()
        result = await asyncio.to_thread(
            repo.show,
            ref=arguments.ref,
            path=arguments.path,
            include_patch=arguments.include_patch,
        )
        return GitShowResult(result=result)


class GitDiffTool:
    """Tool wrapper around ``GitRepo.diff``: diff two refs (read-only)."""

    name = "git_diff"
    description = (
        "Diff two refs (read-only): a `--stat` summary by default, or the (size-capped) patch "
        "when `stat_only=false`. `path` limits to one file; refs are shas/branches/tags."
    )
    arguments_model = GitDiffArguments
    result_model = GitDiffResult

    async def run(self, tool_context: ToolContext, arguments: GitDiffArguments) -> GitDiffResult:
        """Diff two refs, off the event loop."""
        repo = tool_context.require_git_repo()
        result = await asyncio.to_thread(
            repo.diff,
            from_ref=arguments.from_ref,
            to_ref=arguments.to_ref,
            path=arguments.path,
            stat_only=arguments.stat_only,
        )
        return GitDiffResult(result=result)


class GitBlameTool:
    """Tool wrapper around ``GitRepo.blame``: attribute a file's line range to commits."""

    name = "git_blame"
    description = (
        "Blame a file's line range (read-only): the short sha, author, and subject that last "
        "changed each line. `ref` defaults to the working tree; keep the [start_line, "
        "end_line] window tight (capped)."
    )
    arguments_model = GitBlameArguments
    result_model = GitBlameResult

    async def run(self, tool_context: ToolContext, arguments: GitBlameArguments) -> GitBlameResult:
        """Blame a line range, off the event loop."""
        repo = tool_context.require_git_repo()
        result = await asyncio.to_thread(
            repo.blame,
            path=arguments.path,
            start_line=arguments.start_line,
            end_line=arguments.end_line,
            ref=arguments.ref,
        )
        return GitBlameResult(result=result)


class GitReadFileTool:
    """Tool wrapper around ``GitRepo.read_at_ref``: read a file as of a specific commit."""

    name = "git_read_file"
    description = (
        "Read a file's contents AS OF a commit (`git show <ref>:<path>`) as `cat -n` numbered "
        "lines with `offset`/`limit` paging — like `read_file` but at a historical `ref`. "
        "Reads the code as it actually ran: the traced commit may differ from the current "
        "checkout."
    )
    arguments_model = GitReadFileArguments
    result_model = GitReadFileResult

    async def run(
        self, tool_context: ToolContext, arguments: GitReadFileArguments
    ) -> GitReadFileResult:
        """Read a file at a ref, off the event loop."""
        repo = tool_context.require_git_repo()
        result = await asyncio.to_thread(
            repo.read_at_ref,
            ref=arguments.ref,
            path=arguments.path,
            offset=arguments.offset,
            limit=arguments.limit,
        )
        return GitReadFileResult(result=result)
