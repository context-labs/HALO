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
        "List commits from the git history of the repository (read-only). Returns bounded "
        "`CommitSummary` records (full/short sha, author, ISO-8601 `authored_at`, subject). "
        "REGRESSION HUNTING: scope to a trace's timeframe with `since`/`until` (ISO-8601, "
        "e.g. the trace's start/end timestamps) to see what shipped in that window; use "
        "`pickaxe_string` to find the commit that introduced or removed an exact string (a "
        "prompt fragment, a tool name like `spotify__login`, an error literal), or "
        "`pickaxe_regex` for a POSIX extended-regex pattern. `ref_range` ('A..B' or a "
        "branch/sha) and `path` "
        "(repo-relative) narrow further. If `has_more`, tighten the window rather than "
        "raising `max_commits`. Cite commits by short sha."
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
        "Inspect one commit (read-only). Returns the commit metadata plus a `--stat` file "
        "summary by default; set `include_patch=true` for the (size-capped) diff body. Use "
        "after `git_log`/`git_blame` surfaces a suspect short sha to see exactly what it "
        "changed. `path` limits the stat/patch to one file. Cite the commit by short sha."
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
        "Diff two refs (read-only), e.g. a known-good commit/tag (`from_ref`) vs a later one "
        "(`to_ref`) to see what changed across a regression window. Default `stat_only=true` "
        "(cheap file summary); set false for the (size-capped) patch. `path` limits to one "
        "file. Refs are shas/branches/tags."
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
        "Blame a line range of a file (read-only): for each line, the short sha, author, and "
        "commit subject that last changed it. Take a problematic `path:line` from "
        "`grep_files`/`read_file` and blame it to the commit that introduced the behavior, "
        "then `git_show` that sha. `ref` defaults to the current checkout. Keep the "
        "[start_line, end_line] window tight (capped)."
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
        "Read a file's full contents AS OF a specific commit (`git show <ref>:<path>`), as "
        "`cat -n` numbered lines with `offset`/`limit` paging — the same shape as `read_file`, "
        "but at a historical `ref` instead of the working tree. The traces were produced by "
        "whatever commit was deployed then, which may differ from the current checkout; use "
        "this to read the code as it actually ran. Cite as `path:line`."
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
