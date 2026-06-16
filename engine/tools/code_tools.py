from __future__ import annotations

import asyncio

from engine.code.models import (
    GlobFilesArguments,
    GlobFilesResult,
    GrepFilesArguments,
    GrepFilesResult,
    ReadFileArguments,
    ReadFileResult,
    RepoTree,
    ViewRepoTreeArguments,
    ViewRepoTreeResult,
)
from engine.tools.tool_protocol import ToolContext


class ViewRepoTreeTool:
    """Tool wrapper around ``CodeRepo.tree``: a directory overview of the repo.

    Stateless: the live ``CodeRepo`` comes through ``tool_context.code_repo``
    (wired by the per-run ``make_ctx`` factory). Registration is gated upstream
    on a configured repo, so ``tool_context.code_repo`` is always populated here.
    """

    name = "view_repo_tree"
    description = (
        "Return a directory-tree overview of the code repository (depth/entry capped, "
        "honors .gitignore, VCS/build/cache dirs pruned). Call this once for orientation "
        "before searching; the result is cached for the run. Use `glob_files` for anything "
        "the tree's caps leave out, and `grep_files`/`read_file` to drill into contents."
    )
    arguments_model = ViewRepoTreeArguments
    result_model = ViewRepoTreeResult

    async def run(
        self, tool_context: ToolContext, arguments: ViewRepoTreeArguments
    ) -> ViewRepoTreeResult:
        """Return the cached (lazily rendered) repo tree overview."""
        del arguments
        repo = tool_context.require_code_repo()
        # ``repo.tree`` shells out to ripgrep on first access — off the event loop.
        tree = await asyncio.to_thread(lambda: repo.tree)
        return ViewRepoTreeResult(result=RepoTree(root=str(repo.root), tree=tree))


class GlobFilesTool:
    """Tool wrapper around ``CodeRepo.glob``: discover repo files by glob pattern.

    Stateless: the live ``CodeRepo`` comes through ``tool_context.code_repo``
    (wired by the per-run ``make_ctx`` factory). Registration is gated upstream
    on a configured repo, so ``tool_context.code_repo`` is always populated here.
    """

    name = "glob_files"
    description = (
        "Find files in the code repository by glob pattern (gitignore-style: a pattern "
        "without `/` matches at any depth, so `*.py` finds all Python files; `engine/*.py` "
        "is anchored to that directory). Returns matching paths and byte sizes (sorted, "
        "honors .gitignore, junk dirs like .git/node_modules pruned). If `has_more` is true, "
        "narrow the pattern rather than raising `max_results`. Use this to locate files by "
        "name; use `grep_files` to search file contents."
    )
    arguments_model = GlobFilesArguments
    result_model = GlobFilesResult

    async def run(
        self, tool_context: ToolContext, arguments: GlobFilesArguments
    ) -> GlobFilesResult:
        """Match files against the glob pattern, confined to the repo root."""
        repo = tool_context.require_code_repo()
        # ripgrep subprocess — run off the event loop so concurrent agents don't stall.
        result = await asyncio.to_thread(repo.glob, arguments.pattern, arguments.max_results)
        return GlobFilesResult(result=result)


class GrepFilesTool:
    """Tool wrapper around ``CodeRepo.grep``: regex content search across the repo."""

    name = "grep_files"
    description = (
        "Regex-search file contents across the code repository (honors .gitignore). Returns "
        "up to `max_matches` records (`path`, 1-based `line_number`, matched `line_text`), "
        "plus `has_more`. Optionally pass `glob_pattern` to confine the search to "
        "matching files (e.g. `engine/**/*.py`). Prefer this over reading whole files "
        "to locate symbols, strings, or definitions; then `read_file` a window around "
        "the line. If `has_more` is true, refine the regex rather than raising "
        "`max_matches`. Batch independent searches in parallel when you can."
    )
    arguments_model = GrepFilesArguments
    result_model = GrepFilesResult

    async def run(
        self, tool_context: ToolContext, arguments: GrepFilesArguments
    ) -> GrepFilesResult:
        """Run a bounded regex search over repo file contents."""
        repo = tool_context.require_code_repo()
        # ripgrep subprocess — run off the event loop so concurrent agents don't stall.
        result = await asyncio.to_thread(
            repo.grep,
            arguments.regex_pattern,
            arguments.glob_pattern,
            arguments.max_matches,
        )
        return GrepFilesResult(result=result)


class ReadFileTool:
    """Tool wrapper around ``CodeRepo.read``: numbered file contents with offset/limit paging."""

    name = "read_file"
    description = (
        "Read a file from the code repository as `cat -n` numbered lines, starting at "
        "1-based `offset` for `limit` lines. Use `glob_files`/`grep_files` to find the "
        "path and line numbers first, then read a window around them. `total_line_count` "
        "lets you page with `offset`; `truncated` flags a clipped long line or response. "
        "Cite code as `path:line` using these line numbers."
    )
    arguments_model = ReadFileArguments
    result_model = ReadFileResult

    async def run(self, tool_context: ToolContext, arguments: ReadFileArguments) -> ReadFileResult:
        """Read a 1-based line window of the file, confined to the repo root."""
        repo = tool_context.require_code_repo()
        # File I/O — run off the event loop so concurrent agents don't stall.
        result = await asyncio.to_thread(
            repo.read,
            arguments.path,
            arguments.offset,
            arguments.limit,
        )
        return ReadFileResult(result=result)
