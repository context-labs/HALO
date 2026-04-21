"""Agent definition with three read-only file tools scoped to a root directory."""
import subprocess
from pathlib import Path

from agents import Agent, RunContextWrapper, function_tool
from pydantic import BaseModel


class AgentContext(BaseModel):
    """Per-run context carrying the filesystem scope the agent may read."""
    root: Path


def _resolve_within_root(root: Path, relative: str) -> Path:
    """Resolve `relative` under `root` and raise if it escapes root."""
    root = root.resolve()
    resolved = (root / relative).resolve()
    resolved.relative_to(root)
    return resolved


@function_tool
def list_files(
    ctx: RunContextWrapper[AgentContext],
    subdir: str = ".",
    pattern: str = "*",
) -> list[str]:
    """List files under subdir (relative to root), optionally filtered by glob pattern.

    Args:
        subdir: Directory relative to root.
        pattern: Glob like "*.py" to filter results. Use "*" for all files.
    """
    target = _resolve_within_root(ctx.context.root, subdir)
    result = subprocess.run(
        ["rg", "--files", "--glob", pattern, "."],
        cwd=target,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


@function_tool
def read_file(
    ctx: RunContextWrapper[AgentContext],
    path: str,
    start: int = 1,
    end: int | None = None,
) -> str:
    """Read a 1-indexed line slice from a file under root; capped at 500 lines.

    Args:
        path: File path relative to root.
        start: First line (1-indexed).
        end: Last line inclusive; defaults to start + 499. Capped at start + 499.
    """
    target = _resolve_within_root(ctx.context.root, path)
    lines = target.read_text().splitlines()
    max_end = min(start + 499, len(lines))
    if end is None:
        end = max_end
    end = min(end, max_end)
    return "\n".join(lines[start - 1 : end])


@function_tool
def grep(
    ctx: RunContextWrapper[AgentContext],
    pattern: str,
    subdir: str = ".",
    glob: str = "*",
) -> list[str]:
    """Search file contents using ripgrep under root/subdir.

    Args:
        pattern: Regex to search for.
        subdir: Directory relative to root.
        glob: File glob like "*.py".
    """
    target = _resolve_within_root(ctx.context.root, subdir)
    result = subprocess.run(
        ["rg", "-n", "--glob", glob, pattern, "."],
        cwd=target,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.splitlines()


def build_agent(model: str) -> Agent:
    return Agent(
        name="HaloCodeHelper",
        instructions=(
            "You answer questions about a local codebase using list_files, grep, and "
            "read_file. Start broad (list_files or grep), then read specific files. "
            "Answer concisely with file:line citations."
        ),
        model=model,
        tools=[list_files, read_file, grep],
    )
