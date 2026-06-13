from __future__ import annotations

from pathlib import Path

import pytest

from engine.code.code_repo import CodeRepo
from engine.code.models import (
    GlobFilesArguments,
    GrepFilesArguments,
    ReadFileArguments,
    ViewRepoTreeArguments,
)
from engine.tools.code_tools import (
    GlobFilesTool,
    GrepFilesTool,
    ReadFileTool,
    ViewRepoTreeTool,
)
from engine.tools.tool_protocol import ToolContext


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "config.py").write_text('CONFIG = {"max_retries": 3}\n')
    (tmp_path / "engine" / "main.py").write_text("def main():\n    return 0\n")
    repo = CodeRepo.open(tmp_path)
    return ToolContext.model_construct(code_repo=repo)


@pytest.mark.asyncio
async def test_glob_files_tool(ctx: ToolContext) -> None:
    tool = GlobFilesTool()
    result = await tool.run(ctx, GlobFilesArguments(pattern="**/*.py"))
    assert [f.path for f in result.result.files] == ["engine/config.py", "engine/main.py"]
    assert result.result.has_more is False


@pytest.mark.asyncio
async def test_grep_files_tool(ctx: ToolContext) -> None:
    tool = GrepFilesTool()
    result = await tool.run(ctx, GrepFilesArguments(regex_pattern="max_retries"))
    assert [(m.path, m.line_number) for m in result.result.matches] == [("engine/config.py", 1)]


@pytest.mark.asyncio
async def test_read_file_tool(ctx: ToolContext) -> None:
    tool = ReadFileTool()
    result = await tool.run(ctx, ReadFileArguments(path="engine/main.py"))
    assert result.result.content == "     1\tdef main():\n     2\t    return 0"
    assert result.result.total_line_count == 2


@pytest.mark.asyncio
async def test_view_repo_tree_tool(ctx: ToolContext) -> None:
    tool = ViewRepoTreeTool()
    result = await tool.run(ctx, ViewRepoTreeArguments())
    assert "engine/" in result.result.tree
    assert "config.py" in result.result.tree
    assert result.result.root  # absolute repo root path


@pytest.mark.asyncio
async def test_view_repo_tree_requires_code_repo() -> None:
    tool = ViewRepoTreeTool()
    with pytest.raises(RuntimeError, match="ToolContext.code_repo required"):
        await tool.run(ToolContext.model_construct(), ViewRepoTreeArguments())


@pytest.mark.asyncio
async def test_glob_requires_code_repo() -> None:
    tool = GlobFilesTool()
    with pytest.raises(RuntimeError, match="ToolContext.code_repo required"):
        await tool.run(ToolContext.model_construct(), GlobFilesArguments(pattern="*.py"))


@pytest.mark.asyncio
async def test_grep_requires_code_repo() -> None:
    tool = GrepFilesTool()
    with pytest.raises(RuntimeError, match="ToolContext.code_repo required"):
        await tool.run(ToolContext.model_construct(), GrepFilesArguments(regex_pattern="x"))


@pytest.mark.asyncio
async def test_read_requires_code_repo() -> None:
    tool = ReadFileTool()
    with pytest.raises(RuntimeError, match="ToolContext.code_repo required"):
        await tool.run(ToolContext.model_construct(), ReadFileArguments(path="x.py"))
