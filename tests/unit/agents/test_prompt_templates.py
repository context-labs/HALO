from __future__ import annotations

from pathlib import Path

from engine.agents.prompt_templates import (
    CODE_REPO_PROMPT_SECTION_TEMPLATE,
    COMPACTION_SYSTEM_PROMPT,
    FINAL_SENTINEL,
    GIT_REPO_PROMPT_SECTION_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    render_root_system_prompt,
    render_subagent_system_prompt,
)
from engine.code.code_repo import CodeRepo
from engine.git.git_repo import GitRepo
from tests.unit.git.git_fixture import build_empty_git_repo


def _code_repo(tmp_path: Path) -> CodeRepo:
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "main.py").write_text("print('hi')\n")
    return CodeRepo.open(tmp_path)


def _git_repo(tmp_path: Path) -> GitRepo:
    repo = GitRepo.open(build_empty_git_repo(tmp_path))
    assert repo is not None
    return repo


def test_final_sentinel_constant() -> None:
    assert FINAL_SENTINEL == "<final/>"


def test_root_prompt_includes_sentinel_system_prompt_and_caps() -> None:
    text = render_root_system_prompt(
        maximum_depth=2,
        maximum_parallel_subagents=4,
        code_repo=None,
        git_repo=None,
    )
    assert FINAL_SENTINEL in text
    assert SYSTEM_PROMPT in text
    assert "maximum_depth=2" in text
    assert "Spawn at most 4 subagents concurrently." in text


def test_root_prompt_omits_code_section_without_repo() -> None:
    text = render_root_system_prompt(
        maximum_depth=2,
        maximum_parallel_subagents=4,
        code_repo=None,
        git_repo=None,
    )
    assert "Code repository:" not in text


def test_root_prompt_includes_code_section_with_repo(tmp_path: Path) -> None:
    repo = _code_repo(tmp_path)
    text = render_root_system_prompt(
        maximum_depth=2,
        maximum_parallel_subagents=4,
        code_repo=repo,
        git_repo=None,
    )
    assert CODE_REPO_PROMPT_SECTION_TEMPLATE.format(repo_root=repo.root) in text


def test_root_prompt_omits_git_section_without_repo() -> None:
    text = render_root_system_prompt(
        maximum_depth=2,
        maximum_parallel_subagents=4,
        code_repo=None,
        git_repo=None,
    )
    assert "Git history:" not in text


def test_root_prompt_includes_git_section_with_repo(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    text = render_root_system_prompt(
        maximum_depth=2,
        maximum_parallel_subagents=4,
        code_repo=None,
        git_repo=repo,
    )
    assert GIT_REPO_PROMPT_SECTION_TEMPLATE.format(repo_root=repo.root) in text


def test_subagent_prompt_reports_depth_caps_and_system_prompt() -> None:
    text = render_subagent_system_prompt(
        depth=1,
        maximum_depth=2,
        maximum_parallel_subagents=4,
        code_repo=None,
        git_repo=None,
    )
    assert "depth=1" in text
    assert "maximum_depth=2" in text
    assert "spawn at most 4" in text and "concurrently" in text
    assert SYSTEM_PROMPT in text
    assert FINAL_SENTINEL in text
    assert "Code repository:" not in text
    assert "Git history:" not in text


def test_subagent_prompt_includes_code_section_with_repo(tmp_path: Path) -> None:
    repo = _code_repo(tmp_path)
    text = render_subagent_system_prompt(
        depth=1,
        maximum_depth=2,
        maximum_parallel_subagents=4,
        code_repo=repo,
        git_repo=None,
    )
    assert CODE_REPO_PROMPT_SECTION_TEMPLATE.format(repo_root=repo.root) in text


def test_subagent_prompt_includes_git_section_with_repo(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    text = render_subagent_system_prompt(
        depth=1,
        maximum_depth=2,
        maximum_parallel_subagents=4,
        code_repo=None,
        git_repo=repo,
    )
    assert GIT_REPO_PROMPT_SECTION_TEMPLATE.format(repo_root=repo.root) in text


def test_compaction_and_synthesis_prompts_are_strings() -> None:
    assert isinstance(COMPACTION_SYSTEM_PROMPT, str) and COMPACTION_SYSTEM_PROMPT
    assert isinstance(SYNTHESIS_SYSTEM_PROMPT, str) and SYNTHESIS_SYSTEM_PROMPT
