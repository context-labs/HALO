from __future__ import annotations

from pathlib import Path

from engine.agents.prompt_templates import (
    COMPACTION_SYSTEM_PROMPT,
    FINAL_SENTINEL,
    SYNTHESIS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    render_root_system_prompt,
    render_subagent_system_prompt,
)
from engine.code.code_repo import CodeRepo


def _code_repo(tmp_path: Path) -> CodeRepo:
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "main.py").write_text("print('hi')\n")
    return CodeRepo.open(tmp_path)


def test_final_sentinel_constant() -> None:
    assert FINAL_SENTINEL == "<final/>"


def test_root_prompt_includes_sentinel_system_prompt_and_caps() -> None:
    text = render_root_system_prompt(
        maximum_depth=2,
        maximum_parallel_subagents=4,
        code_repo=None,
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
    )
    assert "Code repository:" not in text


def test_root_prompt_includes_code_section_with_repo(tmp_path: Path) -> None:
    repo = _code_repo(tmp_path)
    text = render_root_system_prompt(
        maximum_depth=2,
        maximum_parallel_subagents=4,
        code_repo=repo,
    )
    assert "Code repository:" in text
    assert str(repo.root) in text
    assert "engine/" in text
    assert "main.py" in text
    assert "`path:line`" in text


def test_subagent_prompt_reports_depth_caps_and_system_prompt() -> None:
    text = render_subagent_system_prompt(
        depth=1,
        maximum_depth=2,
        maximum_parallel_subagents=4,
        code_repo=None,
    )
    assert "depth=1" in text
    assert "maximum_depth=2" in text
    assert "spawn at most 4" in text and "concurrently" in text
    assert SYSTEM_PROMPT in text
    assert FINAL_SENTINEL in text
    assert "Code repository:" not in text


def test_subagent_prompt_includes_code_section_with_repo(tmp_path: Path) -> None:
    repo = _code_repo(tmp_path)
    text = render_subagent_system_prompt(
        depth=1,
        maximum_depth=2,
        maximum_parallel_subagents=4,
        code_repo=repo,
    )
    assert "Code repository:" in text
    assert str(repo.root) in text


def test_compaction_and_synthesis_prompts_are_strings() -> None:
    assert isinstance(COMPACTION_SYSTEM_PROMPT, str) and COMPACTION_SYSTEM_PROMPT
    assert isinstance(SYNTHESIS_SYSTEM_PROMPT, str) and SYNTHESIS_SYSTEM_PROMPT
