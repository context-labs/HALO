from __future__ import annotations

from engine.agents.prompt_templates import (
    COMPACTION_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    FINAL_SENTINEL,
    SYNTHESIS_SYSTEM_PROMPT,
    render_root_system_prompt,
    render_subagent_system_prompt,
)


def test_final_sentinel_constant() -> None:
    assert FINAL_SENTINEL == "<final/>"


def test_root_prompt_includes_sentinel_instructions_and_caps() -> None:
    text = render_root_system_prompt(
        instructions="Investigate failing traces.",
        maximum_depth=2,
        maximum_parallel_subagents=4,
    )
    assert FINAL_SENTINEL in text
    assert "Investigate failing traces." in text
    assert "maximum_depth=2" in text
    assert "Spawn at most 4 subagents concurrently." in text


def test_root_prompt_uses_default_system_prompt_when_instructions_none() -> None:
    text = render_root_system_prompt(
        instructions=None,
        maximum_depth=1,
        maximum_parallel_subagents=2,
    )
    assert DEFAULT_SYSTEM_PROMPT in text
    assert FINAL_SENTINEL in text


def test_subagent_prompt_reports_depth_and_caps() -> None:
    text = render_subagent_system_prompt(
        instructions="You are a sub.",
        depth=1,
        maximum_depth=2,
        maximum_parallel_subagents=4,
    )
    assert "depth=1" in text
    assert "maximum_depth=2" in text
    assert "spawn at most 4" in text and "concurrently" in text
    assert FINAL_SENTINEL in text


def test_subagent_prompt_uses_default_system_prompt_when_instructions_none() -> None:
    text = render_subagent_system_prompt(
        instructions=None,
        depth=1,
        maximum_depth=2,
        maximum_parallel_subagents=2,
    )
    assert DEFAULT_SYSTEM_PROMPT in text


def test_compaction_and_synthesis_prompts_are_strings() -> None:
    assert isinstance(COMPACTION_SYSTEM_PROMPT, str) and COMPACTION_SYSTEM_PROMPT
    assert isinstance(SYNTHESIS_SYSTEM_PROMPT, str) and SYNTHESIS_SYSTEM_PROMPT
