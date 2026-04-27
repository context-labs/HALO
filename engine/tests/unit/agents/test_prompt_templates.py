from __future__ import annotations

from engine.agents.prompt_templates import (
    COMPACTION_SYSTEM_PROMPT,
    FINAL_SENTINEL,
    ROOT_SYSTEM_PROMPT_TEMPLATE,
    SUBAGENT_SYSTEM_PROMPT_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
    render_root_system_prompt,
    render_subagent_system_prompt,
)


def test_final_sentinel_constant() -> None:
    assert FINAL_SENTINEL == "<final/>"


def test_root_prompt_includes_sentinel_instruction() -> None:
    text = render_root_system_prompt(
        instructions="Investigate failing traces.",
        maximum_depth=2,
        maximum_parallel_subagents=4,
    )
    assert FINAL_SENTINEL in text
    assert "Investigate failing traces." in text


def test_subagent_prompt_reports_depth() -> None:
    text = render_subagent_system_prompt(
        instructions="You are a sub.",
        depth=1,
        maximum_depth=2,
        maximum_parallel_subagents=4,
    )
    assert "depth=1" in text
    assert "maximum_depth=2" in text
    # Regression: the parallel cap was passed to .format() but had no
    # placeholder, so it was silently dropped.
    assert "4" in text and "concurrently" in text


def test_compaction_and_synthesis_prompts_are_strings() -> None:
    assert isinstance(COMPACTION_SYSTEM_PROMPT, str) and COMPACTION_SYSTEM_PROMPT
    assert isinstance(SYNTHESIS_SYSTEM_PROMPT, str) and SYNTHESIS_SYSTEM_PROMPT
    assert "<final/>" in ROOT_SYSTEM_PROMPT_TEMPLATE
    assert "{instructions}" in SUBAGENT_SYSTEM_PROMPT_TEMPLATE
