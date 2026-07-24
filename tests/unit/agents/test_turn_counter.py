from __future__ import annotations

from typing import cast

import pytest

from engine.agents.turn_counter import _render_nudge


class TestRenderNudge:
    def test_terse_when_4_or_more_remaining_root(self) -> None:
        # remaining = 10 - 1 + 1 = 10
        assert _render_nudge(current=1, maximum=10, is_root=True) == "[HALO: turn 1 of 10]"

    def test_terse_when_4_or_more_remaining_subagent(self) -> None:
        # remaining = 5 - 2 + 1 = 4
        assert _render_nudge(current=2, maximum=5, is_root=False) == "[HALO: turn 2 of 5]"

    def test_three_remaining_root(self) -> None:
        # remaining = 10 - 8 + 1 = 3
        assert _render_nudge(current=8, maximum=10, is_root=True) == (
            "[HALO: turn 8 of 10 — 3 turns left]"
        )

    def test_three_remaining_subagent(self) -> None:
        # remaining = 5 - 3 + 1 = 3
        assert _render_nudge(current=3, maximum=5, is_root=False) == (
            "[HALO: turn 3 of 5 — 3 turns left]"
        )

    def test_two_remaining_root(self) -> None:
        # remaining = 10 - 9 + 1 = 2
        assert _render_nudge(current=9, maximum=10, is_root=True) == (
            "[HALO: turn 9 of 10 — 2 turns left]"
        )

    def test_last_turn_root_mentions_final_answer_tool(self) -> None:
        # remaining = 10 - 10 + 1 = 1
        text = _render_nudge(current=10, maximum=10, is_root=True)
        assert text.startswith("[HALO: turn 10 of 10 — last turn.")
        assert "`final_answer`" in text
        assert text.endswith("now.]")

    def test_last_turn_subagent_does_not_mention_final_answer_tool(self) -> None:
        # remaining = 4 - 4 + 1 = 1
        text = _render_nudge(current=4, maximum=4, is_root=False)
        assert text.startswith("[HALO: turn 4 of 4 — last turn.")
        assert "final_answer" not in text
        assert "concise" in text

    def test_max_one_first_turn_is_last_turn_root(self) -> None:
        text = _render_nudge(current=1, maximum=1, is_root=True)
        assert text.startswith("[HALO: turn 1 of 1 — last turn.")
        assert "`final_answer`" in text

    def test_over_budget_defensive_fallback(self) -> None:
        # current > maximum: shouldn't happen (SDK raises MaxTurnsExceeded
        # first), but the formatter must not crash or produce a negative
        # "turns left." Defensive: just emit the count.
        text = _render_nudge(current=6, maximum=5, is_root=True)
        assert "turn 6 of 5" in text
        assert "-1 turns left" not in text  # no negative numbers


from agents.run_config import CallModelData, ModelInputData

from engine.agents.turn_counter import TurnCounterInputFilter


def _make_call_data(input_items: list[dict], instructions: str | None = "") -> CallModelData:
    """Minimal CallModelData stub for filter tests.

    The filter only reads ``data.model_data.input`` and
    ``data.model_data.instructions``; the agent and context fields are
    irrelevant to its behavior.
    """
    return CallModelData(
        model_data=ModelInputData(input=list(input_items), instructions=instructions),
        agent=None,  # type: ignore[arg-type]
        context=None,
    )


class TestTurnCounterInputFilter:
    def test_first_call_appends_terse_nudge_for_root(self) -> None:
        f = TurnCounterInputFilter(max_turns=10, is_root=True)
        original = [{"role": "user", "content": "do the thing"}]
        result = f(_make_call_data(original))
        assert result.input[-1] == {"role": "user", "content": "[HALO: turn 1 of 10]"}
        assert result.input[:-1] == original

    def test_counter_advances_across_calls(self) -> None:
        f = TurnCounterInputFilter(max_turns=10, is_root=True)
        for expected_turn in (1, 2, 3, 4, 5):
            result = f(_make_call_data([{"role": "user", "content": "x"}]))
            appended = cast(dict, result.input[-1])
            assert appended["content"] == f"[HALO: turn {expected_turn} of 10]"

    def test_two_filter_instances_have_independent_counters(self) -> None:
        f1 = TurnCounterInputFilter(max_turns=5, is_root=True)
        f2 = TurnCounterInputFilter(max_turns=5, is_root=False)
        f1(_make_call_data([{"role": "user", "content": "a"}]))
        f1(_make_call_data([{"role": "user", "content": "a"}]))
        result = f2(_make_call_data([{"role": "user", "content": "b"}]))
        appended = cast(dict, result.input[-1])
        assert appended["content"] == "[HALO: turn 1 of 5]"

    def test_does_not_mutate_input_list(self) -> None:
        f = TurnCounterInputFilter(max_turns=10, is_root=True)
        original = [{"role": "user", "content": "hi"}]
        snapshot = list(original)
        f(_make_call_data(original))
        assert original == snapshot, "filter must not mutate the caller's input list"

    def test_passes_instructions_through_unchanged(self) -> None:
        f = TurnCounterInputFilter(max_turns=10, is_root=True)
        result = f(_make_call_data([{"role": "user", "content": "x"}], instructions="sys text"))
        assert result.instructions == "sys text"

    def test_subagent_last_turn_uses_subagent_wording(self) -> None:
        f = TurnCounterInputFilter(max_turns=2, is_root=False)
        f(_make_call_data([{"role": "user", "content": "x"}]))
        result = f(_make_call_data([{"role": "user", "content": "x"}]))
        text = cast(str, cast(dict, result.input[-1])["content"])
        assert "concise" in text
        assert "final_answer" not in text

    def test_appended_item_is_user_role(self) -> None:
        f = TurnCounterInputFilter(max_turns=10, is_root=True)
        result = f(_make_call_data([]))
        appended = cast(dict, result.input[-1])
        assert appended["role"] == "user"
