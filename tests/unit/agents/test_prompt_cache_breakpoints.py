"""Cache breakpoints on the stable head of a Responses ``input`` array.

The main agent loop replays the whole conversation every turn, so the
system + task-prompt head is re-sent byte-identically on every turn and every
retry. Before this, only ``compactor`` and ``synthesis_tool`` — two one-shot
auxiliary calls — marked anything cacheable; the loop that spends nearly all of
HALO's tokens marked nothing.

These tests pin the wire shape as much as the behaviour: a malformed replay
item is the failure that deterministically burned a whole retry budget in run
ba6fc95c, so the rewrite must stay confined to item shapes it can be sure of.
"""

from __future__ import annotations

from engine.agents.prompt_caching import (
    CACHE_CONTROL_EPHEMERAL,
    apply_prompt_cache_breakpoints,
)


def _cached_part(text: str) -> list[dict[str, object]]:
    return [
        {
            "type": "input_text",
            "text": text,
            "cache_control": CACHE_CONTROL_EPHEMERAL,
        }
    ]


def test_marks_system_and_last_user_item_of_the_head() -> None:
    items = apply_prompt_cache_breakpoints(
        [
            {"role": "system", "content": "engine system prompt"},
            {"role": "user", "content": "analyse these traces"},
            {"role": "assistant", "content": "on it"},
        ]
    )

    assert items[0]["content"] == _cached_part("engine system prompt")
    assert items[1]["content"] == _cached_part("analyse these traces")
    # Assistant output uses ``output_text``, not ``input_text`` — leaving it
    # untouched is what keeps this rewrite provably wire-safe.
    assert items[2] == {"role": "assistant", "content": "on it"}


def test_single_head_item_gets_exactly_one_breakpoint() -> None:
    items = apply_prompt_cache_breakpoints(
        [
            {"role": "system", "content": "only the system prompt"},
            {"role": "assistant", "content": "hi"},
        ]
    )

    assert items[0]["content"] == _cached_part("only the system prompt")
    assert items[1] == {"role": "assistant", "content": "hi"}


def test_never_touches_tool_turn_items() -> None:
    """``function_call`` / ``function_call_output`` have no content-part list.

    The head scan stops at the first non-input-role item, so the growing
    tool-result history is passed through byte-for-byte.
    """
    original = [
        {"role": "system", "content": "sys"},
        {"type": "function_call", "call_id": "c1", "name": "query_traces", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "rows"},
    ]
    items = apply_prompt_cache_breakpoints(original)

    assert items[0]["content"] == _cached_part("sys")
    assert items[1] == original[1]
    assert items[2] == original[2]


def test_head_stops_at_the_first_non_input_role() -> None:
    """A later ``user`` item is not a breakpoint — it isn't in the stable head.

    Only the contiguous leading run is byte-stable across turns. A user message
    appended after tool turns (the "Continue." nudge, for instance) moves, so
    marking it would write a cache entry nothing later reads.
    """
    items = apply_prompt_cache_breakpoints(
        [
            {"role": "system", "content": "sys"},
            {"type": "function_call_output", "call_id": "c1", "output": "rows"},
            {"role": "user", "content": "Continue."},
        ]
    )

    assert items[2] == {"role": "user", "content": "Continue."}


def test_leaves_empty_and_already_structured_content_alone() -> None:
    already = [{"type": "input_text", "text": "pre-built"}]
    items = apply_prompt_cache_breakpoints(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": already},
        ]
    )

    # Nothing worth caching, and nothing double-wrapped.
    assert items[0] == {"role": "system", "content": ""}
    assert items[1] == {"role": "user", "content": already}


def test_does_not_mutate_the_input_list() -> None:
    original = [{"role": "system", "content": "sys"}]
    apply_prompt_cache_breakpoints(original)
    assert original == [{"role": "system", "content": "sys"}]


def test_empty_input_is_a_noop() -> None:
    assert apply_prompt_cache_breakpoints([]) == []
