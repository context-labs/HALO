"""Responses API ``input`` items must carry no provider-specific extras.

Regression for a release-gate failure: the main agent loop briefly wrapped
system/user ``content`` in an ``input_text`` part carrying Anthropic's
``cache_control`` hint, on the assumption that OpenAI would ignore an
unrecognised key the way Chat Completions does.

It does not. The Responses API validates strictly and rejects the request:

    400 - Unknown parameter: 'input[0].content[0].cache_control'

Every turn rebuilds the same input, so that 400 is deterministic and burns
the whole retry budget — the identical failure shape as run ba6fc95c, which
``responses_input.py`` exists to prevent.

Unit and integration tests all passed while this was broken, because they
never assert on the *wire* shape and the local stack routes through LiteLLM,
which tolerates the extra key. Only a direct-to-OpenAI call fails. Hence this
test: it pins the contract without needing a live provider.

``as_cached_system_message`` is unaffected and still in use — Chat Completions
genuinely does tolerate ``cache_control``. The two API surfaces differ, and
that difference is the whole point of this file.
"""

from __future__ import annotations

from typing import Any

from engine.agents.responses_input import to_responses_input
from engine.models.messages import AgentMessage, AgentToolCall, AgentToolFunction

# Keys no Responses input item may carry. Anything a provider-specific
# optimisation wants to attach belongs behind a provider check, not here.
FORBIDDEN_KEYS = {"cache_control"}


def _walk(value: Any) -> list[str]:
    """Every dict key anywhere inside a rendered item."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            found.append(key)
            found.extend(_walk(nested))
    elif isinstance(value, list):
        for entry in value:
            found.extend(_walk(entry))
    return found


def _rendered_history() -> list[dict[str, Any]]:
    return to_responses_input(
        [
            AgentMessage(role="system", content="engine system prompt"),
            AgentMessage(role="user", content="task prompt"),
            AgentMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    AgentToolCall(
                        id="c1",
                        function=AgentToolFunction(name="view_trace", arguments="{}"),
                    )
                ],
            ),
            AgentMessage(role="tool", content="rows", tool_call_id="c1"),
            AgentMessage(role="assistant", content="done"),
        ]
    )


def test_no_provider_specific_keys_reach_the_wire() -> None:
    for item in _rendered_history():
        keys = set(_walk(item))
        assert not (keys & FORBIDDEN_KEYS), (
            f"{sorted(keys & FORBIDDEN_KEYS)} in {item!r} — the Responses API "
            "rejects unknown parameters with a deterministic 400"
        )


def test_message_content_stays_a_plain_string() -> None:
    """The content-list shape is what smuggles extra keys onto the wire.

    Keeping message content a bare string makes the failure above structurally
    impossible rather than merely absent.
    """
    for item in _rendered_history():
        if "role" in item:
            assert isinstance(item["content"], str), (
                f"{item['role']} content must be a plain string, got "
                f"{type(item['content']).__name__}"
            )
