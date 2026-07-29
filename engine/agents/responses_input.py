"""Render chat-shaped ``AgentMessage`` history as Responses API input items.

The engine stores conversation history in the OpenAI chat-completions shape
(``AgentMessage``: role/content/tool_calls/tool_call_id) because that shape is
portable across providers and is what compaction and context tools operate on.
The Agents SDK, however, drives the **Responses API**, whose ``input`` array
has no ``tool_calls`` field and no ``tool`` role — tool turns are first-class
``function_call`` / ``function_call_output`` items, and every message item
must carry ``content``.

Passing chat-shaped dicts through used to work only by accident for histories
without tool turns; a tool-call-only assistant turn (``content`` dropped by
``exclude_none`` dumping) is rejected with
``400 missing_required_parameter: 'input[N].content'``. Because the runner
rebuilds the identical input on every retry, that one rejection deterministically
burned the whole retry budget and failed the run (run ba6fc95c).
"""

from __future__ import annotations

from typing import Any

from engine.models.messages import AgentMessage, MessageContent


def _content_or_empty(content: MessageContent) -> str | list[Any]:
    """``content`` is required on Responses message items; ``None`` must render
    as an empty string, never as an absent key."""
    return content if content is not None else ""


def to_responses_input(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """Convert a chat-shaped message array into wire-valid Responses input items.

    - ``system`` / ``user`` / plain ``assistant`` messages → message items with
      ``content`` always present.
    - assistant ``tool_calls`` → one ``function_call`` item per call, preceded
      by a message item only when the turn also carried text content.
    - ``tool`` results → ``function_call_output`` items keyed by ``call_id``.
    """
    items: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id or "",
                    "output": _content_or_empty(message.content),
                }
            )
            continue
        if message.role == "assistant" and message.tool_calls:
            if message.content:
                items.append({"role": "assistant", "content": message.content})
            for call in message.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                )
            continue
        items.append({"role": message.role, "content": _content_or_empty(message.content)})
    return items
