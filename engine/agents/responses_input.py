"""Render engine chat-shaped messages into Responses-API input items.

The engine stores conversation history as chat-completions-shaped
``AgentMessage``s: an assistant turn that called tools is one message with a
``tool_calls`` array and results are ``role="tool"`` messages. But the Agents
SDK runs on the OpenAI **Responses API** (``OpenAIProvider`` defaults to
``use_responses=True``), whose ``input`` array has no ``tool_calls`` field —
a ``role="assistant"`` item must carry ``content``, tool calls are standalone
``{"type": "function_call"}`` items, and results are
``{"type": "function_call_output"}`` items.

Sending the chat shape verbatim made every rerun-from-local-history attempt
after a tool turn (mid-stream failure recovery, refusal retries, the
``final_answer`` reprompt) fail deterministically with
``400 Missing required parameter: 'input[N].content'`` — the API sees the
assistant tool-call message as a content-less message item. Because
``is_retriable_llm_error`` deliberately retries non-terminal 400s, the same
invalid history was resent until the circuit breaker tripped and the run died
with ``EngineAgentExhaustedError``. This module is the render boundary that
keeps the local-history rebuild valid for the Responses API.
"""

from __future__ import annotations

import json
from typing import Any

from engine.models.messages import AgentMessage, MessageContent


def _content_to_text(content: MessageContent) -> str:
    """Flatten message content to plain text for a ``function_call_output``.

    Tool results are stored as strings by the event mapper; ``None`` and the
    (unused today) structured-parts shape degrade to ``""``/JSON rather than
    raising, so a rerun never dies on an odd historical item.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content)


def to_responses_input(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """Convert rendered ``AgentMessage``s into Responses-API input items.

    Mapping:
      - system/user/assistant text messages → ``{"role", "content"}`` message
        items (``None`` content becomes ``""`` — the Responses API requires
        ``content`` on message items).
      - assistant messages with ``tool_calls`` → an optional message item for
        any text content, then one ``function_call`` item per tool call.
      - ``role="tool"`` results → ``function_call_output`` items keyed by
        ``call_id``.

    Item order is preserved, so the context invariant (every ``function_call``
    is followed by its matching ``function_call_output``) carries over from
    ``AgentContext.trim_incomplete_tool_turn``.
    """
    items: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id or "",
                    "output": _content_to_text(msg.content),
                }
            )
            continue
        if msg.role == "assistant" and msg.tool_calls:
            if msg.content:
                items.append({"role": "assistant", "content": msg.content})
            for tc in msg.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                )
            continue
        content = msg.content if msg.content is not None else ""
        items.append({"role": msg.role, "content": content})
    return items
