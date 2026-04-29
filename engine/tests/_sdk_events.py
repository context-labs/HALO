"""Real-SDK ``StreamEvent`` factories for unit tests.

Tests that drive the engine through the OpenAI Agents SDK boundary used to
build ``SimpleNamespace`` fakes shaped like the SDK's stream events. That
worked while the mapper was duck-typed, but it became fragile once the
mapper switched to ``isinstance`` checks against the real SDK item classes.
Factories here build the *actual* SDK objects (``RunItemStreamEvent``,
``ToolCallItem``, ``ResponseFunctionToolCall``, etc.) with minimal
scaffolding so test bodies stay focused on the assertion, not the
constructor noise.
"""

from __future__ import annotations

from typing import Any

from agents import Agent
from agents.items import MessageOutputItem, ToolCallItem, ToolCallOutputItem
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
)

# Real SDK ``RunItemBase`` requires an ``Agent`` reference. None of the
# engine code under test reads it; one shared stand-in keeps the test
# bodies focused on shape rather than on agent setup.
SHARED_AGENT = Agent(name="test-root")


def assistant_message_event(*, item_id: str, text: str) -> RunItemStreamEvent:
    """Build a real ``RunItemStreamEvent`` carrying a ``MessageOutputItem``."""
    raw = ResponseOutputMessage(
        id=item_id,
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )
    return RunItemStreamEvent(
        name="message_output_created",
        item=MessageOutputItem(agent=SHARED_AGENT, raw_item=raw),
    )


def tool_call_event(
    *,
    call_id: str,
    name: str,
    arguments: str = "{}",
    raw_id: str | None = None,
) -> RunItemStreamEvent:
    """Build a real ``RunItemStreamEvent`` carrying a ``ToolCallItem`` (Responses-API shape)."""
    raw = ResponseFunctionToolCall(
        id=raw_id,
        call_id=call_id,
        name=name,
        arguments=arguments,
        type="function_call",
    )
    return RunItemStreamEvent(
        name="tool_called",
        item=ToolCallItem(agent=SHARED_AGENT, raw_item=raw),
    )


def tool_output_event(
    *,
    call_id: str,
    output: str = "ok",
    raw_id: str | None = None,
) -> RunItemStreamEvent:
    """Build a real ``RunItemStreamEvent`` carrying a ``ToolCallOutputItem``.

    Uses the dict raw-item form, which is what Chat-Completions adapters and
    replay-from-input paths actually emit. The mapper's SDK property access
    handles both shapes uniformly; using the dict form here exercises the
    harder code path.
    """
    raw_item: dict[str, Any] = {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }
    if raw_id is not None:
        raw_item["id"] = raw_id
    return RunItemStreamEvent(
        name="tool_output",
        item=ToolCallOutputItem(agent=SHARED_AGENT, raw_item=raw_item, output=output),
    )


def text_delta_event(*, item_id: str, delta: str) -> RawResponsesStreamEvent:
    """Build a real ``RawResponsesStreamEvent`` for an ``output_text.delta``."""
    return RawResponsesStreamEvent(
        data=ResponseTextDeltaEvent(
            type="response.output_text.delta",
            item_id=item_id,
            delta=delta,
            output_index=0,
            content_index=0,
            sequence_number=0,
            logprobs=[],
        )
    )
