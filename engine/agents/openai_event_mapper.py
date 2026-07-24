from __future__ import annotations

from dataclasses import dataclass

from agents.items import MessageOutputItem, ToolCallItem, ToolCallOutputItem
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent, StreamEvent
from openai.types.responses import ResponseOutputRefusal, ResponseOutputText
from pydantic import ValidationError

from engine.agents.agent_context_items import AgentContextItem
from engine.agents.agent_execution import AgentExecution
from engine.models.engine_output import AgentOutputItem, AgentTextDelta
from engine.models.messages import AgentMessage, AgentToolCall, AgentToolFunction
from engine.tools.final_answer_tool import FINAL_ANSWER_TOOL_NAME, FinalAnswerArguments


@dataclass
class MappedEvent:
    """One normalized SDK event, split by what the runner should do with each piece.

    A single raw event can produce up to three things: a context item to append, an
    output item to emit on the bus, and/or a streaming text delta. Any may be None.
    """

    context_item: AgentContextItem | None = None
    output_item: AgentOutputItem | None = None
    delta: AgentTextDelta | None = None
    refusal_text: str | None = None


class OpenAiEventMapper:
    """Normalizes OpenAI Agents SDK stream events into Engine context/output/delta items.

    Owns the boundary between the SDK's internal event shapes and the Engine's typed
    AgentContextItem / AgentOutputItem / AgentTextDelta. Detects root-agent
    finalization — a ``final_answer`` tool call — and marks the corresponding
    output item ``final=True``.

    Stateful only across one agent's stream: holds the call_id→tool_name
    map so a ``ToolCallOutputItem`` can carry the function name through to
    ``AgentMessage.name``. The SDK does not expose ``tool_name`` on the
    output item (the canonical Responses-API ``FunctionCallOutput`` shape
    has no ``name`` field), but the preceding ``ToolCallItem`` does, so we
    remember it from there. Compaction summaries and Chat-Completions
    replay both read ``item.name`` and would otherwise see ``None``.
    """

    def __init__(self) -> None:
        self._tool_names_by_call_id: dict[str, str] = {}
        # call_ids of root ``final_answer`` calls whose *output* was emitted
        # as a final-flagged assistant message. The SDK still executes the
        # tool (``StopAtTools`` runs it, then stops); the matching
        # ``ToolCallOutputItem`` is kept in context (the call/result pair
        # must stay intact for resends) but suppressed from the output bus —
        # the transformed final message already delivered the answer.
        self._final_answer_call_ids: set[str] = set()

    def to_mapped_event(
        self,
        raw_event: StreamEvent,
        *,
        execution: AgentExecution,
        is_root: bool,
    ) -> MappedEvent:
        """Dispatch an SDK event to the right sub-mapper; unknown shapes are dropped."""
        if isinstance(raw_event, RawResponsesStreamEvent):
            return self._map_raw_delta(raw_event, execution=execution)

        if isinstance(raw_event, RunItemStreamEvent):
            item = raw_event.item
            if isinstance(item, MessageOutputItem):
                return self._map_assistant_message(item, execution=execution)
            if isinstance(item, ToolCallItem):
                return self._map_tool_call(item, execution=execution, is_root=is_root)
            if isinstance(item, ToolCallOutputItem):
                return self._map_tool_output(item, execution=execution)

        return MappedEvent()

    def _map_raw_delta(
        self, raw_event: RawResponsesStreamEvent, *, execution: AgentExecution
    ) -> MappedEvent:
        """Extract a streaming text delta from a Responses-API ``response.output_text.delta`` event."""
        data = raw_event.data
        if getattr(data, "type", None) != "response.output_text.delta":
            return MappedEvent()
        delta = AgentTextDelta(
            sequence=0,
            agent_id=execution.agent_id,
            parent_agent_id=execution.parent_agent_id,
            parent_tool_call_id=execution.parent_tool_call_id,
            depth=execution.depth,
            item_id=str(getattr(data, "item_id", "")),
            text_delta=str(getattr(data, "delta", "")),
        )
        return MappedEvent(delta=delta)

    def _map_assistant_message(
        self, item: MessageOutputItem, *, execution: AgentExecution
    ) -> MappedEvent:
        """Build the assistant ``AgentMessage`` from a ``ResponseOutputMessage``.

        Never final: root finalization only happens through the
        ``final_answer`` tool call (see ``_map_final_answer_call``).
        """
        raw_item = item.raw_item
        item_id = raw_item.id
        parts = raw_item.content
        text = "".join(part.text for part in parts if isinstance(part, ResponseOutputText))
        refusal_text = _extract_refusal_text(parts=parts, text=text)
        if refusal_text is not None:
            return MappedEvent(refusal_text=refusal_text)

        content: str | None = text or None
        context_item = AgentContextItem(
            item_id=item_id,
            role="assistant",
            content=content,
            tool_calls=None,
            agent_id=execution.agent_id,
            parent_agent_id=execution.parent_agent_id,
            parent_tool_call_id=execution.parent_tool_call_id,
        )
        output_item = AgentOutputItem(
            sequence=0,
            agent_id=execution.agent_id,
            parent_agent_id=execution.parent_agent_id,
            parent_tool_call_id=execution.parent_tool_call_id,
            agent_name=execution.agent_name,
            depth=execution.depth,
            item=AgentMessage(role="assistant", content=content, tool_calls=None),
        )
        return MappedEvent(context_item=context_item, output_item=output_item)

    def _map_tool_call(
        self, item: ToolCallItem, *, execution: AgentExecution, is_root: bool
    ) -> MappedEvent:
        """Project a ``ToolCallItem`` into the engine's assistant-with-tool_calls shape.

        Uses the SDK's ``call_id`` / ``tool_name`` properties (added in 0.14.6),
        which transparently handle both the Pydantic and dict forms of
        ``raw_item`` so the mapper does not need its own normalization step.
        ``arguments`` is the only field the SDK doesn't expose as a property,
        so it's read directly off ``raw_item`` with a single shape check.

        A root ``final_answer`` call with a parseable non-empty ``answer`` is
        handled by ``_map_final_answer_call``: the *output* item becomes a plain
        final-flagged assistant message carrying the answer text, while the
        *context* item stays a genuine tool call so the rendered messages array
        remains valid on resend. A malformed/empty ``final_answer`` call falls
        through to normal tool-call mapping — no ``final`` flag is set, and the
        runner's finalization reprompt handles recovery.
        """
        call_id = item.call_id or ""
        name = item.tool_name or ""
        if is_root and name == FINAL_ANSWER_TOOL_NAME:
            mapped = self._map_final_answer_call(item, execution=execution, call_id=call_id)
            if mapped is not None:
                return mapped
        # Remember the name so the matching ``_map_tool_output`` can fill
        # in ``AgentContextItem.name``. Compactor and Chat-Completions
        # replay both read that field; the SDK doesn't surface it on the
        # output item, so the call → output correlation has to live here.
        if call_id and name:
            self._tool_names_by_call_id[call_id] = name
        arguments = _read_arguments(item)
        tc = AgentToolCall(
            id=call_id,
            function=AgentToolFunction(name=name, arguments=arguments),
        )
        # Synthetic item_id keyed by call_id keeps the tool_call entry
        # distinct from its tool_output entry in ``AgentContext._index``,
        # which is keyed by ``item_id`` and would otherwise let the second
        # append silently overwrite the first.
        item_id = f"tool-call-{call_id}"
        context_item = AgentContextItem(
            item_id=item_id,
            role="assistant",
            content=None,
            tool_calls=[tc],
            agent_id=execution.agent_id,
            parent_agent_id=execution.parent_agent_id,
            parent_tool_call_id=execution.parent_tool_call_id,
        )
        output_item = AgentOutputItem(
            sequence=0,
            agent_id=execution.agent_id,
            parent_agent_id=execution.parent_agent_id,
            parent_tool_call_id=execution.parent_tool_call_id,
            agent_name=execution.agent_name,
            depth=execution.depth,
            item=AgentMessage(role="assistant", content=None, tool_calls=[tc]),
        )
        return MappedEvent(context_item=context_item, output_item=output_item)

    def _map_final_answer_call(
        self, item: ToolCallItem, *, execution: AgentExecution, call_id: str
    ) -> MappedEvent | None:
        """Map a root ``final_answer`` call: real tool call in context, final message on the bus.

        The tool call is the run-termination protocol, not real tool work: its
        ``answer`` argument IS the final answer. The *output* item is a plain
        assistant message with ``final=True`` so consumers of
        ``AgentOutputItem.final`` (runtime shims, transcript stores) receive
        the answer as a simple final message.

        The *context* item stays the genuine tool call, and ``_map_tool_output``
        keeps its acknowledgement result in context too. With
        ``parallel_tool_calls`` the model may batch ``final_answer`` with other
        tool calls in one turn; splicing a plain assistant message between
        another call and its result would corrupt the rendered messages array
        on a mid-stream-failure resend. Keeping the call/result pair intact
        makes every resend a valid conversation regardless of batch order.

        Returns ``None`` when ``arguments`` don't parse or ``answer`` is
        empty/whitespace — the caller then maps the call normally (unflagged),
        leaving recovery to the runner's finalization reprompt.
        """
        raw_arguments = _read_arguments(item)
        try:
            parsed = FinalAnswerArguments.model_validate_json(raw_arguments or "{}")
        except ValidationError:
            return None
        answer = parsed.answer.strip()
        if not answer:
            return None
        if call_id:
            self._final_answer_call_ids.add(call_id)
            self._tool_names_by_call_id[call_id] = FINAL_ANSWER_TOOL_NAME
        tc = AgentToolCall(
            id=call_id,
            function=AgentToolFunction(name=FINAL_ANSWER_TOOL_NAME, arguments=raw_arguments),
        )
        context_item = AgentContextItem(
            item_id=f"tool-call-{call_id}",
            role="assistant",
            content=None,
            tool_calls=[tc],
            agent_id=execution.agent_id,
            parent_agent_id=execution.parent_agent_id,
            parent_tool_call_id=execution.parent_tool_call_id,
        )
        output_item = AgentOutputItem(
            sequence=0,
            agent_id=execution.agent_id,
            parent_agent_id=execution.parent_agent_id,
            parent_tool_call_id=execution.parent_tool_call_id,
            agent_name=execution.agent_name,
            depth=execution.depth,
            item=AgentMessage(role="assistant", content=answer, tool_calls=None),
            final=True,
        )
        return MappedEvent(context_item=context_item, output_item=output_item)

    def _map_tool_output(
        self, item: ToolCallOutputItem, *, execution: AgentExecution
    ) -> MappedEvent:
        """Project a ``ToolCallOutputItem`` into the engine's tool-role message shape.

        Reads SDK-exposed surfaces (``item.call_id``, ``item.output``) plus
        the call_id→name map populated by the preceding ``_map_tool_call``.
        The OpenAI Responses-API ``FunctionCallOutput`` shape has no
        ``name`` field, so without that correlation the compaction summary
        and Chat-Completions replay both lose the function name.
        """
        call_id = item.call_id or ""
        suppress_output = bool(call_id) and call_id in self._final_answer_call_ids
        if suppress_output:
            self._final_answer_call_ids.discard(call_id)
        content = "" if item.output is None else str(item.output)
        name = self._tool_names_by_call_id.pop(call_id, None) if call_id else None
        item_id = f"tool-result-{call_id}"
        context_item = AgentContextItem(
            item_id=item_id,
            role="tool",
            content=content,
            tool_call_id=call_id,
            name=name,
            agent_id=execution.agent_id,
            parent_agent_id=execution.parent_agent_id,
            parent_tool_call_id=execution.parent_tool_call_id,
        )
        if suppress_output:
            # The transformed final message already delivered the answer on
            # the bus; the acknowledgement stays context-only so the resent
            # messages array keeps the final_answer call/result pair intact.
            return MappedEvent(context_item=context_item)
        output_item = AgentOutputItem(
            sequence=0,
            agent_id=execution.agent_id,
            parent_agent_id=execution.parent_agent_id,
            parent_tool_call_id=execution.parent_tool_call_id,
            agent_name=execution.agent_name,
            depth=execution.depth,
            item=AgentMessage(
                role="tool",
                content=content,
                tool_call_id=call_id,
                name=name,
            ),
        )
        return MappedEvent(context_item=context_item, output_item=output_item)


def _read_arguments(item: ToolCallItem) -> str:
    """Pull the JSON ``arguments`` field off a function-call raw item.

    The SDK does not expose a property for this — only ``call_id`` and
    ``tool_name`` got first-class accessors in 0.14.6. ``raw_item`` is a
    union of multiple OpenAI tool-call types (function, computer, web
    search, ...) plus ``dict[str, Any]``. We only register function
    tools, so in practice the raw item is a ``ResponseFunctionToolCall``
    or its dict form; both expose ``arguments`` as a JSON string. The
    single shape check below is the one place the dual-form union leaks
    through to the mapper.
    """
    raw = item.raw_item
    if isinstance(raw, dict):
        return str(raw.get("arguments") or "")
    return str(getattr(raw, "arguments", "") or "")


_TEXT_REFUSAL_PREFIXES = (
    "i'm sorry, but i cannot assist with that request",
    "i’m sorry, but i cannot assist with that request",
    "i am sorry, but i cannot assist with that request",
    "sorry, but i cannot assist with that request",
)


def _extract_refusal_text(
    *, parts: list[ResponseOutputText | ResponseOutputRefusal], text: str
) -> str | None:
    refusal_parts = [
        part.refusal.strip()
        for part in parts
        if isinstance(part, ResponseOutputRefusal) and part.refusal.strip()
    ]
    if refusal_parts:
        return "\n".join(refusal_parts)

    normalized = " ".join(text.strip().lower().split())
    if any(normalized.startswith(prefix) for prefix in _TEXT_REFUSAL_PREFIXES):
        return text.strip()
    return None
