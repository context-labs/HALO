from __future__ import annotations

from dataclasses import dataclass

from agents.items import MessageOutputItem, ToolCallItem, ToolCallOutputItem
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent, StreamEvent

from engine.agents.agent_context_items import AgentContextItem
from engine.agents.agent_execution import AgentExecution
from engine.agents.prompt_templates import FINAL_SENTINEL
from engine.models.engine_output import AgentOutputItem, AgentTextDelta
from engine.models.messages import AgentMessage, AgentToolCall, AgentToolFunction


@dataclass
class MappedEvent:
    """One normalized SDK event, split by what the runner should do with each piece.

    A single raw event can produce up to three things: a context item to append, an
    output item to emit on the bus, and/or a streaming text delta. Any may be None.
    """

    context_item: AgentContextItem | None = None
    output_item: AgentOutputItem | None = None
    delta: AgentTextDelta | None = None


class OpenAiEventMapper:
    """Normalizes OpenAI Agents SDK stream events into Engine context/output/delta items.

    Owns the boundary between the SDK's internal event shapes and the Engine's typed
    AgentContextItem / AgentOutputItem / AgentTextDelta. Detects the ``<final/>``
    sentinel on root-agent assistant text and marks the corresponding output item.

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
                return self._map_assistant_message(item, execution=execution, is_root=is_root)
            if isinstance(item, ToolCallItem):
                return self._map_tool_call(item, execution=execution)
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
        self, item: MessageOutputItem, *, execution: AgentExecution, is_root: bool
    ) -> MappedEvent:
        """Build the assistant ``AgentMessage`` from a ``ResponseOutputMessage`` and detect ``<final/>``."""
        raw_item = item.raw_item
        item_id = str(getattr(raw_item, "id", "") or "")
        parts = getattr(raw_item, "content", None) or []
        text = "".join(
            getattr(p, "text", "")
            for p in parts
            if getattr(p, "type", None) in ("output_text", "text")
        )

        final = False
        if is_root and text and FINAL_SENTINEL in text:
            final = True
            text = text.replace(FINAL_SENTINEL, "").rstrip()

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
            final=final,
        )
        return MappedEvent(context_item=context_item, output_item=output_item)

    def _map_tool_call(self, item: ToolCallItem, *, execution: AgentExecution) -> MappedEvent:
        """Project a ``ToolCallItem`` into the engine's assistant-with-tool_calls shape.

        Uses the SDK's ``call_id`` / ``tool_name`` properties (added in 0.14.6),
        which transparently handle both the Pydantic and dict forms of
        ``raw_item`` so the mapper does not need its own normalization step.
        ``arguments`` is the only field the SDK doesn't expose as a property,
        so it's read directly off ``raw_item`` with a single shape check.
        """
        call_id = item.call_id or ""
        name = item.tool_name or ""
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
