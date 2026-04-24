from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.agents.agent_context_items import AgentContextItem
from engine.agents.agent_execution import AgentExecution
from engine.agents.prompt_templates import FINAL_SENTINEL
from engine.models.engine_output import AgentOutputItem, AgentTextDelta
from engine.models.messages import AgentMessage, AgentToolCall, AgentToolFunction


@dataclass
class MappedEvent:
    context_item: AgentContextItem | None = None
    output_item: AgentOutputItem | None = None
    delta: AgentTextDelta | None = None


class OpenAiEventMapper:
    def to_mapped_event(
        self,
        raw_event: Any,
        *,
        execution: AgentExecution,
        is_root: bool,
    ) -> MappedEvent:
        kind = getattr(raw_event, "type", None)

        if kind == "raw_response_event":
            return self._map_raw_delta(raw_event, execution=execution)

        if kind == "run_item_stream_event":
            item = getattr(raw_event, "item", None)
            if item is None:
                return MappedEvent()
            item_kind = getattr(item, "type", None)
            if item_kind == "message_output_item":
                return self._map_assistant_message(item, execution=execution, is_root=is_root)
            if item_kind == "tool_call_item":
                return self._map_tool_call(item, execution=execution)
            if item_kind == "tool_call_output_item":
                return self._map_tool_output(item, execution=execution)
            return MappedEvent()

        # Legacy direct-item test shape (kept for existing unit tests).
        if kind == "message_output_item":
            return self._map_assistant_message_legacy(raw_event, execution=execution, is_root=is_root)
        if kind == "tool_call_item":
            return self._map_tool_call_legacy(raw_event, execution=execution)
        if kind == "tool_call_output_item":
            return self._map_tool_output_legacy(raw_event, execution=execution)

        return MappedEvent()

    def _map_raw_delta(self, raw: Any, *, execution: AgentExecution) -> MappedEvent:
        data = getattr(raw, "data", None)
        if data is None:
            return MappedEvent()
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

    def _extract_assistant_text(self, raw_item: Any) -> tuple[str, str]:
        # raw_item is openai.types.responses.ResponseOutputMessage
        item_id = str(getattr(raw_item, "id", "") or "")
        parts = getattr(raw_item, "content", None) or []
        text_parts = [
            getattr(p, "text", "")
            for p in parts
            if getattr(p, "type", None) in ("output_text", "text")
        ]
        return item_id, "".join(text_parts)

    def _build_assistant(
        self,
        *,
        execution: AgentExecution,
        is_root: bool,
        item_id: str,
        text: str,
        tool_calls: list[AgentToolCall] | None,
    ) -> MappedEvent:
        final = False
        if is_root and text and FINAL_SENTINEL in text:
            final = True
            text = text.replace(FINAL_SENTINEL, "").rstrip()

        content: str | None = text or None
        context_item = AgentContextItem(
            item_id=item_id,
            role="assistant",
            content=content,
            tool_calls=tool_calls,
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
            item=AgentMessage(role="assistant", content=content, tool_calls=tool_calls),
            final=final,
        )
        return MappedEvent(context_item=context_item, output_item=output_item)

    def _map_assistant_message(
        self,
        item: Any,
        *,
        execution: AgentExecution,
        is_root: bool,
    ) -> MappedEvent:
        raw_item = getattr(item, "raw_item", item)
        item_id, text = self._extract_assistant_text(raw_item)
        return self._build_assistant(
            execution=execution,
            is_root=is_root,
            item_id=item_id,
            text=text,
            tool_calls=None,
        )

    def _map_tool_call(self, item: Any, *, execution: AgentExecution) -> MappedEvent:
        # item.raw_item is a ResponseFunctionToolCall
        raw = getattr(item, "raw_item", item)
        call_id = str(getattr(raw, "call_id", "") or getattr(raw, "id", "") or "")
        tc = AgentToolCall(
            id=call_id,
            function=AgentToolFunction(
                name=str(getattr(raw, "name", "") or ""),
                arguments=str(getattr(raw, "arguments", "") or ""),
            ),
        )
        item_id = str(getattr(raw, "id", "") or call_id)
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

    def _map_tool_output(self, item: Any, *, execution: AgentExecution) -> MappedEvent:
        raw = getattr(item, "raw_item", item)
        call_id = str(getattr(raw, "call_id", "") or "")
        output = getattr(item, "output", None)
        if output is None:
            output = getattr(raw, "output", "")
        content = str(output)
        name = getattr(raw, "name", None) or getattr(item, "name", None)
        item_id = str(getattr(raw, "id", "") or call_id)
        context_item = AgentContextItem(
            item_id=item_id,
            role="tool",
            content=content,
            tool_call_id=call_id,
            name=str(name) if name else None,
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
                name=str(name) if name else None,
            ),
        )
        return MappedEvent(context_item=context_item, output_item=output_item)

    # --- legacy shapes kept alive so the original unit tests still pass ---

    def _map_assistant_message_legacy(
        self,
        raw: Any,
        *,
        execution: AgentExecution,
        is_root: bool,
    ) -> MappedEvent:
        msg = raw.message
        text_parts = [
            getattr(p, "text", "")
            for p in (getattr(msg, "content", None) or [])
            if getattr(p, "type", None) == "output_text"
        ]
        text = "".join(text_parts)
        item_id = str(getattr(msg, "id", ""))
        return self._build_assistant(
            execution=execution,
            is_root=is_root,
            item_id=item_id,
            text=text,
            tool_calls=None,
        )

    def _map_tool_call_legacy(self, raw: Any, *, execution: AgentExecution) -> MappedEvent:
        call = raw.tool_call
        tc = AgentToolCall(
            id=str(call.id),
            function=AgentToolFunction(
                name=str(call.function.name),
                arguments=str(call.function.arguments),
            ),
        )
        item_id = str(getattr(raw, "message_id", call.id))
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

    def _map_tool_output_legacy(self, raw: Any, *, execution: AgentExecution) -> MappedEvent:
        item_id = str(getattr(raw, "message_id", raw.tool_call_id))
        content = str(raw.output)
        context_item = AgentContextItem(
            item_id=item_id,
            role="tool",
            content=content,
            tool_call_id=str(raw.tool_call_id),
            name=str(getattr(raw, "name", "")) or None,
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
                tool_call_id=str(raw.tool_call_id),
                name=str(getattr(raw, "name", "")) or None,
            ),
        )
        return MappedEvent(context_item=context_item, output_item=output_item)
