from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from agents import Agent, FunctionTool, RunContextWrapper, Runner
from loguru import logger

from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_run_state import EngineRunState
from engine.agents.openai_event_mapper import OpenAiEventMapper
from engine.models.engine_output import AgentOutputItem
from engine.agents.prompt_templates import render_subagent_system_prompt
from engine.errors import EngineMaxDepthExceededError
from engine.tools.agent_context_tools import GetContextItemTool
from engine.tools.run_code_tool import RunCodeTool
from engine.tools.subagent_result import SubagentToolResult
from engine.tools.synthesis_tool import SynthesisTool
from engine.tools.tool_protocol import ToolContext, to_sdk_function_tool
from engine.tools.trace_tools import (
    CountTracesTool,
    GetDatasetOverviewTool,
    QueryTracesTool,
    SearchTraceTool,
    ViewTraceTool,
)


def build_root_sdk_agent(
    *,
    engine_config,
    run_state: EngineRunState,
    agent_execution: AgentExecution,
) -> Agent[EngineRunState]:
    semaphore = asyncio.Semaphore(engine_config.maximum_parallel_subagents)
    tools = _child_tools_for_depth(depth=0, run_state=run_state, semaphore=semaphore)

    return Agent[EngineRunState](
        name=engine_config.root_agent.name,
        instructions="",
        model=engine_config.root_agent.model.name,
        tools=tools,
    )


def _child_tools_for_depth(
    *,
    depth: int,
    run_state: EngineRunState,
    semaphore: asyncio.Semaphore | None,
) -> list[FunctionTool]:
    engine_config = run_state.config

    def make_ctx(wrapper: RunContextWrapper[Any]) -> ToolContext:
        return ToolContext.model_construct(
            run_state=run_state,
            trace_store=run_state.trace_store,
            output_bus=run_state.output_bus,
        )

    leaf_tools: list[FunctionTool] = [
        to_sdk_function_tool(GetDatasetOverviewTool(), context_factory=make_ctx),
        to_sdk_function_tool(QueryTracesTool(), context_factory=make_ctx),
        to_sdk_function_tool(CountTracesTool(), context_factory=make_ctx),
        to_sdk_function_tool(ViewTraceTool(), context_factory=make_ctx),
        to_sdk_function_tool(SearchTraceTool(), context_factory=make_ctx),
        to_sdk_function_tool(GetContextItemTool(), context_factory=make_ctx),
        to_sdk_function_tool(
            SynthesisTool(model_name=engine_config.synthesis_model.name),
            context_factory=make_ctx,
        ),
        to_sdk_function_tool(RunCodeTool(sandbox_config=engine_config.sandbox), context_factory=make_ctx),
    ]

    if depth >= engine_config.maximum_depth:
        return leaf_tools

    assert semaphore is not None, "semaphore required when sub-depth subagent tool is built"
    subagent_tool = _build_subagent_as_tool(
        run_state=run_state,
        child_depth=depth + 1,
        semaphore=semaphore,
    )
    return leaf_tools + [subagent_tool]


def _build_subagent_as_tool(
    *,
    run_state: EngineRunState,
    child_depth: int,
    semaphore: asyncio.Semaphore,
) -> FunctionTool:
    engine_config = run_state.config
    child_agent = Agent[EngineRunState](
        name=engine_config.subagent.name,
        instructions=render_subagent_system_prompt(
            instructions=engine_config.subagent.instructions,
            depth=child_depth,
            maximum_depth=engine_config.maximum_depth,
            maximum_parallel_subagents=engine_config.maximum_parallel_subagents,
        ),
        model=engine_config.subagent.model.name,
        tools=_child_tools_for_depth(depth=child_depth, run_state=run_state, semaphore=semaphore),
    )

    mapper = OpenAiEventMapper()
    output_bus = run_state.output_bus

    async def custom_output_extractor(run_result) -> str:
        final_text = ""
        for item in getattr(run_result, "new_items", []):
            if getattr(item, "type", None) == "message_output_item":
                raw_item = getattr(item, "raw_item", None)
                if raw_item is None:
                    continue
                parts = [
                    getattr(p, "text", "")
                    for p in (getattr(raw_item, "content", None) or [])
                    if getattr(p, "type", None) in ("output_text", "text")
                ]
                text = "".join(parts).strip()
                if text:
                    final_text = text
        return SubagentToolResult(
            child_agent_id="",
            answer=final_text,
            output_start_sequence=0,
            output_end_sequence=0,
            turns_used=0,
            tool_calls_made=0,
        ).model_dump_json()

    sdk_tool = child_agent.as_tool(
        tool_name="call_subagent",
        tool_description="Delegate a focused question to a subagent. Returns the subagent's answer.",
        custom_output_extractor=custom_output_extractor,
    )

    async def guarded_invoke(ctx: RunContextWrapper[Any], raw_arguments: str) -> str:
        if child_depth > engine_config.maximum_depth:
            raise EngineMaxDepthExceededError(
                f"subagent invoked at depth={child_depth} > maximum_depth={engine_config.maximum_depth}"
            )

        async with semaphore:
            child_execution = AgentExecution(
                agent_id=f"sub-{uuid.uuid4().hex[:8]}",
                agent_name=engine_config.subagent.name,
                depth=child_depth,
                parent_agent_id=None,
                parent_tool_call_id=None,
            )
            run_state.register(child_execution)
            start_seq: int | None = None
            end_seq: int | None = None
            try:
                stream = Runner.run_streamed(
                    starting_agent=child_agent,
                    input=raw_arguments,
                    context=run_state,
                )
                async for ev in stream.stream_events():
                    mapped = mapper.to_mapped_event(ev, execution=child_execution, is_root=False)
                    if mapped.output_item is not None:
                        emitted = await output_bus.emit(mapped.output_item)
                        if start_seq is None:
                            start_seq = emitted.sequence
                        end_seq = emitted.sequence
                        # Track turns + tool calls for SubagentToolResult
                        item = mapped.output_item.item
                        if item.role == "assistant":
                            if item.tool_calls:
                                child_execution.tool_calls_made += len(item.tool_calls)
                            else:
                                child_execution.turns_used += 1
                    if mapped.delta is not None:
                        await output_bus.emit(mapped.delta)

                child_execution.output_start_sequence = start_seq
                child_execution.output_end_sequence = end_seq

                run_result = stream
                if hasattr(stream, "wait_for_final_output"):
                    run_result = await stream.wait_for_final_output()

                extracted_json = await custom_output_extractor(run_result)
                result = SubagentToolResult.model_validate_json(extracted_json).model_copy(update={
                    "child_agent_id": child_execution.agent_id,
                    "output_start_sequence": start_seq or 0,
                    "output_end_sequence": end_seq or 0,
                    "turns_used": child_execution.turns_used,
                    "tool_calls_made": child_execution.tool_calls_made,
                })
                return result.model_dump_json()
            except Exception as exc:
                logger.warning(
                    "subagent {} failed at depth={}: {}: {}",
                    child_execution.agent_id, child_depth,
                    type(exc).__name__, exc,
                )
                failure = SubagentToolResult(
                    child_agent_id=child_execution.agent_id,
                    answer=f"Subagent failed: {type(exc).__name__}: {exc}",
                    output_start_sequence=start_seq or 0,
                    output_end_sequence=end_seq or 0,
                    turns_used=child_execution.turns_used,
                    tool_calls_made=child_execution.tool_calls_made,
                )
                return failure.model_dump_json()

    sdk_tool.on_invoke_tool = guarded_invoke
    return sdk_tool


def _wrap_with_semaphore(
    fn: Callable[..., Awaitable[Any]],
    semaphore: asyncio.Semaphore,
) -> Callable[..., Awaitable[Any]]:
    async def wrapped(*args, **kwargs):
        async with semaphore:
            return await fn(*args, **kwargs)
    return wrapped
