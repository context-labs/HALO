from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from agents import Agent, FunctionTool, RunContextWrapper
from agents.tool_context import ToolContext as SdkToolContext
from loguru import logger

from engine.agents.agent_context import AgentContext
from engine.agents.agent_context_items import AgentContextItem
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_run_state import EngineRunState
from engine.agents.openai_agent_runner import OpenAiAgentRunner
from engine.agents.openai_compactor import build_openai_compactor_factory
from engine.agents.prompt_templates import render_subagent_system_prompt
from engine.errors import EngineAgentExhaustedError, EngineMaxDepthExceededError
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
    tools = _child_tools_for_depth(
        depth=0,
        run_state=run_state,
        semaphore=semaphore,
        parent_execution=agent_execution,
    )

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
    semaphore: asyncio.Semaphore,
    parent_execution: AgentExecution,
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

    subagent_tool = _build_subagent_as_tool(
        run_state=run_state,
        child_depth=depth + 1,
        semaphore=semaphore,
        parent_execution=parent_execution,
    )
    return leaf_tools + [subagent_tool]


def _build_subagent_as_tool(
    *,
    run_state: EngineRunState,
    child_depth: int,
    semaphore: asyncio.Semaphore,
    parent_execution: AgentExecution,
) -> FunctionTool:
    engine_config = run_state.config
    subagent_system_prompt = render_subagent_system_prompt(
        instructions=engine_config.subagent.instructions,
        depth=child_depth,
        maximum_depth=engine_config.maximum_depth,
        maximum_parallel_subagents=engine_config.maximum_parallel_subagents,
    )
    # ``as_tool()``'s schema is fixed (``AgentAsToolInput`` shape) and does not
    # depend on the wrapped agent's tool list, so this stub Agent is enough to
    # produce the FunctionTool wrapper. The real child Agent is rebuilt per
    # invocation inside ``guarded_invoke`` so each invocation can pass its own
    # ``child_execution`` as ``parent_execution`` to grandchildren.
    stub_child_agent = Agent[EngineRunState](
        name=engine_config.subagent.name,
        instructions="",
        model=engine_config.subagent.model.name,
        tools=[],
    )

    sdk_tool = stub_child_agent.as_tool(
        tool_name="call_subagent",
        tool_description="Delegate a focused question to a subagent. Returns the subagent's answer.",
    )

    # Annotating ``ctx`` as ``SdkToolContext`` (rather than the SDK's narrower
    # ``RunContextWrapper``) is load-bearing: the SDK's tool dispatcher inspects
    # this annotation in ``agents/tool.py:_get_function_tool_invoke_context``
    # and only passes the rich context (with ``tool_call_id``) when it sees
    # ``ToolContext``. With ``RunContextWrapper`` the SDK forks down to a bare
    # wrapper and ``tool_call_id`` is lost.
    async def guarded_invoke(ctx: SdkToolContext[Any], raw_arguments: str) -> str:
        if child_depth > engine_config.maximum_depth:
            raise EngineMaxDepthExceededError(
                f"subagent invoked at depth={child_depth} > maximum_depth={engine_config.maximum_depth}"
            )

        async with semaphore:
            child_execution = AgentExecution(
                agent_id=f"sub-{uuid.uuid4().hex[:8]}",
                agent_name=engine_config.subagent.name,
                depth=child_depth,
                parent_agent_id=parent_execution.agent_id,
                parent_tool_call_id=ctx.tool_call_id,
            )
            run_state.register(child_execution)

            child_agent = Agent[EngineRunState](
                name=engine_config.subagent.name,
                instructions="",
                model=engine_config.subagent.model.name,
                tools=_child_tools_for_depth(
                    depth=child_depth,
                    run_state=run_state,
                    semaphore=semaphore,
                    parent_execution=child_execution,
                ),
            )

            child_context = AgentContext(
                items=[
                    AgentContextItem(item_id="sys-0", role="system", content=subagent_system_prompt),
                    AgentContextItem(item_id="in-0", role="user", content=raw_arguments),
                ],
                compaction_model=engine_config.compaction_model,
                text_message_compaction_keep_last_messages=engine_config.text_message_compaction_keep_last_messages,
                tool_call_compaction_keep_last_messages=engine_config.tool_call_compaction_keep_last_messages,
            )

            async def _run_streamed(*, agent, input, context):
                return run_state.runner.run_streamed(
                    starting_agent=agent,
                    input=input,
                    context=context,
                    max_turns=engine_config.subagent.maximum_turns,
                )

            runner = OpenAiAgentRunner(
                run_streamed=_run_streamed,
                compactor_factory=build_openai_compactor_factory(engine_config),
            )

            try:
                await runner.run(
                    sdk_agent=child_agent,
                    agent_context=child_context,
                    agent_execution=child_execution,
                    output_bus=run_state.output_bus,
                    is_root=False,
                    run_context=run_state,
                )
            except EngineAgentExhaustedError as exc:
                logger.warning(
                    "subagent {} exhausted retries at depth={}: {}",
                    child_execution.agent_id, child_depth, exc,
                )
                return _failure_result(child_execution, f"Subagent exhausted retries: {exc}")
            except Exception as exc:
                logger.warning(
                    "subagent {} failed at depth={}: {}: {}",
                    child_execution.agent_id, child_depth,
                    type(exc).__name__, exc,
                )
                return _failure_result(child_execution, f"Subagent failed: {type(exc).__name__}: {exc}")

            answer = _extract_final_answer(child_context)
            result = SubagentToolResult(
                child_agent_id=child_execution.agent_id,
                answer=answer,
                output_start_sequence=child_execution.output_start_sequence or 0,
                output_end_sequence=child_execution.output_end_sequence or 0,
                turns_used=child_execution.turns_used,
                tool_calls_made=child_execution.tool_calls_made,
            )
            return result.model_dump_json()

    sdk_tool.on_invoke_tool = guarded_invoke
    return sdk_tool


def _extract_final_answer(context: AgentContext) -> str:
    """Walk the context backwards and return the last assistant text message."""
    for item in reversed(context.items):
        if item.role != "assistant" or item.tool_calls:
            continue
        if isinstance(item.content, str) and item.content.strip():
            return item.content.strip()
    return ""


def _failure_result(execution: AgentExecution, message: str) -> str:
    return SubagentToolResult(
        child_agent_id=execution.agent_id,
        answer=message,
        output_start_sequence=execution.output_start_sequence or 0,
        output_end_sequence=execution.output_end_sequence or 0,
        turns_used=execution.turns_used,
        tool_calls_made=execution.tool_calls_made,
    ).model_dump_json()


def _wrap_with_semaphore(
    fn: Callable[..., Awaitable[Any]],
    semaphore: asyncio.Semaphore,
) -> Callable[..., Awaitable[Any]]:
    async def wrapped(*args, **kwargs):
        async with semaphore:
            return await fn(*args, **kwargs)
    return wrapped
