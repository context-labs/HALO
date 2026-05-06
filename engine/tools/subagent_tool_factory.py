from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from agents import Agent, FunctionTool, RunContextWrapper, Tool
from agents.agent_tool_input import AgentAsToolInput
from agents.tool_context import ToolContext as SdkToolContext

from engine.agents.agent_context import AgentContext
from engine.agents.agent_context_items import AgentContextItem
from engine.agents.agent_execution import AgentExecution
from engine.agents.compactor import build_compactor_factory
from engine.agents.engine_run_state import EngineRunState
from engine.agents.openai_agent_runner import OpenAiAgentRunner
from engine.agents.prompt_templates import render_subagent_system_prompt
from engine.errors import EngineAgentExhaustedError, EngineMaxDepthExceededError
from engine.telemetry.tracing import halo_agent_span
from engine.tools.agent_context_tools import GetContextItemTool
from engine.tools.run_code_tool import RunCodeTool
from engine.tools.subagent_result import SubagentToolResult
from engine.tools.synthesis_tool import SynthesisTool
from engine.tools.tool_protocol import ToolContext, to_sdk_function_tool
from engine.tools.trace_tools import (
    CountTracesTool,
    GetDatasetOverviewTool,
    QueryTracesTool,
    SearchSpanTool,
    SearchTraceTool,
    ViewSpansTool,
    ViewTraceTool,
)

logger = logging.getLogger(__name__)


def build_root_sdk_agent(
    *,
    engine_config,
    run_state: EngineRunState,
    agent_execution: AgentExecution,
    agent_context: AgentContext,
) -> Agent[EngineRunState]:
    """Construct the root SDK Agent wired with all leaf tools plus a depth-aware ``call_subagent``.

    Subagent spawning is gated by a *per-depth* ``asyncio.Semaphore`` sized to
    ``maximum_parallel_subagents``. Each depth has its own pool, so a parent
    holding a depth-N slot while it waits for a depth-(N+1) grandchild
    cannot block that grandchild — they contend on different semaphores.

    A single shared semaphore would deadlock at default config: with
    ``maximum_parallel_subagents`` parents holding every depth-N slot,
    every depth-(N+1) grandchild would block forever waiting for a slot
    its parent is holding.
    """
    semaphores_by_depth = build_subagent_semaphores(engine_config)
    tools = _child_tools_for_depth(
        depth=0,
        run_state=run_state,
        semaphores_by_depth=semaphores_by_depth,
        parent_execution=agent_execution,
        parent_context=agent_context,
    )

    return Agent[EngineRunState](
        name=engine_config.root_agent.name,
        instructions="",
        model=engine_config.root_agent.model.name,
        model_settings=engine_config.root_agent.model.to_sdk_model_settings(),
        tools=tools,
    )


def build_subagent_semaphores(engine_config) -> dict[int, asyncio.Semaphore]:
    """Build one ``asyncio.Semaphore`` per spawnable depth.

    Keys are depths ``1..maximum_depth`` (the depths a subagent can be spawned
    at — depth 0 is the root and is never gated). Each semaphore is sized to
    ``maximum_parallel_subagents``.
    """
    return {
        d: asyncio.Semaphore(engine_config.maximum_parallel_subagents)
        for d in range(1, engine_config.maximum_depth + 1)
    }


def _child_tools_for_depth(
    *,
    depth: int,
    run_state: EngineRunState,
    semaphores_by_depth: dict[int, asyncio.Semaphore],
    parent_execution: AgentExecution,
    parent_context: AgentContext,
) -> list[Tool]:
    """Return the tool list available to an agent at ``depth``.

    Always includes the leaf trace/synthesis/run_code/get_context tools. A
    ``call_subagent`` tool is appended only when ``depth < maximum_depth``, which
    is how depth enforcement is wired structurally rather than via runtime checks.

    ``parent_context`` is the AgentContext of the agent that owns these tools;
    it's plumbed into ``ToolContext.agent_context`` so tools like
    ``get_context_item`` can look up that agent's stored items.
    """
    engine_config = run_state.config

    def make_ctx(wrapper: RunContextWrapper[Any]) -> ToolContext:
        return ToolContext.model_construct(
            run_state=run_state,
            trace_store=run_state.trace_store,
            output_bus=run_state.output_bus,
            agent_context=parent_context,
            sandbox=run_state.sandbox,
        )

    leaf_tools: list[Tool] = [
        to_sdk_function_tool(GetDatasetOverviewTool(), context_factory=make_ctx),
        to_sdk_function_tool(QueryTracesTool(), context_factory=make_ctx),
        to_sdk_function_tool(CountTracesTool(), context_factory=make_ctx),
        to_sdk_function_tool(ViewTraceTool(), context_factory=make_ctx),
        to_sdk_function_tool(ViewSpansTool(), context_factory=make_ctx),
        to_sdk_function_tool(SearchTraceTool(), context_factory=make_ctx),
        to_sdk_function_tool(SearchSpanTool(), context_factory=make_ctx),
        to_sdk_function_tool(GetContextItemTool(), context_factory=make_ctx),
        to_sdk_function_tool(
            SynthesisTool(
                model=engine_config.synthesis_model,
                model_provider=engine_config.model_provider,
            ),
            context_factory=make_ctx,
        ),
    ]

    if run_state.sandbox is not None:
        leaf_tools.append(to_sdk_function_tool(RunCodeTool(), context_factory=make_ctx))

    if depth >= engine_config.maximum_depth:
        return leaf_tools

    subagent_tool = _build_subagent_as_tool(
        run_state=run_state,
        child_depth=depth + 1,
        semaphores_by_depth=semaphores_by_depth,
        parent_execution=parent_execution,
    )
    return leaf_tools + [subagent_tool]


def _build_subagent_as_tool(
    *,
    run_state: EngineRunState,
    child_depth: int,
    semaphores_by_depth: dict[int, asyncio.Semaphore],
    parent_execution: AgentExecution,
) -> FunctionTool:
    """Wrap a fresh subagent as an SDK FunctionTool with depth gating, bus streaming, and a typed result.

    Uses ``Agent.as_tool`` for the schema, then overrides ``on_invoke_tool`` so each
    invocation builds its own child Agent + AgentContext + AgentExecution and streams
    child events through the shared bus. Returns a JSON-serialized
    ``SubagentToolResult`` to the parent on success or failure.
    """
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
        model_settings=engine_config.subagent.model.to_sdk_model_settings(),
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
        """SDK-side tool entrypoint for ``call_subagent``: gate, semaphore-acquire, run, return result."""
        # Defense-in-depth: ``_child_tools_for_depth`` already gates this structurally;
        # keep the runtime check so a future refactor can't silently re-enable recursion.
        if child_depth > engine_config.maximum_depth:
            raise EngineMaxDepthExceededError(
                f"subagent invoked at depth={child_depth} > maximum_depth={engine_config.maximum_depth}"
            )

        # ``as_tool()`` builds a tool whose ``raw_arguments`` is JSON-encoded
        # ``AgentAsToolInput`` (i.e. ``{"input": "..."}``); the SDK's own
        # ``_run_agent_impl`` extracts ``params["input"]`` before calling the
        # nested agent, so we mirror that here. Without this, the subagent
        # would see the raw JSON wrapper instead of the delegated question.
        delegated_input = AgentAsToolInput.model_validate_json(raw_arguments).input

        async with semaphores_by_depth[child_depth]:
            child_execution = AgentExecution(
                agent_id=f"sub-{uuid.uuid4().hex[:8]}",
                agent_name=engine_config.subagent.name,
                depth=child_depth,
                parent_agent_id=parent_execution.agent_id,
                parent_tool_call_id=ctx.tool_call_id,
            )
            run_state.register(child_execution)

            child_context = AgentContext(
                items=[
                    AgentContextItem(
                        item_id="sys-0", role="system", content=subagent_system_prompt
                    ),
                    AgentContextItem(item_id="in-0", role="user", content=delegated_input),
                ],
                compaction_model=engine_config.compaction_model,
                text_message_compaction_keep_last_messages=engine_config.text_message_compaction_keep_last_messages,
                tool_call_compaction_keep_last_turns=engine_config.tool_call_compaction_keep_last_turns,
            )

            child_agent = Agent[EngineRunState](
                name=engine_config.subagent.name,
                instructions="",
                model=engine_config.subagent.model.name,
                model_settings=engine_config.subagent.model.to_sdk_model_settings(),
                tools=_child_tools_for_depth(
                    depth=child_depth,
                    run_state=run_state,
                    semaphores_by_depth=semaphores_by_depth,
                    parent_execution=child_execution,
                    parent_context=child_context,
                ),
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
                compactor_factory=build_compactor_factory(engine_config),
            )

            with halo_agent_span(name=engine_config.subagent.name, system="openai"):
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
                        "subagent %s exhausted retries at depth=%s: %s",
                        child_execution.agent_id,
                        child_depth,
                        exc,
                    )
                    return _failure_result(child_execution, f"Subagent exhausted retries: {exc}")
                except Exception as exc:
                    logger.warning(
                        "subagent %s failed at depth=%s: %s: %s",
                        child_execution.agent_id,
                        child_depth,
                        type(exc).__name__,
                        exc,
                    )
                    return _failure_result(
                        child_execution, f"Subagent failed: {type(exc).__name__}: {exc}"
                    )

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
    """Build the JSON SubagentToolResult returned to the parent when a subagent fails recoverably."""
    return SubagentToolResult(
        child_agent_id=execution.agent_id,
        answer=message,
        output_start_sequence=execution.output_start_sequence or 0,
        output_end_sequence=execution.output_end_sequence or 0,
        turns_used=execution.turns_used,
        tool_calls_made=execution.tool_calls_made,
    ).model_dump_json()
