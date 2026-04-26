from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from agents import Runner

from engine.agents.agent_context import AgentContext
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.engine_run_state import EngineRunState
from engine.agents.openai_agent_runner import OpenAiAgentRunner
from engine.agents.openai_compactor import build_openai_compactor_factory
from engine.agents.prompt_templates import render_root_system_prompt
from engine.engine_config import EngineConfig
from engine.models.engine_output import AgentOutputItem, EngineStreamEvent
from engine.models.messages import AgentMessage
from engine.tools.subagent_tool_factory import build_root_sdk_agent
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore


async def stream_engine_async(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
) -> AsyncIterator[EngineStreamEvent]:
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=engine_config.trace_index,
    )
    trace_store = TraceStore.load(trace_path=trace_path, index_path=index_path)

    output_bus = EngineOutputBus()
    run_state = EngineRunState(
        trace_store=trace_store,
        output_bus=output_bus,
        config=engine_config,
    )

    root_execution = AgentExecution(
        agent_id=f"root-{uuid.uuid4().hex[:8]}",
        agent_name=engine_config.root_agent.name,
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )
    run_state.register(root_execution)

    root_context = AgentContext.from_input_messages(messages, engine_config)

    sdk_agent = build_root_sdk_agent(
        engine_config=engine_config,
        run_state=run_state,
        agent_execution=root_execution,
    )

    async def _run_streamed(*, agent, input, context):
        return Runner.run_streamed(starting_agent=agent, input=input, context=context)

    async def _drive() -> None:
        runner = OpenAiAgentRunner(
            run_streamed=_run_streamed,
            compactor_factory=build_openai_compactor_factory(engine_config),
        )
        await runner.run(
            sdk_agent=sdk_agent,
            agent_context=root_context,
            agent_execution=root_execution,
            output_bus=output_bus,
            is_root=True,
            run_context=run_state,
        )
        await output_bus.close()

    task = asyncio.create_task(_drive())

    try:
        async for event in output_bus.stream():
            yield event
        await task
    except BaseException:
        task.cancel()
        raise


async def run_engine_async(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
) -> list[AgentOutputItem]:
    out: list[AgentOutputItem] = []
    async for event in stream_engine_async(messages, engine_config, trace_path):
        if isinstance(event, AgentOutputItem):
            out.append(event)
    return out


def stream_engine(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
) -> list[EngineStreamEvent]:
    async def _collect() -> list[EngineStreamEvent]:
        out: list[EngineStreamEvent] = []
        async for ev in stream_engine_async(messages, engine_config, trace_path):
            out.append(ev)
        return out
    return asyncio.run(_collect())


def run_engine(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
) -> list[AgentOutputItem]:
    return asyncio.run(run_engine_async(messages, engine_config, trace_path))


def to_messages_array(
    input_messages: list[AgentMessage],
    results: list[AgentOutputItem],
    engine_config: EngineConfig,
) -> list[AgentMessage]:
    """Reconstruct a messages list suitable for a follow-up engine call.

    Returns the engine-rendered system message + the caller's non-system input
    messages + every depth=0 item the engine emitted, in order. Append a new
    user message and pass the result back to `run_engine_async` to continue
    the conversation.
    """
    expected_system = render_root_system_prompt(
        instructions=engine_config.root_agent.instructions,
        maximum_depth=engine_config.maximum_depth,
        maximum_parallel_subagents=engine_config.maximum_parallel_subagents,
    )
    body = (
        input_messages[1:]
        if input_messages and input_messages[0].role == "system"
        else input_messages
    )
    out: list[AgentMessage] = [AgentMessage(role="system", content=expected_system), *body]
    out.extend(r.item for r in results if r.depth == 0)
    return out
