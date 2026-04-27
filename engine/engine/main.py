from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from engine.agents.agent_context import AgentContext
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.engine_run_state import EngineRunState
from engine.agents.openai_agent_runner import OpenAiAgentRunner
from engine.agents.openai_compactor import build_openai_compactor_factory
from engine.agents.runner_protocol import RunnerProtocol
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
    *,
    runner: RunnerProtocol | None = None,
) -> AsyncIterator[EngineStreamEvent]:
    """Run the HALO engine and stream events as they happen.

    Yields ``AgentOutputItem`` (assistant messages, tool calls, tool results)
    interleaved with ``AgentTextDelta`` (incremental token deltas). Items from
    subagents are interleaved with the root in monotonic ``sequence`` order.

    The ``runner`` keyword argument is a TEST SEAM: pass a custom
    ``RunnerProtocol`` (e.g. ``FakeRunner`` from the probes kit) to drive
    the engine with a scripted event stream instead of calling the OpenAI
    Agents SDK. Production callers leave it ``None`` to use ``agents.Runner``.
    """
    # TODO: ensure_index_exists could return the trace itself so we dont need to re load it from file in TraceStore.load
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=engine_config.trace_index,
    )
    trace_store = TraceStore.load(trace_path=trace_path, index_path=index_path)

    output_bus = EngineOutputBus()
    # TODO: dotn use untyped dict
    # TODO: Use proper mock (ie MagicMock) for test runner so we dont need to make it swappable in prod code. Get rid of runner completely from EngineRunState
    run_state_kwargs: dict = {
        "trace_store": trace_store,
        "output_bus": output_bus,
        "config": engine_config,
    }
    if runner is not None:
        run_state_kwargs["runner"] = runner
    run_state = EngineRunState(**run_state_kwargs)

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
        agent_context=root_context,
    )

    async def _run_streamed(*, agent, input, context):
        return run_state.runner.run_streamed(
            starting_agent=agent,
            input=input,
            context=context,
            max_turns=engine_config.root_agent.maximum_turns,
        )

    async def _drive() -> None:
        runner = OpenAiAgentRunner(
            run_streamed=_run_streamed,
            compactor_factory=build_openai_compactor_factory(engine_config),
        )
        try:
            await runner.run(
                sdk_agent=sdk_agent,
                agent_context=root_context,
                agent_execution=root_execution,
                output_bus=output_bus,
                is_root=True,
                run_context=run_state,
            )
            await output_bus.close()
        except BaseException as exc:
            await output_bus.fail(exc)

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
    *,
    runner: RunnerProtocol | None = None,
) -> list[AgentOutputItem]:
    """Run the engine to completion and return all ``AgentOutputItem``s.

    Streaming text deltas are filtered out; only durable items (assistant
    messages, tool calls, tool results) are returned. See
    ``stream_engine_async`` for the streaming variant and for the meaning of
    the ``runner`` test seam.
    """
    out: list[AgentOutputItem] = []
    async for event in stream_engine_async(messages, engine_config, trace_path, runner=runner):
        if isinstance(event, AgentOutputItem):
            out.append(event)
    return out


def stream_engine(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
    *,
    runner: RunnerProtocol | None = None,
) -> list[EngineStreamEvent]:
    """Synchronous wrapper around ``stream_engine_async``. Collects every
    streamed event into a list and returns it. Use the async variant if you
    want incremental results."""

    async def _collect() -> list[EngineStreamEvent]:
        out: list[EngineStreamEvent] = []
        async for ev in stream_engine_async(messages, engine_config, trace_path, runner=runner):
            out.append(ev)
        return out

    return asyncio.run(_collect())


def run_engine(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
    *,
    runner: RunnerProtocol | None = None,
) -> list[AgentOutputItem]:
    """Synchronous wrapper around ``run_engine_async``."""
    return asyncio.run(run_engine_async(messages, engine_config, trace_path, runner=runner))
