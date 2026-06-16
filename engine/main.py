from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import TypeVar

from agents import RunConfig, Runner
from agents.models.openai_provider import OpenAIProvider
from openai import AsyncOpenAI

from engine.agents.agent_context import AgentContext
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.engine_run_state import EngineRunState
from engine.agents.openai_agent_runner import OpenAiAgentRunner
from engine.agents.turn_counter import TurnCounterInputFilter
from engine.code.code_repo import CodeRepo
from engine.engine_config import EngineConfig
from engine.git.git_repo import GitRepo
from engine.models.engine_output import AgentOutputItem, EngineStreamEvent
from engine.models.messages import AgentMessage
from engine.sandbox.sandbox import Sandbox
from engine.telemetry import resolve_run_id, setup_telemetry
from engine.telemetry.tracing import halo_agent_span
from engine.tools.subagent_tool_factory import build_root_sdk_agent
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore

_T = TypeVar("_T")
logger = logging.getLogger(__name__)


async def stream_engine_async(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
    *,
    telemetry: bool = False,
) -> AsyncGenerator[EngineStreamEvent, None]:
    """Run the HALO engine and stream events as they happen.

    Yields ``AgentOutputItem`` (assistant messages, tool calls, tool results)
    interleaved with ``AgentTextDelta`` (incremental token deltas). Items from
    subagents are interleaved with the root in monotonic ``sequence`` order.

    Set ``telemetry=True`` to emit OpenInference traces of HALO's own LLM /
    tool / agent activity. Routing: if ``CATALYST_OTLP_TOKEN`` is set, spans
    are uploaded to inference.net Catalyst over OTLP. Otherwise spans are
    written to the local JSONL file at ``$HALO_TELEMETRY_PATH`` (default:
    ``./halo-telemetry-{run_id}.jsonl``). Off by default — no overhead, no
    file writes, no env var reads when ``telemetry=False``.
    """
    run_id = resolve_run_id()
    telemetry_handle = setup_telemetry(enable=telemetry, run_id=run_id)
    try:
        client = AsyncOpenAI(
            base_url=engine_config.model_provider.base_url,
            api_key=engine_config.model_provider.api_key,
            default_headers=engine_config.model_provider.default_headers,
        )
        try:
            # Root + subagent share ``agent_id="halo"`` so Catalyst's Agents
            # view collapses every HALO run into one identity row; the span
            # name still distinguishes root vs subagent work in the trace tree.
            with halo_agent_span(span_name="halo-root.run", agent_id="halo", system="openai"):
                sandbox = Sandbox.get()

                index_path = await TraceIndexBuilder.ensure_index_exists(
                    trace_path=trace_path,
                    config=engine_config.trace_index,
                )
                trace_store = TraceStore.load(trace_path=trace_path, index_path=index_path)

                # Resolve the code repo (or None) before any LLM call so a bad
                # ``repo_path`` fails fast. None leaves the code tools unregistered.
                code_repo = (
                    CodeRepo.open(engine_config.repo_path)
                    if engine_config.repo_path is not None
                    else None
                )
                # Git tools are additive: the same repo_path, enabled only when it's
                # a git work tree and git is on PATH. GitRepo.open never raises —
                # None just leaves the git tools unregistered.
                git_repo = (
                    GitRepo.open(engine_config.repo_path)
                    if engine_config.repo_path is not None
                    else None
                )

                output_bus = EngineOutputBus()
                run_state = EngineRunState(
                    trace_store=trace_store,
                    output_bus=output_bus,
                    config=engine_config,
                    sandbox=sandbox,
                    code_repo=code_repo,
                    git_repo=git_repo,
                    openai_client=client,
                )

                root_execution = AgentExecution(
                    agent_id=f"root-{uuid.uuid4().hex[:8]}",
                    agent_name=engine_config.root_agent.name,
                    depth=0,
                    parent_agent_id=None,
                    parent_tool_call_id=None,
                )
                run_state.register(root_execution)

                root_context = AgentContext.from_input_messages(
                    messages=messages,
                    engine_config=engine_config,
                    code_repo=code_repo,
                    git_repo=git_repo,
                )

                sdk_agent = build_root_sdk_agent(
                    engine_config=engine_config,
                    run_state=run_state,
                    agent_execution=root_execution,
                    agent_context=root_context,
                )

                async def _run_streamed(*, agent, input, context):
                    # Fresh filter per SDK Runner.run_streamed invocation so
                    # OpenAiAgentRunner retries reset the counter alongside the
                    # SDK's own max_turns counter. Without this, a transient LLM
                    # failure on the first turn would leave _current=1 and the
                    # next attempt would render "[HALO: turn 2 of M]" while the
                    # SDK is internally on turn 1.
                    #
                    # ``model_provider`` pins the SDK to HALO's configured
                    # ``AsyncOpenAI`` for this run instead of having
                    # ``OpenAIProvider`` lazy-construct its own from env vars
                    # (which loses ``default_headers`` and the deterministic
                    # close in ``finally`` below). Per-call wiring also keeps
                    # SDK state out of process globals — subagents inherit the
                    # same client via the matching pass in
                    # ``subagent_tool_factory``, so test paths that invoke
                    # ``call_subagent.on_invoke_tool`` directly (bypassing
                    # ``stream_engine_async``) stay symmetric with production.
                    run_config = RunConfig(
                        model_provider=OpenAIProvider(openai_client=client),
                        call_model_input_filter=TurnCounterInputFilter(
                            max_turns=engine_config.root_agent.maximum_turns,
                            is_root=True,
                        ),
                    )
                    return Runner.run_streamed(
                        starting_agent=agent,
                        input=input,
                        context=context,
                        max_turns=engine_config.root_agent.maximum_turns,
                        run_config=run_config,
                    )

                async def _drive() -> None:
                    agent_runner = OpenAiAgentRunner(
                        run_streamed=_run_streamed,
                        client=run_state.openai_client,
                        refusal_retries=engine_config.root_agent.refusal_retries,
                    )
                    try:
                        await agent_runner.run(
                            sdk_agent=sdk_agent,
                            agent_context=root_context,
                            agent_execution=root_execution,
                            output_bus=output_bus,
                            is_root=True,
                            run_context=run_state,
                        )
                        await output_bus.close()
                    except Exception as exc:
                        await output_bus.fail(exc)
                    except BaseException as exc:
                        # CancelledError / KeyboardInterrupt / SystemExit: drain the bus
                        # so the consumer doesn't hang on _queue.get(), then re-raise so
                        # the task transitions to the proper cancelled/failed state.
                        await output_bus.fail(exc)
                        raise

                task = asyncio.create_task(_drive())

                try:
                    async for event in output_bus.stream():
                        yield event
                    await task
                except BaseException:
                    task.cancel()
                    try:
                        await task
                    except BaseException:
                        pass
                    raise
        finally:
            try:
                await client.close()
            except Exception:
                logger.exception("failed to close AsyncOpenAI client")
    finally:
        if telemetry_handle is not None:
            telemetry_handle.shutdown()


async def stream_engine_output_async(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
    *,
    telemetry: bool = False,
) -> AsyncGenerator[AgentOutputItem, None]:
    """Stream durable ``AgentOutputItem``s only — no text deltas.

    Same lifecycle and arguments as ``stream_engine_async``, but
    pre-filters incremental ``AgentTextDelta`` events. Use this when
    you want to log or persist each completed step (assistant
    message, tool call, tool result) as it lands without dealing
    with the streaming-token noise.
    """
    async for event in stream_engine_async(
        messages, engine_config, trace_path, telemetry=telemetry
    ):
        if isinstance(event, AgentOutputItem):
            yield event


def _drive_sync(agen: AsyncGenerator[_T, None]) -> Iterator[_T]:
    """Drive an async iterator on a private event loop, yielding each item.

    Always calls ``agen.aclose()`` before closing the loop so that the
    underlying async generator's ``finally`` blocks run — even when the
    caller breaks out of the sync generator early (which raises
    ``GeneratorExit`` into this function's ``yield``). Without this,
    background tasks and telemetry handles started inside
    ``stream_engine_async`` would leak.
    """
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                yield loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                return
    finally:
        try:
            loop.run_until_complete(agen.aclose())
        finally:
            loop.close()


def stream_engine_output(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
    *,
    telemetry: bool = False,
) -> Iterator[AgentOutputItem]:
    """Synchronous generator around ``stream_engine_output_async``.

    Yields each ``AgentOutputItem`` as it lands so sync callers can
    log / persist per-step without writing async code. Drives the
    underlying async iterator on a private event loop one step at a
    time. If you want everything in a list, call ``run_engine`` (sync,
    collects to list) instead.
    """
    yield from _drive_sync(
        stream_engine_output_async(messages, engine_config, trace_path, telemetry=telemetry)
    )


async def run_engine_async(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
    *,
    telemetry: bool = False,
) -> list[AgentOutputItem]:
    """Run the engine to completion and return all ``AgentOutputItem``s.

    Streaming text deltas are filtered out; only durable items (assistant
    messages, tool calls, tool results) are returned. See
    ``stream_engine_async`` for the streaming variant.
    """
    return [
        item
        async for item in stream_engine_output_async(
            messages, engine_config, trace_path, telemetry=telemetry
        )
    ]


def stream_engine(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
    *,
    telemetry: bool = False,
) -> Iterator[EngineStreamEvent]:
    """Synchronous generator around ``stream_engine_async``.

    Yields each ``EngineStreamEvent`` (durable items + text deltas)
    as it lands. Drives the async iterator on a private event loop
    one step at a time. If you want everything in a list, call
    ``run_engine`` (sync, collects to list) instead.
    """
    yield from _drive_sync(
        stream_engine_async(messages, engine_config, trace_path, telemetry=telemetry)
    )


def run_engine(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
    *,
    telemetry: bool = False,
) -> list[AgentOutputItem]:
    """Synchronous wrapper around ``run_engine_async``."""
    return asyncio.run(run_engine_async(messages, engine_config, trace_path, telemetry=telemetry))
