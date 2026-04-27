from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from engine.agents.agent_context import AgentContext, Compactor
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.openai_event_mapper import OpenAiEventMapper
from engine.errors import EngineAgentExhaustedError


def _is_retriable_llm_error(exc: BaseException) -> bool:
    """Classify an exception as a transient LLM failure worth retrying."""
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    return False


MAX_CONSECUTIVE_LLM_FAILURES = 10

RunStreamedCallable = Callable[..., Awaitable[Any]]
CompactorFactory = Callable[[AgentExecution], Compactor]


class OpenAiAgentRunner:
    """Drives one OpenAI Agents SDK ``Agent`` and bridges its event stream into Engine state.

    Per call to ``run``, retries the SDK call on transient LLM errors up to a circuit
    breaker, normalizes streamed SDK events through ``OpenAiEventMapper``, appends
    context items, emits output items to the bus, and runs compaction when the turn
    completes. Used by both the root agent (in ``main.py``) and subagents (in
    ``subagent_tool_factory``).
    """

    def __init__(
        self,
        run_streamed: RunStreamedCallable,
        compactor_factory: CompactorFactory,
        event_mapper: OpenAiEventMapper | None = None,
    ) -> None:
        """``run_streamed`` is injected so root and subagent paths can supply their own
        max_turns and starting agent. ``compactor_factory`` produces a per-execution
        compactor bound to whatever model EngineConfig pins for compaction."""
        self._run_streamed = run_streamed
        self._compactor_factory = compactor_factory
        self._mapper = event_mapper or OpenAiEventMapper()

    async def run(
        self,
        *,
        sdk_agent: Any,
        agent_context: AgentContext,
        agent_execution: AgentExecution,
        output_bus: EngineOutputBus,
        is_root: bool,
        run_context: Any | None = None,
    ) -> None:
        """Execute one agent end-to-end: SDK stream → context items → bus → compaction.

        Raises ``EngineAgentExhaustedError`` when consecutive transient LLM failures
        exceed ``MAX_CONSECUTIVE_LLM_FAILURES``. Non-retriable exceptions propagate.
        """
        messages = [m.model_dump(exclude_none=True) for m in agent_context.to_messages_array()]
        last_exc: BaseException | None = None

        while agent_execution.consecutive_llm_failures < MAX_CONSECUTIVE_LLM_FAILURES:
            try:
                # TODO: Pretty sure an iterator wont throw here
                stream = await self._run_streamed(agent=sdk_agent, input=messages, context=run_context)
            except Exception as exc:
                if not _is_retriable_llm_error(exc):
                    raise
                last_exc = exc
                agent_execution.record_llm_failure()
                logger.warning(
                    "llm call failed for agent_id={} (failure {} of {})",
                    agent_execution.agent_id,
                    agent_execution.consecutive_llm_failures,
                    MAX_CONSECUTIVE_LLM_FAILURES,
                )
                continue

            agent_execution.record_llm_success()

            async for raw_event in stream.stream_events():
                mapped = self._mapper.to_mapped_event(raw_event, execution=agent_execution, is_root=is_root)
                if mapped.context_item is not None:
                    agent_context.append(mapped.context_item)
                if mapped.output_item is not None:
                    emitted = await output_bus.emit(mapped.output_item)
                    if agent_execution.output_start_sequence is None:
                        agent_execution.output_start_sequence = emitted.sequence
                    agent_execution.output_end_sequence = emitted.sequence
                    item = mapped.output_item.item
                    if item.role == "assistant":
                        if item.tool_calls:
                            agent_execution.tool_calls_made += len(item.tool_calls)
                        else:
                            agent_execution.turns_used += 1
                if mapped.delta is not None:
                    await output_bus.emit(mapped.delta)

            # TODO: Remove passing in compactor_factory, the compactor agent/llm config should be instantiated once and re used
            await agent_context.compact_old_items(self._compactor_factory(agent_execution))
            return

        raise EngineAgentExhaustedError(
            f"agent {agent_execution.agent_id} exhausted after {MAX_CONSECUTIVE_LLM_FAILURES} consecutive failures"
        ) from last_exc
