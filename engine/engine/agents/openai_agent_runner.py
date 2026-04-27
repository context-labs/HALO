from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agents import set_default_openai_client
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from engine.agents.agent_context import AgentContext, Compactor
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.openai_event_mapper import OpenAiEventMapper
from engine.errors import EngineAgentExhaustedError
from engine.model_provider_config import ModelProviderConfig


def configure_default_sdk_client(provider: ModelProviderConfig) -> None:
    """Bind the OpenAI Agents SDK's default client to the configured endpoint.

    The SDK uses a process-global client, so this is best-effort for callers
    running multiple engines in one process. We only override when at least
    one of ``base_url`` / ``api_key`` is set; otherwise the SDK keeps using
    its env-driven default.
    """
    if provider.base_url is None and provider.api_key is None:
        return
    set_default_openai_client(AsyncOpenAI(base_url=provider.base_url, api_key=provider.api_key))


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
logger = logging.getLogger(__name__)


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

        ``Runner.run_streamed`` returns immediately and the actual LLM request happens
        lazily inside ``stream.stream_events()``, so the retry try/except wraps the
        full iteration — connection errors, timeouts, rate limits, and 5xx surface
        there, not on the ``run_streamed`` call.

        Retry only fires when zero events were processed in the failed attempt.
        Once any event has been applied to ``agent_context``/``agent_execution``/
        ``output_bus``, a replay would corrupt state (duplicate context items,
        double-counted turns, duplicate bus emissions). Mid-stream failures are
        surfaced to the caller; turn-boundary recovery happens one layer up
        (parent agents see a ``SubagentToolResult`` failure; root agents see
        ``output_bus.fail()``).
        """
        last_exc: BaseException | None = None
        # Built once: retry only fires when events_seen == 0, so the context can't
        # have been mutated between attempts.
        messages = [m.model_dump(exclude_none=True) for m in agent_context.to_messages_array()]

        while agent_execution.consecutive_llm_failures < MAX_CONSECUTIVE_LLM_FAILURES:
            events_seen = 0
            try:
                stream = await self._run_streamed(
                    agent=sdk_agent, input=messages, context=run_context
                )
                async for raw_event in stream.stream_events():
                    events_seen += 1
                    mapped = self._mapper.to_mapped_event(
                        raw_event, execution=agent_execution, is_root=is_root
                    )
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
            except Exception as exc:
                if events_seen > 0 or not _is_retriable_llm_error(exc):
                    raise
                last_exc = exc
                agent_execution.record_llm_failure()
                logger.warning(
                    "llm call failed for agent_id=%s (failure %s of %s)",
                    agent_execution.agent_id,
                    agent_execution.consecutive_llm_failures,
                    MAX_CONSECUTIVE_LLM_FAILURES,
                )
                continue

            agent_execution.record_llm_success()
            # TODO: Remove passing in compactor_factory, the compactor agent/llm config should be instantiated once and re used
            await agent_context.compact_old_items(self._compactor_factory(agent_execution))
            return

        raise EngineAgentExhaustedError(
            f"agent {agent_execution.agent_id} exhausted after {MAX_CONSECUTIVE_LLM_FAILURES} consecutive failures"
        ) from last_exc
