from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from openai import APIStatusError, AsyncOpenAI

from engine.agents.agent_context import AgentContext
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.llm_retry import (
    DEFAULT_BACKOFF_BASE_SECONDS,
    DEFAULT_BACKOFF_CAP_SECONDS,
    backoff_delay,
    is_retriable_llm_error,
)
from engine.agents.openai_event_mapper import OpenAiEventMapper
from engine.errors import EngineAgentExhaustedError, EngineAgentRefusedError

MAX_CONSECUTIVE_LLM_FAILURES = 10

# Transient user nudge appended when a root run ends without finalizing
# (no ``final_answer`` call, no legacy ``<final/>`` sentinel). Mirrors the
# refusal-retry "Continue." pattern: appended to the resent input only,
# never stored in ``AgentContext``.
FINAL_ANSWER_REPROMPT_MESSAGE = (
    "[HALO: your run ended without calling the `final_answer` tool. "
    "Call `final_answer` now with your complete final answer.]"
)

RunStreamedCallable = Callable[..., Awaitable[Any]]
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
        client: AsyncOpenAI,
        event_mapper: OpenAiEventMapper | None = None,
        refusal_retries: int = 0,
        final_reprompts: int = 0,
        retry_backoff_base: float = DEFAULT_BACKOFF_BASE_SECONDS,
        retry_backoff_cap: float = DEFAULT_BACKOFF_CAP_SECONDS,
    ) -> None:
        """``run_streamed`` is injected so root and subagent paths can supply their own
        max_turns and starting agent. ``client`` is the per-run AsyncOpenAI used for
        compaction calls. ``final_reprompts`` is how many times a run that ends
        without a final-flagged item is re-nudged to call ``final_answer`` —
        only meaningful with ``is_root=True`` (subagent paths leave it 0).
        ``retry_backoff_base``/``retry_backoff_cap`` shape the
        full-jitter exponential backoff between LLM retries (base <= 0 disables
        sleeping; used by tests)."""
        self._run_streamed = run_streamed
        self._client = client
        self._mapper = event_mapper or OpenAiEventMapper()
        self._refusal_retries = refusal_retries
        self._final_reprompts = final_reprompts
        self._retry_backoff_base = retry_backoff_base
        self._retry_backoff_cap = retry_backoff_cap

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

        Mid-stream failures (events already processed when the stream dies —
        dropped connections, incomplete chunked reads, stale ``rs_*`` 400s from
        the SDK re-sending Responses-API reasoning items, streams that end
        without a terminal event and surface as the SDK's
        ``ModelBehaviorError: Model did not produce a final response!``) are
        recovered by
        rerunning from the LOCAL conversation history (INF-3504 / INF-3308):
        completed turns stay in ``agent_context``; only a trailing incomplete
        tool turn is trimmed (see ``AgentContext.trim_incomplete_tool_turn``)
        so the rendered message array stays valid, then the next attempt
        resumes from the last consistent point instead of losing the run.
        Output items already emitted to the bus cannot be retracted — consumers
        may see a trimmed turn followed by its regenerated equivalent, which is
        the accepted tradeoff versus failing a long run outright. Each retry
        sleeps with full-jitter exponential backoff.
        """
        last_exc: BaseException | None = None
        refusal_attempts = 0
        pending_refusal_retry = False
        last_refusal_text: str | None = None
        final_seen = False
        final_reprompt_attempts = 0
        pending_final_reprompt = False

        while agent_execution.consecutive_llm_failures < MAX_CONSECUTIVE_LLM_FAILURES:
            events_seen = 0
            items_before_attempt = len(agent_context.items)
            attempt_refusal_text: str | None = None
            messages = [m.model_dump(exclude_none=True) for m in agent_context.to_messages_array()]
            if pending_refusal_retry:
                # Sometimes gpt 5.5 randomly refuses requests. We simply need to reprompt it to continue.
                messages.append({"role": "user", "content": "Continue."})
            if pending_final_reprompt:
                messages.append({"role": "user", "content": FINAL_ANSWER_REPROMPT_MESSAGE})
            try:
                stream = await self._run_streamed(
                    agent=sdk_agent, input=messages, context=run_context
                )
                # Like the refusal retry, the pending nudge stays set until the
                # stream starts successfully so a transient failure on the
                # nudged attempt re-sends it instead of silently dropping it.
                pending_refusal_retry = False
                pending_final_reprompt = False
                async for raw_event in stream.stream_events():
                    events_seen += 1
                    mapped = self._mapper.to_mapped_event(
                        raw_event, execution=agent_execution, is_root=is_root
                    )
                    if mapped.refusal_text is not None:
                        attempt_refusal_text = mapped.refusal_text
                        last_refusal_text = mapped.refusal_text
                        continue
                    if mapped.context_item is not None:
                        attempt_refusal_text = None
                        agent_context.append(mapped.context_item)
                    if mapped.output_item is not None:
                        emitted = await output_bus.emit(mapped.output_item)
                        if mapped.output_item.final:
                            final_seen = True
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
                if not is_retriable_llm_error(exc):
                    raise
                if events_seen > 0:
                    # Mid-stream failure: trim a trailing incomplete tool turn
                    # so the next attempt reruns from the last consistent point
                    # of the local conversation history. Completed turns from
                    # this attempt are preserved.
                    removed = agent_context.trim_incomplete_tool_turn(
                        min_items=items_before_attempt
                    )
                    for item in removed:
                        if item.role == "assistant" and item.tool_calls:
                            agent_execution.tool_calls_made -= len(item.tool_calls)
                        elif item.role == "assistant":
                            agent_execution.turns_used -= 1
                    logger.warning(
                        "mid-stream llm failure for agent_id=%s after %s events "
                        "(%s incomplete tail item(s) trimmed); rerunning from local history",
                        agent_execution.agent_id,
                        events_seen,
                        len(removed),
                    )
                last_exc = exc
                agent_execution.record_llm_failure()
                status_error = exc if isinstance(exc, APIStatusError) else None
                logger.warning(
                    "llm call failed for agent_id=%s with %s (status=%s code=%s) "
                    "(failure %s of %s)",
                    agent_execution.agent_id,
                    type(exc).__name__,
                    status_error.status_code if status_error else None,
                    status_error.code if status_error else None,
                    agent_execution.consecutive_llm_failures,
                    MAX_CONSECUTIVE_LLM_FAILURES,
                )
                if status_error is not None:
                    logger.debug(
                        "llm status-error body for agent_id=%s: %s",
                        agent_execution.agent_id,
                        status_error.body,
                    )
                await asyncio.sleep(
                    backoff_delay(
                        agent_execution.consecutive_llm_failures,
                        base=self._retry_backoff_base,
                        cap=self._retry_backoff_cap,
                    )
                )
                continue

            if attempt_refusal_text is not None:
                if refusal_attempts < self._refusal_retries:
                    refusal_attempts += 1
                    pending_refusal_retry = True
                    logger.warning(
                        "model refusal for agent_id=%s; retrying refusal %s of %s",
                        agent_execution.agent_id,
                        refusal_attempts,
                        self._refusal_retries,
                    )
                    continue
                raise EngineAgentRefusedError(
                    f"agent {agent_execution.agent_id} exhausted after "
                    f"{self._refusal_retries} model-refusal retries: {last_refusal_text}"
                )

            if is_root and not final_seen and final_reprompt_attempts < self._final_reprompts:
                # The SDK run ended (a plain no-tool-call assistant turn, or a
                # malformed ``final_answer`` call) without a final-flagged item.
                # Nudge once and rerun from local history — the transient user
                # message is only on the resent input, like the refusal retry.
                final_reprompt_attempts += 1
                pending_final_reprompt = True
                agent_execution.record_llm_success()
                logger.warning(
                    "root run ended without finalizing for agent_id=%s; "
                    "reprompting for final_answer (%s of %s)",
                    agent_execution.agent_id,
                    final_reprompt_attempts,
                    self._final_reprompts,
                )
                continue

            agent_execution.record_llm_success()
            await agent_context.compact_old_items(self._client)
            return

        raise EngineAgentExhaustedError(
            f"agent {agent_execution.agent_id} exhausted after {MAX_CONSECUTIVE_LLM_FAILURES} consecutive failures"
        ) from last_exc
