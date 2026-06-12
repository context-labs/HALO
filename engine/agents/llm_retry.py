from __future__ import annotations

import asyncio
import logging
import random
import re
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

DEFAULT_BACKOFF_BASE_SECONDS = 0.5
DEFAULT_BACKOFF_CAP_SECONDS = 30.0

# 400s that reference server-side response state are retriable for HALO
# (INF-3308): the engine rebuilds every request from the local
# ``AgentContext`` conversation history, so a rerun does not replay the
# payload that embedded the stale state. The usual culprit is the OpenAI
# Agents SDK re-sending Responses-API reasoning items (``rs_…`` ids)
# accumulated within one streamed run — when the provider/proxy chain
# (LiteLLM → Azure Foundry) no longer recognizes them, the next internal
# turn 400s. Plain 400s (bad fields, validation errors) stay
# non-retriable: an identical replay would deterministically fail again.
_STALE_RESPONSE_STATE_RE = re.compile(
    r"previous_response"
    r"|\brs_[A-Za-z0-9]"
    r"|required following item"
    r"|item with id",
    re.IGNORECASE,
)


def is_stale_response_state_error(exc: BaseException) -> bool:
    """True for 400s caused by stale server-side Responses state (``rs_*`` ids,
    ``previous_response`` chains, broken item pairing)."""
    if not isinstance(exc, APIStatusError) or exc.status_code != 400:
        return False
    return bool(_STALE_RESPONSE_STATE_RE.search(str(exc)))


def is_retriable_llm_error(exc: BaseException) -> bool:
    """Classify an exception as an LLM failure worth retrying.

    Retriable:
      - transport failures: connect errors, timeouts, dropped / incomplete
        streamed reads (including raw ``httpx`` errors that escape the SDK
        wrapper mid-stream, e.g. ``RemoteProtocolError: peer closed
        connection without sending complete message body``)
      - rate limits and provider 5xx
      - generic ``APIError`` (provider stream errors such as
        "The model produced invalid content")
      - 400s referencing stale server-side response state — see
        :func:`is_stale_response_state_error` (INF-3308)
    """
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500 or is_stale_response_state_error(exc)
    if isinstance(exc, APIError):
        return True
    if isinstance(exc, (httpx.HTTPError, TimeoutError)):
        return True
    return False


def backoff_delay(
    failure_count: int,
    *,
    base: float = DEFAULT_BACKOFF_BASE_SECONDS,
    cap: float = DEFAULT_BACKOFF_CAP_SECONDS,
) -> float:
    """Full-jitter exponential backoff: ``uniform(0, min(cap, base * 2**(n-1)))``.

    ``failure_count`` is 1-based (the first failure sleeps up to ``base``).
    A non-positive ``base`` disables sleeping entirely (used by tests).
    """
    if base <= 0:
        return 0.0
    ceiling = min(cap, base * (2 ** max(0, failure_count - 1)))
    return random.uniform(0, ceiling)


async def call_with_retries(
    fn: Callable[[], Awaitable[_T]],
    *,
    description: str,
    max_attempts: int = 4,
    backoff_base: float = DEFAULT_BACKOFF_BASE_SECONDS,
    backoff_cap: float = DEFAULT_BACKOFF_CAP_SECONDS,
) -> _T:
    """Await ``fn()`` with retries on transient LLM errors.

    Intended for HALO's non-streaming summarization calls (compaction,
    synthesis), which previously had no retry at all. Non-retriable errors
    and the final retriable failure propagate unchanged.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except Exception as exc:
            if attempt >= max_attempts or not is_retriable_llm_error(exc):
                raise
            delay = backoff_delay(attempt, base=backoff_base, cap=backoff_cap)
            logger.warning(
                "%s failed with %s (attempt %s of %s); retrying in %.2fs",
                description,
                type(exc).__name__,
                attempt,
                max_attempts,
                delay,
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
