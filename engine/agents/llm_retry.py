from __future__ import annotations

import asyncio
import logging
import random
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

# A 400 normally means "deterministically bad request". But HALO rebuilds every
# request from the local ``AgentContext`` history, so the common Responses-API
# culprits (stale ``rs_*`` reasoning-item ids, ``previous_response`` chains,
# broken item pairing) are fixed by a clean rerun (INF-3308). The usual trigger
# is the provider/proxy chain (eg LiteLLM → Azure Foundry) no longer recognizing
# server-side state minted on an earlier hop. So we treat 400s as retriable
# *except* the small, stable set below that a clean rerun cannot fix — keyed off
# the SDK's structured error ``code`` rather than matching error prose. A 400
# whose ``code`` the proxy dropped (``None``) is retried: bounded waste, capped
# by the runner circuit breaker, beats crashing a run on a recoverable error.
_TERMINAL_400_CODES = frozenset(
    {
        "context_length_exceeded",
        "content_filter",
        "string_above_max_length",
    }
)


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
      - 400s except the terminal codes in ``_TERMINAL_400_CODES`` — a clean
        rerun from local history fixes stale server-side state (INF-3308)
    """
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        if exc.status_code >= 500:
            return True
        if exc.status_code == 400:
            return exc.code not in _TERMINAL_400_CODES
        return False
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
