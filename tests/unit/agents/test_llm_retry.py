from __future__ import annotations

import httpx
import pytest
from openai import APIConnectionError, APIError, BadRequestError, InternalServerError

from engine.agents.llm_retry import (
    backoff_delay,
    call_with_retries,
    is_retriable_llm_error,
    is_stale_response_state_error,
)

_REQ = httpx.Request("POST", "https://api.openai.com/v1/responses")


def _status_error(status: int, message: str):
    response = httpx.Response(status, request=_REQ)
    if status == 400:
        return BadRequestError(
            message=message, response=response, body={"error": {"message": message}}
        )
    return InternalServerError(
        message=message, response=response, body={"error": {"message": message}}
    )


def test_transport_and_5xx_are_retriable() -> None:
    assert is_retriable_llm_error(APIConnectionError(request=_REQ))
    assert is_retriable_llm_error(_status_error(500, "upstream exploded"))
    assert is_retriable_llm_error(
        APIError(message="incomplete chunked read", request=_REQ, body=None)
    )
    # Raw httpx errors can escape the SDK wrapper mid-stream.
    assert is_retriable_llm_error(
        httpx.RemoteProtocolError("peer closed connection without sending complete message body")
    )
    assert is_retriable_llm_error(TimeoutError())


def test_plain_400_is_not_retriable() -> None:
    exc = _status_error(400, "bad field")
    assert not is_stale_response_state_error(exc)
    assert not is_retriable_llm_error(exc)


@pytest.mark.parametrize(
    "message",
    [
        "Item with id 'rs_0123abcdef' not found.",
        "Item 'rs_99' of type 'reasoning' was provided without its required following item.",
        "previous_response_id 'resp_123' not found",
    ],
)
def test_stale_response_state_400_is_retriable(message: str) -> None:
    exc = _status_error(400, message)
    assert is_stale_response_state_error(exc)
    assert is_retriable_llm_error(exc)


def test_unrelated_exception_is_not_retriable() -> None:
    assert not is_retriable_llm_error(ValueError("nope"))


def test_backoff_delay_is_bounded_and_disableable() -> None:
    assert backoff_delay(1, base=0.0) == 0.0
    for failure_count in range(1, 12):
        delay = backoff_delay(failure_count, base=0.5, cap=3.0)
        assert 0.0 <= delay <= 3.0


@pytest.mark.asyncio
async def test_call_with_retries_recovers_from_transient_failure() -> None:
    attempts = 0

    async def flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise APIConnectionError(request=_REQ)
        return "ok"

    result = await call_with_retries(
        flaky, description="test call", max_attempts=4, backoff_base=0.0
    )
    assert result == "ok"
    assert attempts == 3


@pytest.mark.asyncio
async def test_call_with_retries_gives_up_after_max_attempts() -> None:
    attempts = 0

    async def always_fails():
        nonlocal attempts
        attempts += 1
        raise APIConnectionError(request=_REQ)

    with pytest.raises(APIConnectionError):
        await call_with_retries(
            always_fails, description="test call", max_attempts=3, backoff_base=0.0
        )
    assert attempts == 3


@pytest.mark.asyncio
async def test_call_with_retries_propagates_non_retriable_immediately() -> None:
    attempts = 0

    async def bad_request():
        nonlocal attempts
        attempts += 1
        raise _status_error(400, "bad field")

    with pytest.raises(BadRequestError):
        await call_with_retries(
            bad_request, description="test call", max_attempts=4, backoff_base=0.0
        )
    assert attempts == 1
