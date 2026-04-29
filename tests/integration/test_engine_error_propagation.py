"""Engine error propagation: SDK failures must surface through the bus, not deadlock.

Two pathways the runner-level unit tests can't see:

1. Non-retriable SDK error → ``stream_engine_async``'s consumer must observe the
   real exception, not block forever waiting for events on a never-closed bus.
2. Retriable-error exhaustion → ``EngineAgentExhaustedError`` propagates
   through the engine boundary, not just through ``OpenAiAgentRunner.run``.

A ``TimeoutError`` from ``run_with_fake`` is the deadlock signal — the bus
was never closed on the error path.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from openai import APIConnectionError, BadRequestError

from engine.errors import EngineAgentExhaustedError
from tests.probes.probe_kit import FakeRunner, run_with_fake


@pytest.mark.asyncio
async def test_non_retriable_error_propagates_without_deadlock() -> None:
    """A 400-class SDK error raised inside the driver must surface as itself
    to the stream consumer. If the engine forgets to close the output bus on
    the failure path, ``run_with_fake`` times out instead — that's the bug
    this test guards against."""
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(400, request=request)
    runner = FakeRunner(
        BadRequestError(message="bad", response=response, body={"error": {"message": "bad"}}),
    )

    result = await run_with_fake(runner, timeout_seconds=3.0)

    assert not isinstance(result.error, asyncio.TimeoutError), (
        "engine deadlocked on non-retriable error — output bus was not closed"
    )
    assert isinstance(result.error, BadRequestError)
    assert len(runner.calls) == 1


@pytest.mark.asyncio
async def test_circuit_breaker_exhaustion_propagates_through_engine() -> None:
    """Ten consecutive retriable errors trip the breaker. The runner-level
    unit tests confirm ``EngineAgentExhaustedError`` is raised inside
    ``OpenAiAgentRunner.run``; this test confirms it propagates all the way
    out through ``stream_engine_async`` rather than getting swallowed at the
    engine boundary."""
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    runner = FakeRunner(*[APIConnectionError(request=request) for _ in range(10)])

    result = await run_with_fake(runner)

    assert isinstance(result.error, EngineAgentExhaustedError)
    assert len(runner.calls) == 10
