"""Worked example: circuit breaker / retry classification.

Run with::

    cd engine && uv run python -m tests.probes.example_circuit_breaker

Prints one ``PASS:`` or ``FAIL:`` line per check and exits nonzero if any
check failed.

This file demonstrates the conventions every probe script should follow:

  1. Compose ``FakeRunner`` programs for each scenario you want to probe.
  2. Call ``run_with_fake`` (always returns; never raises — failures land in
     ``RunResult.error``).
  3. Make focused assertions about ``output_items`` / ``error`` / runner.calls.
  4. Print ``PASS: <one-line description>`` or ``FAIL: <one-line description
     — observed: ...>``.
  5. Track failures and ``sys.exit(1)`` at the bottom if any.

The pathways probed here are (1) retriable error → retry → success, (2) ten
consecutive retriable errors → ``EngineAgentExhaustedError``, and (3)
non-retriable error → propagates immediately. They map directly to
``OpenAiAgentRunner`` lines 16-21, 54, and 90-93.
"""

from __future__ import annotations

import asyncio
import sys

import httpx
from openai import APIConnectionError, BadRequestError

from engine.errors import EngineAgentExhaustedError
from tests.probes.probe_kit import (
    FakeRunner,
    make_assistant_text,
    run_with_fake,
)

_FAILURES: list[str] = []


def _check(condition: bool, description: str, observed: str = "") -> None:
    if condition:
        print(f"PASS: {description}")
    else:
        suffix = f" — observed: {observed}" if observed else ""
        print(f"FAIL: {description}{suffix}")
        _FAILURES.append(description)


# ---------------------------------------------------------------------------


async def probe_baseline_success() -> None:
    """Sanity check: a single assistant message ending in <final/> produces
    one output item with final=True. Confirms the kit wires into the engine."""
    runner = FakeRunner(
        [make_assistant_text("hello\n<final/>", item_id="m1")],
    )
    result = await run_with_fake(runner)

    _check(result.error is None, "baseline: run completes without error",
           observed=f"error={type(result.error).__name__ if result.error else None}")
    _check(any(item.final for item in result.output_items),
           "baseline: at least one item has final=True")
    _check(len(runner.calls) == 1,
           "baseline: FakeRunner.run_streamed called exactly once",
           observed=f"calls={len(runner.calls)}")


async def probe_retry_then_success() -> None:
    """Retriable APIConnectionError on the first call, then a success on the
    second. The runner should retry, succeed, and emit a final item."""
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    runner = FakeRunner(
        APIConnectionError(request=fake_request),
        [make_assistant_text("recovered\n<final/>", item_id="m2")],
    )
    result = await run_with_fake(runner)

    _check(result.error is None,
           "retry-then-success: run completes without error",
           observed=f"error={type(result.error).__name__ if result.error else None}")
    _check(len(runner.calls) == 2,
           "retry-then-success: FakeRunner called twice (1 retry + 1 success)",
           observed=f"calls={len(runner.calls)}")
    _check(any(item.final for item in result.output_items),
           "retry-then-success: at least one item has final=True")


async def probe_circuit_breaker_exhaustion() -> None:
    """Ten consecutive retriable errors trip the breaker and raise
    ``EngineAgentExhaustedError`` through the stream."""
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    runner = FakeRunner(*[APIConnectionError(request=fake_request) for _ in range(10)])
    result = await run_with_fake(runner)

    _check(isinstance(result.error, EngineAgentExhaustedError),
           "exhaustion: error type is EngineAgentExhaustedError",
           observed=f"error={type(result.error).__name__ if result.error else None}")
    _check(len(runner.calls) == 10,
           "exhaustion: FakeRunner called exactly MAX_CONSECUTIVE_LLM_FAILURES times",
           observed=f"calls={len(runner.calls)}")


async def probe_non_retriable_propagates() -> None:
    """A non-retriable error (BadRequestError → 400) should propagate
    immediately without retry."""
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    fake_response = httpx.Response(400, request=fake_request)
    runner = FakeRunner(
        BadRequestError(message="bad", response=fake_response, body={"error": {"message": "bad"}}),
    )
    result = await run_with_fake(runner)

    _check(isinstance(result.error, BadRequestError),
           "non-retriable: BadRequestError propagates",
           observed=f"error={type(result.error).__name__ if result.error else None}")
    _check(len(runner.calls) == 1,
           "non-retriable: FakeRunner called once (no retry)",
           observed=f"calls={len(runner.calls)}")


# ---------------------------------------------------------------------------


async def main() -> int:
    await probe_baseline_success()
    await probe_retry_then_success()
    await probe_circuit_breaker_exhaustion()
    await probe_non_retriable_propagates()

    if _FAILURES:
        print(f"\n{len(_FAILURES)} check(s) failed:")
        for desc in _FAILURES:
            print(f"  - {desc}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
