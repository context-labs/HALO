from __future__ import annotations

import inspect

import engine.main as main
from engine.main import _drive_sync


def test_public_entrypoints_exist_and_are_async() -> None:
    assert inspect.isasyncgenfunction(main.stream_engine_async)
    assert inspect.iscoroutinefunction(main.run_engine_async)
    assert callable(main.stream_engine)
    assert callable(main.run_engine)


def test_async_signatures_match() -> None:
    for fn in (main.stream_engine_async, main.run_engine_async):
        params = list(inspect.signature(fn).parameters)
        assert params[:3] == ["messages", "engine_config", "trace_path"]


def test_drive_sync_runs_finally_on_early_break() -> None:
    """Regression: early break must trigger the async generator's finally
    block so background tasks / telemetry handles get cleaned up."""
    cleaned_up: list[bool] = []

    async def _producer():
        try:
            for i in range(10):
                yield i
        finally:
            cleaned_up.append(True)

    seen: list[int] = []
    for value in _drive_sync(_producer()):
        seen.append(value)
        if value == 2:
            break

    assert seen == [0, 1, 2]
    assert cleaned_up == [True]


def test_drive_sync_runs_finally_on_consumer_exception() -> None:
    """Regression: an exception raised by the consumer must propagate
    through the sync generator AND trigger the async producer's finally."""
    cleaned_up: list[bool] = []

    async def _producer():
        try:
            for i in range(10):
                yield i
        finally:
            cleaned_up.append(True)

    class Boom(Exception):
        pass

    try:
        for value in _drive_sync(_producer()):
            if value == 1:
                raise Boom
    except Boom:
        pass

    assert cleaned_up == [True]


def test_drive_sync_runs_finally_on_full_consumption() -> None:
    cleaned_up: list[bool] = []

    async def _producer():
        try:
            yield 1
            yield 2
        finally:
            cleaned_up.append(True)

    assert list(_drive_sync(_producer())) == [1, 2]
    assert cleaned_up == [True]
