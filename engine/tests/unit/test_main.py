from __future__ import annotations

import inspect

import engine.main as main


def test_public_entrypoints_exist_and_are_async() -> None:
    assert inspect.isasyncgenfunction(main.stream_engine_async)
    assert inspect.iscoroutinefunction(main.run_engine_async)
    assert callable(main.stream_engine)
    assert callable(main.run_engine)


def test_async_signatures_match() -> None:
    for fn in (main.stream_engine_async, main.run_engine_async):
        params = list(inspect.signature(fn).parameters)
        assert params[:3] == ["messages", "engine_config", "trace_path"]
