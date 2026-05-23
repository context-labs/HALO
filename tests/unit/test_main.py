from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from openai import AsyncOpenAI

import engine.agents.agent_context as agent_context_module
import engine.main as engine_main
from engine.agents.agent_config import AgentConfig
from engine.agents.agent_context_items import AgentContextItem
from engine.engine_config import EngineConfig
from engine.main import _drive_sync
from engine.model_config import ModelConfig
from engine.models.messages import AgentMessage
from tests._sdk_events import assistant_message_event
from tests.probes.probe_kit import FakeRunner


async def _noop_compact(
    *,
    client: AsyncOpenAI,
    compaction_model: ModelConfig,
    item: AgentContextItem,
) -> str:
    del client, compaction_model, item
    return ""


def test_public_entrypoints_exist_and_are_async() -> None:
    assert inspect.isasyncgenfunction(engine_main.stream_engine_async)
    assert inspect.iscoroutinefunction(engine_main.run_engine_async)
    assert callable(engine_main.stream_engine)
    assert callable(engine_main.run_engine)


def test_async_signatures_match() -> None:
    for fn in (engine_main.stream_engine_async, engine_main.run_engine_async):
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


def _assistant_text(text: str):
    return assistant_message_event(item_id="msg-1", text=text)


def _config() -> EngineConfig:
    agent = AgentConfig(
        name="root",
        model=ModelConfig(name="gpt-5.4-mini"),
        maximum_turns=4,
    )
    return EngineConfig(
        root_agent=agent,
        subagent=agent.model_copy(update={"name": "sub"}),
        synthesis_model=ModelConfig(name="gpt-5.4-mini"),
        compaction_model=ModelConfig(name="gpt-5.4-mini"),
        text_message_compaction_keep_last_messages=0,
        tool_call_compaction_keep_last_turns=0,
        maximum_depth=0,
        maximum_parallel_subagents=1,
    )


@pytest.mark.asyncio
async def test_engine_wires_configured_client_via_run_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixtures_dir: Path,
) -> None:
    """Regression for the wired_tools-vs-stream_engine_async asymmetry.

    The engine must pass its ``AsyncOpenAI`` to the SDK via
    ``RunConfig.model_provider`` for every ``Runner.run_streamed`` call.
    Process-global ``set_default_openai_client`` is fragile: subagent
    tool factories invoked outside ``stream_engine_async`` (e.g. via
    ``tests/integration/tool_isolation_kit.wired_tools``) never see
    the default, and the SDK silently falls back to an ``AsyncOpenAI``
    built from env vars — losing ``default_headers`` and the
    deterministic close path. Wire it per-call instead.
    """
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())

    class _StubAsyncOpenAI:
        def __init__(
            self,
            *,
            base_url: str | None = None,
            api_key: str | None = None,
            default_headers: dict[str, str] | None = None,
        ) -> None:
            del base_url, api_key, default_headers
            self.close = AsyncMock()

    stub_client_instance: _StubAsyncOpenAI | None = None

    def _capture_client(
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> _StubAsyncOpenAI:
        nonlocal stub_client_instance
        stub_client_instance = _StubAsyncOpenAI(
            base_url=base_url, api_key=api_key, default_headers=default_headers
        )
        return stub_client_instance

    monkeypatch.setattr(engine_main, "AsyncOpenAI", _capture_client)
    monkeypatch.setattr(agent_context_module, "compact", _noop_compact)

    runner = FakeRunner([_assistant_text("Final.\n<final/>")])
    monkeypatch.setattr("agents.Runner.run_streamed", runner.run_streamed)

    await engine_main.run_engine_async(
        [AgentMessage(role="user", content="hi")], _config(), trace_path
    )

    assert stub_client_instance is not None
    assert len(runner.calls) >= 1
    run_config = runner.calls[0]["run_config"]
    model_provider = run_config.model_provider
    assert isinstance(model_provider, engine_main.OpenAIProvider)
    # ``OpenAIProvider`` stores the passed client on ``_client`` and short-
    # circuits lazy construction; verify the engine's client is what the
    # SDK will use rather than an env-var-built fallback.
    assert model_provider._get_client() is stub_client_instance
