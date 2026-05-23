from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from openai import AsyncOpenAI

import engine.agents.agent_context as agent_context_module
import engine.main as engine_main
from engine.agents.agent_config import AgentConfig
from engine.agents.agent_context_items import AgentContextItem
from engine.engine_config import EngineConfig
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


def _assistant_text(text: str):
    return assistant_message_event(item_id="msg-1", text=text)


def _config() -> EngineConfig:
    agent = AgentConfig(
        name="root",
        model=ModelConfig(name="gpt-5.4-mini"),
        maximum_turns=2,
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


class _StubAsyncOpenAI:
    """Stand-in for ``AsyncOpenAI`` that records ``close()`` calls."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        del base_url, api_key, default_headers
        self.close = AsyncMock()


def _install_stub_client(monkeypatch: pytest.MonkeyPatch) -> _StubAsyncOpenAI:
    """Patch ``engine.main.AsyncOpenAI`` to return a fresh stub. Returns the
    stub so the test can assert on ``close`` lifecycle."""
    stub_client = _StubAsyncOpenAI()

    def _build_stub(
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> _StubAsyncOpenAI:
        del base_url, api_key, default_headers
        return stub_client

    monkeypatch.setattr(engine_main, "AsyncOpenAI", _build_stub)
    return stub_client


@pytest.mark.asyncio
async def test_client_closed_on_normal_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixtures_dir: Path,
) -> None:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())

    stub_client = _install_stub_client(monkeypatch)
    monkeypatch.setattr(agent_context_module, "compact", _noop_compact)

    runner = FakeRunner([_assistant_text("Final answer.\n<final/>")])
    monkeypatch.setattr("agents.Runner.run_streamed", runner.run_streamed)

    await engine_main.run_engine_async(
        [AgentMessage(role="user", content="Summarize the dataset.")],
        _config(),
        trace_path,
    )

    stub_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_client_closed_on_early_consumer_break(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixtures_dir: Path,
) -> None:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())

    stub_client = _install_stub_client(monkeypatch)
    monkeypatch.setattr(agent_context_module, "compact", _noop_compact)

    runner = FakeRunner([_assistant_text("Final answer.\n<final/>")])
    monkeypatch.setattr("agents.Runner.run_streamed", runner.run_streamed)

    agen = engine_main.stream_engine_async(
        [AgentMessage(role="user", content="Summarize the dataset.")],
        _config(),
        trace_path,
    )

    # Consume one event, then close — exercises the GeneratorExit teardown path.
    async for _event in agen:
        break
    await agen.aclose()

    stub_client.close.assert_awaited_once()
