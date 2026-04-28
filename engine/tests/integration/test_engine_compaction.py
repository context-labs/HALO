from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import engine.main as engine_main
from engine.agents.agent_config import AgentConfig
from engine.engine_config import EngineConfig
from engine.model_config import ModelConfig
from engine.models.messages import AgentMessage
from tests._sdk_events import assistant_message_event


class _FakeStream:
    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def stream_events(self):
        for event in self._events:
            yield event


class _FakeRunner:
    def __init__(self, events: list[Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._events = events

    def run_streamed(self, **kwargs: Any) -> _FakeStream:
        self.calls.append(kwargs)
        return _FakeStream(self._events)


def _assistant_text(text: str):
    return assistant_message_event(item_id="msg-1", text=text)


def _config() -> EngineConfig:
    agent = AgentConfig(
        name="root",
        instructions="Answer briefly.",
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
async def test_engine_compaction_uses_configured_compactor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixtures_dir: Path,
) -> None:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())

    compacted_items: list[tuple[str, str]] = []

    def fake_compactor_factory(_config: EngineConfig):
        def factory(_execution):
            async def compact(item):
                compacted_items.append((item.item_id, item.role))
                return f"summary for {item.item_id}"

            return compact

        return factory

    monkeypatch.setattr(engine_main, "build_compactor_factory", fake_compactor_factory)

    runner = _FakeRunner([_assistant_text("Final answer.\n<final/>")])
    results = await engine_main.run_engine_async(
        [AgentMessage(role="user", content="Summarize the dataset.")],
        _config(),
        trace_path,
        runner=runner,
    )

    assert any(item.final for item in results)
    assert compacted_items == [("in-0", "user"), ("msg-1", "assistant")]
    assert runner.calls[0]["max_turns"] == 4
