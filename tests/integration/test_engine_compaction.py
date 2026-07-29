from __future__ import annotations

from pathlib import Path

import pytest

import engine.agents.agent_context as agent_context_module
import engine.main as engine_main
from engine.agents.agent_config import AgentConfig
from engine.agents.agent_context_items import AgentContextItem
from engine.engine_config import EngineConfig
from engine.model_config import ModelConfig
from engine.models.messages import AgentMessage
from tests._sdk_events import assistant_message_event
from tests.probes.probe_kit import FakeRunner, make_final_answer


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
        # Compaction's only consumer is the run checkpoint; with checkpoints
        # off the engine skips it entirely (see
        # test_happy_path_makes_no_compaction_calls). This test verifies the
        # configured compactor is what gets used, so it runs the one path
        # that compacts.
        emit_run_checkpoints=True,
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

    async def fake_compact(*, client, compaction_model, item: AgentContextItem) -> str:
        del client, compaction_model
        compacted_items.append((item.item_id, item.role))
        return f"summary for {item.item_id}"

    monkeypatch.setattr(agent_context_module, "compact", fake_compact)

    runner = FakeRunner([*make_final_answer("Final answer.")])
    monkeypatch.setattr("agents.Runner.run_streamed", runner.run_streamed)
    results = await engine_main.run_engine_async(
        [AgentMessage(role="user", content="Summarize the dataset.")],
        _config(),
        trace_path,
    )

    assert any(item.final for item in results)
    assert compacted_items == [
        ("in-0", "user"),
        ("tool-call-call-final", "assistant"),
        ("tool-result-call-final", "tool"),
    ]
    assert runner.calls[0]["max_turns"] == 4
