from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from agents.tool_context import ToolContext as SdkToolContext

import engine.tools.subagent_tool_factory as subagent_factory
from engine.agents.agent_config import AgentConfig
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.engine_run_state import EngineRunState
from engine.engine_config import EngineConfig
from engine.model_config import ModelConfig
from engine.tools.subagent_result import SubagentToolResult
from engine.tools.subagent_tool_factory import _build_subagent_as_tool
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore


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


def _assistant_text(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="run_item_stream_event",
        item=SimpleNamespace(
            type="message_output_item",
            raw_item=SimpleNamespace(
                id="child-msg-1",
                role="assistant",
                content=[SimpleNamespace(type="output_text", text=text)],
            ),
        ),
    )


def _config() -> EngineConfig:
    root = AgentConfig(
        name="root",
        instructions="Answer briefly.",
        model=ModelConfig(name="gpt-5.4-mini"),
        maximum_turns=4,
    )
    return EngineConfig(
        root_agent=root,
        subagent=root.model_copy(update={"name": "sub", "maximum_turns": 3}),
        synthesis_model=ModelConfig(name="gpt-5.4-mini"),
        compaction_model=ModelConfig(name="gpt-5.4-mini"),
        maximum_depth=1,
        maximum_parallel_subagents=1,
    )


def _tool_context() -> SdkToolContext:
    return SdkToolContext(
        context=None,
        tool_name="call_subagent",
        tool_call_id="parent-call-1",
        tool_arguments='{"input":"How many traces have errors?"}',
    )


@pytest.mark.asyncio
async def test_subagent_tool_streams_child_events_with_parent_linkage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixtures_dir: Path,
) -> None:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=TraceIndexConfig(),
    )
    trace_store = TraceStore.load(trace_path=trace_path, index_path=index_path)

    def fake_compactor_factory(_config: EngineConfig):
        def factory(_execution):
            async def compact(_item):
                return "summary"

            return compact

        return factory

    monkeypatch.setattr(subagent_factory, "build_compactor_factory", fake_compactor_factory)

    cfg = _config()
    output_bus = EngineOutputBus()
    runner = _FakeRunner([_assistant_text("The dataset has one error trace.")])
    run_state = EngineRunState(
        trace_store=trace_store,
        output_bus=output_bus,
        config=cfg,
        sandbox=None,
        runner=runner,
    )
    parent_execution = AgentExecution(
        agent_id="root-1",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )
    run_state.register(parent_execution)

    tool = _build_subagent_as_tool(
        run_state=run_state,
        child_depth=1,
        semaphores_by_depth={1: asyncio.Semaphore(1)},
        parent_execution=parent_execution,
    )

    result_json = await tool.on_invoke_tool(
        _tool_context(),
        '{"input":"How many traces have errors?"}',
    )
    await output_bus.close()
    emitted = [event async for event in output_bus.stream()]
    result = SubagentToolResult.model_validate_json(result_json)

    assert result.answer == "The dataset has one error trace."
    assert result.turns_used == 1
    assert runner.calls[0]["max_turns"] == 3
    assert len(emitted) == 1
    child_item = emitted[0]
    assert child_item.depth == 1
    assert child_item.parent_agent_id == "root-1"
    assert child_item.parent_tool_call_id == "parent-call-1"
