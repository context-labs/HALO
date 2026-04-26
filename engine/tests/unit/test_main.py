from __future__ import annotations

import inspect

import engine.main as main
from engine.agents.agent_config import AgentConfig
from engine.agents.prompt_templates import render_root_system_prompt
from engine.engine_config import EngineConfig
from engine.model_config import ModelConfig
from engine.models.engine_output import AgentOutputItem
from engine.models.messages import AgentMessage


def test_public_entrypoints_exist_and_are_async() -> None:
    assert inspect.isasyncgenfunction(main.stream_engine_async)
    assert inspect.iscoroutinefunction(main.run_engine_async)
    assert callable(main.stream_engine)
    assert callable(main.run_engine)
    assert callable(main.to_messages_array)


def test_async_signatures_match() -> None:
    for fn in (main.stream_engine_async, main.run_engine_async):
        params = list(inspect.signature(fn).parameters)
        assert params[:3] == ["messages", "engine_config", "trace_path"]


def _cfg() -> EngineConfig:
    agent = AgentConfig(
        name="root",
        instructions="Be brief.",
        model=ModelConfig(name="gpt-5.4-mini"),
        maximum_turns=4,
    )
    return EngineConfig(
        root_agent=agent,
        subagent=agent.model_copy(update={"name": "sub"}),
        synthesis_model=ModelConfig(name="gpt-5.4-mini"),
        compaction_model=ModelConfig(name="gpt-5.4-mini"),
    )


def _output_item(role: str, content: str | None, *, depth: int = 0, agent_id: str = "root") -> AgentOutputItem:
    return AgentOutputItem(
        sequence=0,
        agent_id=agent_id,
        parent_agent_id=None,
        parent_tool_call_id=None,
        agent_name=agent_id,
        depth=depth,
        item=AgentMessage(role=role, content=content),
    )


def test_to_messages_array_first_call() -> None:
    cfg = _cfg()
    sys_text = render_root_system_prompt(
        instructions=cfg.root_agent.instructions,
        maximum_depth=cfg.maximum_depth,
        maximum_parallel_subagents=cfg.maximum_parallel_subagents,
    )
    inputs = [AgentMessage(role="user", content="Q1")]
    results = [_output_item("assistant", "A1")]
    out = main.to_messages_array(inputs, results, cfg)
    assert [(m.role, m.content) for m in out] == [
        ("system", sys_text),
        ("user", "Q1"),
        ("assistant", "A1"),
    ]


def test_to_messages_array_dedups_caller_system() -> None:
    cfg = _cfg()
    sys_text = render_root_system_prompt(
        instructions=cfg.root_agent.instructions,
        maximum_depth=cfg.maximum_depth,
        maximum_parallel_subagents=cfg.maximum_parallel_subagents,
    )
    # Caller passes back the prior conversation including the system message
    inputs = [
        AgentMessage(role="system", content=sys_text),
        AgentMessage(role="user", content="Q1"),
        AgentMessage(role="assistant", content="A1"),
    ]
    results = [_output_item("assistant", "A2")]
    out = main.to_messages_array(inputs, results, cfg)
    # Only one system message; rest in order
    assert sum(1 for m in out if m.role == "system") == 1
    assert [(m.role, m.content) for m in out] == [
        ("system", sys_text),
        ("user", "Q1"),
        ("assistant", "A1"),
        ("assistant", "A2"),
    ]


def test_to_messages_array_skips_subagent_items() -> None:
    cfg = _cfg()
    inputs = [AgentMessage(role="user", content="Q")]
    results = [
        _output_item("assistant", "child says hi", depth=1, agent_id="sub"),
        _output_item("assistant", "root says hi", depth=0),
    ]
    out = main.to_messages_array(inputs, results, cfg)
    assert "child says hi" not in [m.content for m in out]
    assert "root says hi" in [m.content for m in out]
