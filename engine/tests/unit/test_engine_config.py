from __future__ import annotations

from engine.agents.agent_config import AgentConfig
from engine.engine_config import EngineConfig
from engine.model_config import ModelConfig


def _agent(name: str) -> AgentConfig:
    return AgentConfig(
        name=name,
        instructions="",
        model=ModelConfig(name="claude-sonnet-4-5"),
        maximum_turns=10,
    )


def test_engine_config_defaults() -> None:
    cfg = EngineConfig(
        root_agent=_agent("root"),
        subagent=_agent("sub"),
        synthesis_model=ModelConfig(name="claude-haiku-4-5"),
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
    )
    assert cfg.text_message_compaction_keep_last_messages == 12
    assert cfg.tool_call_compaction_keep_last_messages == 6
    assert cfg.maximum_depth == 2
    assert cfg.maximum_parallel_subagents == 4
