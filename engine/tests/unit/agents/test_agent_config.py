from __future__ import annotations

from engine.agents.agent_config import AgentConfig
from engine.model_config import ModelConfig


def test_agent_config_constructs() -> None:
    cfg = AgentConfig(
        name="root",
        instructions="You are root.",
        model=ModelConfig(name="claude-opus-4-7"),
        maximum_turns=20,
    )
    assert cfg.name == "root"
    assert cfg.maximum_turns == 20
    assert cfg.model.name == "claude-opus-4-7"
