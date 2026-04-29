from __future__ import annotations

from engine.agents.agent_config import AgentConfig
from engine.engine_config import EngineConfig
from engine.model_config import ModelConfig
from engine.model_provider_config import ModelProviderConfig


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
    assert cfg.tool_call_compaction_keep_last_turns == 3
    assert cfg.maximum_depth == 2
    assert cfg.maximum_parallel_subagents == 4
    assert cfg.model_provider == ModelProviderConfig()
    assert cfg.model_provider.base_url is None
    assert cfg.model_provider.api_key is None


def test_engine_config_accepts_model_provider() -> None:
    cfg = EngineConfig(
        root_agent=_agent("root"),
        subagent=_agent("sub"),
        synthesis_model=ModelConfig(name="claude-haiku-4-5"),
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        model_provider=ModelProviderConfig(
            base_url="https://api.anthropic.com/v1/",
            api_key="sk-ant-test",
        ),
    )
    assert cfg.model_provider.base_url == "https://api.anthropic.com/v1/"
    assert cfg.model_provider.api_key == "sk-ant-test"
