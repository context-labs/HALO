from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from engine.agents.agent_config import AgentConfig
from engine.model_config import ModelConfig
from engine.model_provider_config import ModelProviderConfig
from engine.sandbox.models import SandboxConfig
from engine.traces.models.trace_index_config import TraceIndexConfig


class EngineConfig(BaseModel):
    """Top-level configuration for one Engine run.

    Composes per-domain configs (agents, model bindings, trace index, sandbox) plus
    the compaction thresholds and depth/parallelism caps that bound a run.
    """

    model_config = ConfigDict(extra="forbid")

    root_agent: AgentConfig
    subagent: AgentConfig
    synthesis_model: ModelConfig
    compaction_model: ModelConfig
    model_provider: ModelProviderConfig = Field(default_factory=ModelProviderConfig)
    trace_index: TraceIndexConfig = Field(default_factory=TraceIndexConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    text_message_compaction_keep_last_messages: int = Field(default=12, ge=0)
    tool_call_compaction_keep_last_turns: int = Field(default=3, ge=0)
    maximum_depth: int = Field(default=2, ge=0)
    maximum_parallel_subagents: int = Field(default=4, gt=0)
