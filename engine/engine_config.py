from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from engine.agents.agent_config import AgentConfig
from engine.model_config import ModelConfig
from engine.model_provider_config import ModelProviderConfig
from engine.traces.models.trace_index_config import TraceIndexConfig


class EngineConfig(BaseModel):
    """Top-level configuration for one Engine run.

    Composes per-domain configs (agents, model bindings, trace index, sandbox) plus
    the compaction thresholds and depth/parallelism caps that bound a run.

    ``synthesis_model`` and ``compaction_model`` are required and carry no
    default — a hardcoded default model name would silently route to the
    wrong provider when ``model_provider`` targets a non-OpenAI endpoint.
    Both roles are plain summarization, so a small, cheap model the
    configured provider serves (e.g. ``gpt-4.1-nano`` on OpenAI) is
    recommended over the agents' analysis model.

    ``repo_path`` is the local source checkout that produced the traces. When
    set, the engine opens it at run start and registers the read-only code
    tools (``glob_files``/``grep_files``/``read_file``) so agents can cross-
    reference findings with the implementation and cite ``file:line``. ``None``
    means the feature is off — the code tools are never registered. The path is
    validated at run start (like ``trace_path``), not at config-parse time.
    """

    model_config = ConfigDict(extra="forbid")

    root_agent: AgentConfig
    subagent: AgentConfig
    synthesis_model: ModelConfig
    compaction_model: ModelConfig
    model_provider: ModelProviderConfig = Field(default_factory=ModelProviderConfig)
    trace_index: TraceIndexConfig = Field(default_factory=TraceIndexConfig)
    text_message_compaction_keep_last_messages: int = Field(default=12, ge=0)
    tool_call_compaction_keep_last_turns: int = Field(default=3, ge=0)
    maximum_depth: int = Field(default=2, ge=0)
    maximum_parallel_subagents: int = Field(default=4, gt=0)
    repo_path: Path | None = None
