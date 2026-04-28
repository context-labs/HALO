from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from engine.model_config import ModelConfig


class AgentConfig(BaseModel):
    """Identity, system prompt, model binding, and turn cap for one Engine agent (root or subagent)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    instructions: str
    model: ModelConfig
    maximum_turns: int = Field(gt=0)
