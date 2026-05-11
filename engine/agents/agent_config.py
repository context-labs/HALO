from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from engine.model_config import ModelConfig


class AgentConfig(BaseModel):
    """Identity, model binding, and turn cap for one Engine agent (root or subagent).

    The system prompt is fixed by the engine (see ``SYSTEM_PROMPT`` in
    ``engine/agents/prompt_templates.py``) — it's the usage manual for the
    built-in trace tools.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    model: ModelConfig
    maximum_turns: int = Field(gt=0)
