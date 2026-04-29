from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from engine.model_config import ModelConfig


class AgentConfig(BaseModel):
    """Identity, system prompt, model binding, and turn cap for one Engine agent (root or subagent).

    ``instructions`` is optional. When ``None``, the engine auto-injects
    ``DEFAULT_SYSTEM_PROMPT`` (see ``engine/agents/prompt_templates.py``) at
    prompt-render time, which is a usage manual for the engine's built-in trace
    tools. Pass an explicit string to override.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    instructions: str | None = None
    model: ModelConfig
    maximum_turns: int = Field(gt=0)
