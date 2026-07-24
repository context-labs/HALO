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
    refusal_retries: int = Field(default=0, ge=0)
    # Root-only: how many times to reprompt when the SDK run ends without
    # the agent having finalized (no ``final_answer`` call and no legacy
    # ``<final/>`` sentinel). Models that narrate a plan without calling a
    # tool silently end the SDK run; one nudge recovers most of those.
    # Ignored for subagents (their runner is constructed without it).
    final_answer_reprompts: int = Field(default=1, ge=0)
