from __future__ import annotations

from typing import Literal

from agents.model_settings import ModelSettings
from openai.types.shared import Reasoning
from pydantic import BaseModel, ConfigDict, Field

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
"""Reasoning effort levels supported by OpenAI-compatible reasoning models.

Mirrors ``openai.types.shared.ReasoningEffort`` so values stay in lockstep
with the upstream API.
"""


def max_reasoning_effort_for_model(name: str) -> ReasoningEffort | None:
    """Highest reasoning effort known to be supported by ``name``.

    Returns ``None`` for models we don't recognize as reasoning models, so
    the engine can omit the parameter entirely on providers that would
    400 on it. Returns the strongest documented effort otherwise.

    Conservative prefix match against OpenAI's published support matrix:
    ``xhigh`` is reserved for families that explicitly support it
    (``gpt-5.1-codex-max`` and HALO's default ``gpt-5.5``). Other ``gpt-5``
    and o-series families max at ``"high"``. Update this table when new
    families ship.

    Explicit overrides on ``ModelConfig.reasoning_effort`` always win — see
    ``ModelConfig.effective_reasoning_effort``.
    """
    n = name.lower()

    if n.startswith(("gpt-5.1-codex-max", "gpt-5.5")):
        return "xhigh"

    if n.startswith(("gpt-5", "o1", "o3", "o4")):
        return "high"

    return None


class ModelConfig(BaseModel):
    """LLM model binding plus generation knobs shared by agent, synthesis, and compaction calls.

    ``name`` is an unconstrained string — whatever model identifier the
    configured OpenAI-compatible endpoint expects (``gpt-4o``,
    ``claude-opus-4-7``, ``meta-llama/Llama-3.1-70B-Instruct``, etc.). The
    routing endpoint is set on ``EngineConfig.model_provider``.

    ``reasoning_effort`` is the explicit override. When unset, the engine
    falls back to ``max_reasoning_effort_for_model(name)`` so reasoning
    models get the strongest documented effort by default and non-reasoning
    models stay un-parameterized. See ``effective_reasoning_effort``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    maximum_output_tokens: int | None = Field(default=None, gt=0)
    parallel_tool_calls: bool = True
    reasoning_effort: ReasoningEffort | None = None

    def effective_reasoning_effort(self) -> ReasoningEffort | None:
        """Resolve the reasoning effort to actually send.

        Explicit ``reasoning_effort`` wins. Otherwise default to the model
        family's documented max (``None`` for unknown / non-reasoning
        families).
        """
        if self.reasoning_effort is not None:
            return self.reasoning_effort
        return max_reasoning_effort_for_model(self.name)

    def to_sdk_model_settings(self) -> ModelSettings:
        """Project this config onto the OpenAI Agents SDK's ``ModelSettings``.

        Only fields the engine actively configures are forwarded; the rest
        stay at SDK defaults so provider-side defaults still apply.
        ``reasoning`` is only set when an effort was explicitly requested
        or implied by the model family, which keeps non-reasoning models
        from receiving an unsupported parameter.
        """
        effort = self.effective_reasoning_effort()
        return ModelSettings(
            temperature=self.temperature,
            max_tokens=self.maximum_output_tokens,
            parallel_tool_calls=self.parallel_tool_calls,
            reasoning=Reasoning(effort=effort) if effort is not None else None,
        )
