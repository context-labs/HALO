from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    """LLM model binding plus generation knobs shared by agent, synthesis, and compaction calls.

    ``name`` is an unconstrained string — whatever model identifier the
    configured OpenAI-compatible endpoint expects (``gpt-4o``,
    ``claude-opus-4-7``, ``meta-llama/Llama-3.1-70B-Instruct``, etc.). The
    routing endpoint is set on ``EngineConfig.model_provider``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    maximum_output_tokens: int | None = Field(default=None, gt=0)
    parallel_tool_calls: bool = True
