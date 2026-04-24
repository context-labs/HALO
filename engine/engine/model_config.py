from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

AvailableModelName: TypeAlias = Literal[
    "claude-opus-4-7",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "gpt-5.4",
    "gpt-5.4-mini",
]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: AvailableModelName
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    maximum_output_tokens: int | None = Field(default=None, gt=0)
    parallel_tool_calls: bool = True
