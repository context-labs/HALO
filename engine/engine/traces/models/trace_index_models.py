from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TraceIndexRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    byte_offsets: list[int]
    byte_lengths: list[int]
    span_count: int = Field(ge=0)
    start_time: str
    end_time: str
    has_errors: bool
    service_names: list[str]
    model_names: list[str]
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    project_id: str | None = None
    agent_names: list[str]


class TraceIndexMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    trace_count: int = Field(ge=0)
