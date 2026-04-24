from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class SpanStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str = ""


class SpanResource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attributes: dict[str, Any]


class SpanScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = ""


class SpanRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    trace_id: str
    span_id: str
    parent_span_id: str = ""
    trace_state: str = ""
    name: str
    kind: str
    start_time: str
    end_time: str
    status: SpanStatus
    resource: SpanResource
    scope: SpanScope
    attributes: dict[str, Any]
