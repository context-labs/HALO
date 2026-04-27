from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class SpanStatus(BaseModel):
    """OTel span status: code (e.g. ``STATUS_CODE_ERROR``) and optional human message."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str = ""


class SpanResource(BaseModel):
    """OTel resource block; ``attributes`` carries service.name, service.version, deployment.environment, etc."""

    model_config = ConfigDict(extra="forbid")

    attributes: dict[str, Any]


class SpanScope(BaseModel):
    """OTel instrumentation scope (e.g. ``@arizeai/openinference-instrumentation-anthropic``)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = ""


class SpanRecord(BaseModel):
    """One canonical flat span — one JSONL row of the trace input file.

    Models the top-level OTel fields strongly while leaving ``attributes`` and
    ``resource.attributes`` as typed JSON maps so original upstream OpenInference
    keys (``llm.*``, ``inference.*``, etc.) stay accessible to view/search/render.
    """

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
