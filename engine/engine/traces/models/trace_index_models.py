from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TraceIndexRow(BaseModel):
    """One trace's index entry: byte offsets/lengths into the JSONL plus rollup fields for query/overview.

    Offsets and lengths let TraceStore seek directly to a trace's spans without
    rescanning. Rollups (services, models, agents, tokens, has_errors) feed
    ``query_traces``, ``count_traces``, and ``get_overview`` cheaply.
    """

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
    """Sidecar meta file: schema version, total trace count, and a stat fingerprint of the source JSONL.

    ``source_size`` and ``source_mtime_ns`` are restatted on every
    ``ensure_index_exists`` call and compared against the stored values to
    detect a stale index when the underlying trace JSONL has changed. This is
    intentionally not a content hash — appends bump size and any write bumps
    mtime, which is sufficient for a local, append-mostly trace log.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    trace_count: int = Field(ge=0)
    source_size: int = Field(ge=0)
    source_mtime_ns: int = Field(ge=0)
