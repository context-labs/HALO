"""Trace-record format adapters.

OpenInference is the canonical on-disk shape for new datasets. HF is a
legacy format kept only so existing Catalyst dumps keep working until
they're translated.

Dispatch is **by mapping type**, not by a string tag — adding a new
format is a matter of writing a :class:`TraceView` subclass and
registering it in :func:`view_from_record`.
"""

from __future__ import annotations

from dataset.descriptor import (
    ClaudeCodeMapping,
    DatasetDescriptor,
    HFMapping,
    OpenInferenceMapping,
)
from dataset.formats.trace_view import (
    DocumentView,
    MessageView,
    ToolCallView,
    TraceView,
)

__all__ = [
    "DocumentView",
    "MessageView",
    "ToolCallView",
    "TraceView",
    "view_from_record",
]


def view_from_record(record: dict, descriptor: DatasetDescriptor) -> TraceView:
    """Wrap a raw on-disk record in the appropriate ``TraceView`` for the
    descriptor's ``mapping`` type. Sole entry point used by indexer + tools.
    """
    m = descriptor.mapping
    if isinstance(m, ClaudeCodeMapping):
        from dataset.formats.claude_code import ClaudeCodeTraceView
        return ClaudeCodeTraceView(record, descriptor)
    if isinstance(m, OpenInferenceMapping):
        from dataset.formats.openinference import OpenInferenceTraceView
        return OpenInferenceTraceView(record, descriptor)
    if isinstance(m, HFMapping):
        from dataset.formats.hf import HFTraceView
        return HFTraceView(record, descriptor)
    raise TypeError(
        f"unknown format mapping type {type(m).__name__!r} on descriptor "
        f"{descriptor.id!r}; expected HFMapping, OpenInferenceMapping, or "
        f"ClaudeCodeMapping"
    )
