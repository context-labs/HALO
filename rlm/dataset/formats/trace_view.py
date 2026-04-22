"""Format-agnostic view over a single agent trace record.

Every consumer (indexer, inspect_trace tool, search_trace tool, synthesize
tool) reads traces through this interface instead of poking into the raw
record shape. The two concrete implementations — :mod:`dataset.formats.hf`
and :mod:`dataset.formats.openinference` — translate between their
on-disk format and these canonical accessors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class ToolCallView:
    """One tool invocation inside a trace.

    ``arguments`` is always a JSON string (may be empty). ``result`` is the
    tool's stringified output, if the next matching message carried one.
    """

    id: str | None
    name: str
    arguments: str
    result: str | None = None


@dataclass
class MessageView:
    """One assistant / user / tool message in the ordered trajectory."""

    role: str
    content: str
    tool_calls: list[ToolCallView] = field(default_factory=list)
    # For ``role == "tool"`` this is the id of the assistant tool_call
    # this message is responding to. None otherwise.
    tool_call_id: str | None = None


@dataclass
class DocumentView:
    """One document attached to a trace (for datasets that carry corpus files)."""

    path: str
    content: str | None = None


class TraceView:
    """Abstract view over one trace record.

    Subclasses wrap the raw dict + descriptor and expose the canonical
    accessors below. All accessors should be cheap — typically O(span
    count) for OpenInference, O(message count) for HF — and are allowed
    to return empty/None rather than raise when a field is missing.
    """

    # --- identity / semantics ---
    @property
    def id(self) -> str:
        raise NotImplementedError

    @property
    def query(self) -> str:
        """The top-level user question the agent was asked. Never None;
        returns the empty string when unavailable."""
        raise NotImplementedError

    @property
    def final_answer(self) -> Any | None:
        """The final assistant answer. May be a string, a dict, or None."""
        raise NotImplementedError

    @property
    def ground_truth(self) -> Any | None:
        """Ground-truth signal, if the dataset carries one."""
        return None

    @property
    def outcome(self) -> float | None:
        """Shorthand for ``metric_value(descriptor.primary_metric.name)``."""
        return None

    def metric_value(self, name: str) -> float | None:
        """Resolve a named :class:`dataset.descriptor.Metric` against this
        trace. Returns None when the metric doesn't exist on the
        descriptor or when the source is missing on this record."""
        return None

    @property
    def labels(self) -> dict[str, str]:
        """Categorical labels for filter/group (short name → value)."""
        return {}

    # --- trajectory ---
    def messages(self) -> Iterator[MessageView]:
        """Ordered assistant / user / tool messages in the trajectory."""
        raise NotImplementedError

    def tool_calls(self) -> Iterator[ToolCallView]:
        """Flat iterator over every tool call in the trajectory."""
        for msg in self.messages():
            for tc in msg.tool_calls:
                yield tc

    # --- auxiliary ---
    def documents(self) -> list[DocumentView]:
        """Documents bundled with the trace (code-repo datasets use these);
        empty for datasets that don't carry documents."""
        return []

    @property
    def turns_used(self) -> int | None:
        return None

    @property
    def tool_errors(self) -> int:
        return 0

    @property
    def tool_calls_total(self) -> int:
        """Total count — cached or computed. Defaults to iterating."""
        return sum(1 for _ in self.tool_calls())

    @property
    def usage(self) -> dict[str, int]:
        """``{prompt_tokens, completion_tokens, total_tokens}`` — missing
        keys allowed, missing dict becomes empty."""
        return {}
