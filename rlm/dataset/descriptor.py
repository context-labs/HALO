"""Dataset descriptor — the single source of truth for how halo reads
one agent-trace dataset.

A dataset is described by:

1. **Identity** — id, name, source path, origin model, free-form blurb.
2. **Format mapping** — a typed object that tells the reader layer *how*
   each record is laid out on disk. Two concrete mappings today:
   :class:`HFMapping` (flat JSONL; legacy Catalyst shape) and
   :class:`OpenInferenceMapping` (one OTLP span tree per line, emitted
   by the ``otel-interceptor`` compact step). Adding a new format is a
   matter of writing a new mapping + adapter pair — the descriptor
   changes its ``mapping`` attribute type, no other field shifts.
3. **Semantics** — a primary :class:`Metric`, optional secondary
   :class:`Metric`\\ s, a ground-truth source, and a list of
   :class:`Label`\\ s. These are all *source-agnostic*: each carries a
   ``source`` string that the mapping/adapter knows how to resolve (a
   dotted field path for HF, a span attribute key for OpenInference).
4. **UX** — seed questions, display names, etc.

Security note
-------------
Descriptor modules are ``exec``-ed at startup to load their
``DESCRIPTOR`` attribute. Only install descriptor files you trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union


BucketName = Literal["perfect", "partial", "zero"]


def get_nested(obj: Any, path: str | None) -> Any:
    """Walk a dotted path into a nested dict. Returns None for any missing part.

    ``get_nested({"a": {"b": 1}}, "a.b") == 1``
    ``get_nested({"a": None}, "a.b") is None``
    ``get_nested(x, None) is None``
    """
    if not path:
        return None
    current: Any = obj
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


# ----------------------------------------------------------------------
# Semantics — metrics + labels
# ----------------------------------------------------------------------


@dataclass
class Metric:
    """A single scalar metric on a trace.

    ``source`` is a *format-agnostic reference*. For :class:`HFMapping`
    datasets it's interpreted as a dotted field path into each record
    (``"metadata.file_recall"``). For :class:`OpenInferenceMapping`
    datasets it's interpreted as a span attribute key
    (``"file_recall"``) looked up on the root span first, then any span.
    The adapter layer handles the interpretation; the descriptor just
    stores the string.

    ``kind`` is a small, meaningful enumeration — not free-form text.
    ``score_01`` is the common case (a score in [0, 1]); ``binary`` is a
    0/1 flag; the other kinds let us format + bucket integers, seconds,
    or dollar costs correctly without pretending they live in [0, 1].

    ``higher_is_better`` flips the direction of every bucket predicate.
    ``perfect_threshold`` / ``zero_threshold`` are the cut-offs for the
    ``perfect``/``zero`` buckets; the remainder is ``partial``.
    """

    name: str                                     # stable id: "file_recall"
    source: str                                   # path or attribute key
    kind: Literal["score_01", "binary", "int", "seconds", "dollars"] = "score_01"
    display_name: str | None = None               # defaults to ``name``
    higher_is_better: bool = True
    perfect_threshold: float = 0.999
    zero_threshold: float = 0.001

    @property
    def label(self) -> str:
        return self.display_name or self.name

    # --- Bucket predicates (direction-aware) ---

    def is_perfect(self, v: float) -> bool:
        return v >= self.perfect_threshold if self.higher_is_better else v <= self.perfect_threshold

    def is_zero(self, v: float) -> bool:
        return v < self.zero_threshold if self.higher_is_better else v > self.zero_threshold

    def is_partial(self, v: float) -> bool:
        return not self.is_perfect(v) and not self.is_zero(v)

    def bucket_of(self, v: float) -> BucketName:
        if self.is_perfect(v):
            return "perfect"
        if self.is_zero(v):
            return "zero"
        return "partial"


@dataclass
class Label:
    """A categorical dimension on a trace, used for filter / group-by.

    ``name`` is the stable short id used in filter keys and as the map
    key in :class:`TraceSummary.labels`. ``source`` is the format-
    agnostic reference (dotted path for HF, span attribute key for OI).
    """

    name: str
    source: str
    display_name: str | None = None

    @property
    def label(self) -> str:
        return self.display_name or self.name


# ----------------------------------------------------------------------
# Format mappings — one per supported on-disk shape
# ----------------------------------------------------------------------


@dataclass
class HFMapping:
    """How to read the legacy Catalyst flat-JSONL trace shape.

    Every field is a dotted path into each record. Setting one to
    ``None`` means the dataset doesn't carry that information (e.g.
    ``final_answer_field=None`` for eval sets where only the outcome
    score matters).
    """

    id_field: str = "id"
    query_field: str = "query"
    messages_field: str = "messages"
    final_answer_field: str | None = "final_answer"
    documents_field: str | None = None
    document_path_field: str | None = "path"

    # Metadata block.
    usage_field: str | None = "metadata.usage"
    turns_field: str | None = "metadata.turns_used"
    tool_calls_total_field: str | None = "metadata.total_tool_calls"
    tool_errors_field: str | None = "metadata.tool_errors"


@dataclass
class OpenInferenceMapping:
    """How to read OpenInference-shaped OTLP span trees.

    OpenInference has canonical attribute names (``llm.input_messages.*``,
    ``tool.name``, ``openinference.span.kind``, …), so there's almost
    nothing to configure — the adapter reads them directly. The only
    descriptor-level dial is ``id_attribute``: if your team emits a
    custom trace-id attribute on the root span (e.g. ``query_id``),
    name it here and the view will prefer it over the record's
    ``traceId``.
    """

    id_attribute: str | None = None


FormatMapping = Union[HFMapping, OpenInferenceMapping]


# ----------------------------------------------------------------------
# The descriptor
# ----------------------------------------------------------------------


@dataclass
class DatasetDescriptor:
    """Everything halo needs to load and analyze one trace dataset."""

    # --- Identity ---
    id: str                                       # URL-safe slug; route segment
    name: str                                     # human display name
    source_path: Path                             # raw JSONL on disk
    mapping: FormatMapping                        # typed, NOT a string tag

    source_model: str | None = None
    description: str | None = None

    # --- Semantics ---
    primary_metric: Metric | None = None          # the headline metric
    secondary_metrics: list[Metric] = field(default_factory=list)
    ground_truth_source: str | None = None        # path (HF) / attr key (OI)
    labels: list[Label] = field(default_factory=list)

    # --- UX ---
    seed_questions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Label names must be unique — they're dict keys in ``TraceSummary.labels``.
        names = [lbl.name for lbl in self.labels]
        if len(names) != len(set(names)):
            dups = [n for n in set(names) if names.count(n) > 1]
            raise ValueError(
                f"Duplicate label name(s) in descriptor {self.id!r}: {dups}. "
                f"Each label must have a unique ``name``."
            )
        # Metric names must be unique across primary + secondary.
        metric_names: list[str] = []
        if self.primary_metric is not None:
            metric_names.append(self.primary_metric.name)
        metric_names.extend(m.name for m in self.secondary_metrics)
        if len(metric_names) != len(set(metric_names)):
            dups = [n for n in set(metric_names) if metric_names.count(n) > 1]
            raise ValueError(
                f"Duplicate metric name(s) in descriptor {self.id!r}: {dups}"
            )

    # --- Convenience ---

    @property
    def format(self) -> Literal["hf", "openinference"]:
        return "openinference" if isinstance(self.mapping, OpenInferenceMapping) else "hf"

    @property
    def has_documents(self) -> bool:
        if isinstance(self.mapping, HFMapping):
            return bool(self.mapping.documents_field and self.mapping.document_path_field)
        return False   # OI has no canonical documents attribute yet

    @property
    def has_primary_metric(self) -> bool:
        return self.primary_metric is not None

    @property
    def has_ground_truth(self) -> bool:
        return bool(self.ground_truth_source)

    @property
    def label_names(self) -> list[str]:
        return [lbl.name for lbl in self.labels]

    def label(self, name: str) -> Label | None:
        for lbl in self.labels:
            if lbl.name == name:
                return lbl
        return None

    def metric(self, name: str) -> Metric | None:
        if self.primary_metric is not None and self.primary_metric.name == name:
            return self.primary_metric
        for m in self.secondary_metrics:
            if m.name == name:
                return m
        return None

    def default_index_path(self, index_dir: Path) -> Path:
        """Where the summary index for this dataset lives."""
        return index_dir / f"{self.id}.index.jsonl"
