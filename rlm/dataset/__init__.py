"""Dataset layer: descriptor-driven byte-offset index + random-access reader.

A trace dataset can be hundreds of GB of JSONL. Loading it into memory isn't
possible, so this layer builds a compact summary index (lightweight metadata
+ byte offsets) once per dataset, and reads individual traces on demand by
seeking to their offsets. The index and the toolkit are descriptor-driven —
every dataset is described by a ``DatasetDescriptor`` and the rest adapts.
"""

from __future__ import annotations

from dataset.descriptor import (
    DatasetDescriptor,
    FormatMapping,
    HFMapping,
    Label,
    Metric,
    OpenInferenceMapping,
    get_nested,
)
from dataset.indexer import TraceSummary, build_index, scan_dataset
from dataset.reader import TraceReader
from dataset.store import IndexStore


def __getattr__(name: str):
    """Lazy imports for the autodetect helpers.

    Keeps ``from dataset import DatasetDescriptor`` a fast path for callers
    that don't need the heuristic inference machinery.
    """
    if name in {"infer_descriptor", "descriptor_to_python", "InferenceReport"}:
        from dataset import autodetect as _ad
        return getattr(_ad, name)
    raise AttributeError(name)


__all__ = [
    "DatasetDescriptor",
    "FormatMapping",
    "HFMapping",
    "IndexStore",
    "InferenceReport",
    "Label",
    "Metric",
    "OpenInferenceMapping",
    "TraceReader",
    "TraceSummary",
    "build_index",
    "descriptor_to_python",
    "get_nested",
    "infer_descriptor",
    "scan_dataset",
]
