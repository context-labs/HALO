"""Discovers and loads all dataset descriptors from the ``catalog`` package.

Every module under ``catalog/`` that exposes a ``DESCRIPTOR`` attribute of
type ``DatasetDescriptor`` is auto-registered. Modules whose names start
with ``_`` are skipped (reserved for registry internals and templates).

If a descriptor module fails to import or construct, the registry logs
the error and continues with the remaining datasets so one bad file
doesn't brick the whole UI.

Security
--------
Descriptor modules are executed as Python at import time. Treat the
``catalog/`` directory as trusted-admin configuration; never populate it
from untrusted sources.
"""

from __future__ import annotations

import importlib
import pkgutil
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from dataset import DatasetDescriptor, IndexStore


@dataclass
class RegisteredDataset:
    """One entry in the registry: descriptor + (optionally loaded) store."""

    descriptor: DatasetDescriptor
    index_path: Path
    store: IndexStore | None = None

    @property
    def is_indexed(self) -> bool:
        return self.index_path.exists()


@dataclass
class DatasetRegistry:
    """All datasets known to this deployment, keyed by descriptor id."""

    index_dir: Path
    entries: dict[str, RegisteredDataset] = field(default_factory=dict)
    # Descriptor modules that failed to import or build a descriptor.
    errors: list[dict[str, str]] = field(default_factory=list)

    def load_store(self, dataset_id: str) -> IndexStore:
        entry = self.entries[dataset_id]
        if entry.store is None:
            if not entry.is_indexed:
                raise FileNotFoundError(
                    f"Index for '{dataset_id}' not found at {entry.index_path}. "
                    f"Run `uv run halo index --dataset {dataset_id}` first."
                )
            entry.store = IndexStore.load(entry.index_path, entry.descriptor)
        return entry.store

    def ids(self) -> list[str]:
        return list(self.entries.keys())


def _iter_catalog_descriptors() -> list[tuple[str, DatasetDescriptor | None, str | None, str | None]]:
    """Return (module_name, descriptor_or_None, error_msg, traceback_str) tuples.

    We collect every catalog module, isolating import and descriptor-build
    failures so a single bad file cannot crash the registry.
    """
    import catalog  # noqa: WPS433 — by design
    out: list[tuple[str, DatasetDescriptor | None, str | None, str | None]] = []
    for _, name, _ in pkgutil.iter_modules(catalog.__path__):
        if name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"catalog.{name}")
        except Exception as e:
            out.append((name, None,
                        f"import failed: {type(e).__name__}: {e}",
                        traceback.format_exc()))
            continue
        descriptor = getattr(module, "DESCRIPTOR", None)
        if descriptor is None:
            out.append((name, None, "module has no DESCRIPTOR attribute", None))
            continue
        if not isinstance(descriptor, DatasetDescriptor):
            out.append((
                name, None,
                f"DESCRIPTOR is a {type(descriptor).__name__}, not DatasetDescriptor",
                None,
            ))
            continue
        out.append((name, descriptor, None, None))
    return out


def build_registry(index_dir: Path) -> DatasetRegistry:
    """Discover all catalog descriptors and register them.

    Stores are loaded lazily (on first ``load_store`` call) so unused
    datasets don't cost RAM. Descriptor-module failures are captured in
    ``registry.errors`` instead of crashing the process.
    """
    registry = DatasetRegistry(index_dir=index_dir)
    for name, descriptor, err, tb in _iter_catalog_descriptors():
        if descriptor is None:
            module_path = f"catalog/{name}.py"
            logger.warning("skipping {}: {}", module_path, err)
            registry.errors.append({
                "module": module_path,
                "error": err or "unknown",
                "traceback": tb or "",
            })
            continue
        index_path = descriptor.default_index_path(index_dir)
        registry.entries[descriptor.id] = RegisteredDataset(
            descriptor=descriptor,
            index_path=index_path,
        )
    return registry


def register_live(
    registry: DatasetRegistry,
    descriptor: DatasetDescriptor,
) -> RegisteredDataset:
    """Register a freshly-created descriptor into a running registry.

    Used by the server's ingest endpoint to make a newly-written catalog
    module visible without restarting the process. Replaces any existing
    entry with the same id and invalidates its cached store.
    """
    index_path = descriptor.default_index_path(registry.index_dir)
    entry = RegisteredDataset(descriptor=descriptor, index_path=index_path)
    registry.entries[descriptor.id] = entry
    return entry
