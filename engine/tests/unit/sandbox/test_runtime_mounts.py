from __future__ import annotations

import platform
from pathlib import Path

import pytest

from engine.sandbox.runtime_mounts import discover_python_runtime_mounts


def test_discover_returns_runtime_paths_and_keeps_executable_unresolved() -> None:
    """Mount manifest contains the supplied python executable verbatim plus dedup'd, existing runtime paths."""
    mounts = discover_python_runtime_mounts(python_executable=Path("/usr/bin/env"))

    # Executable is kept as supplied (not resolved) so venv detection
    # downstream isn't broken.
    assert mounts.python_executable == Path("/usr/bin/env")

    assert len(mounts.runtime_paths) > 0
    seen: set[Path] = set()
    for path in mounts.runtime_paths:
        assert path.exists()
        assert path not in seen
        seen.add(path)


def test_discover_defaults_executable_to_sys_executable() -> None:
    import sys

    mounts = discover_python_runtime_mounts()
    assert mounts.python_executable == Path(sys.executable)


@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="library discovery uses /proc/self/maps which is Linux-only",
)
def test_discover_includes_loaded_shared_libraries_on_linux() -> None:
    """On Linux every Python process loads at least libc; that path must surface in ``library_paths``."""
    mounts = discover_python_runtime_mounts()
    assert len(mounts.library_paths) > 0
    for path in mounts.library_paths:
        assert path.is_file()
        assert path.suffix == ".so" or ".so." in path.name
