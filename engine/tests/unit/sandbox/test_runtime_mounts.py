from __future__ import annotations

import platform
from pathlib import Path

import pytest

from engine.sandbox import runtime_mounts
from engine.sandbox.runtime_mounts import discover_python_runtime_mounts


def test_discover_returns_existing_runtime_paths(tmp_path: Path) -> None:
    """The runtime mount manifest contains the resolved python executable and at least one runtime path."""
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.write_text("")

    mounts = discover_python_runtime_mounts(python_executable=python)

    assert mounts.python_executable == python.resolve()
    # Even with a synthetic python executable, sys.prefix etc. resolve from the
    # running interpreter and yield real existing paths.
    assert len(mounts.runtime_paths) > 0
    for path in mounts.runtime_paths:
        assert path.exists()


def test_discover_de_duplicates_runtime_paths(tmp_path: Path) -> None:
    mounts = discover_python_runtime_mounts(python_executable=Path("/usr/bin/env"))
    seen: set[Path] = set()
    for path in mounts.runtime_paths:
        assert path not in seen
        seen.add(path)


@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="library discovery uses /proc/self/maps which is Linux-only",
)
def test_discover_includes_loaded_shared_libraries() -> None:
    """On Linux, ``/proc/self/maps`` produces at least one library path outside the runtime tree."""
    mounts = discover_python_runtime_mounts()
    # We can't make hard assertions about specific lib names because the loader
    # path varies by distro, but every modern Python loads libc, so at least
    # one library path must surface.
    assert len(mounts.library_paths) > 0
    for path in mounts.library_paths:
        assert path.is_file()
        assert path.suffix == ".so" or ".so." in path.name


def test_force_load_bootstrap_libraries_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing optional deps should not break discovery."""
    real_import = __import__

    def _raising_import(name: str, *args, **kwargs):
        if name in ("numpy", "pandas"):
            raise ImportError(f"simulated missing {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _raising_import)
    runtime_mounts._force_load_bootstrap_libraries()
