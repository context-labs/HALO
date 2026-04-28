from __future__ import annotations

import platform
import site
import sys
from pathlib import Path

from engine.sandbox.models import PythonRuntimeMounts


def discover_python_runtime_mounts(*, python_executable: Path | None = None) -> PythonRuntimeMounts:
    """Compute the minimal mount manifest for Python in a sandbox.

    Imports ``numpy`` and ``pandas`` eagerly so ``/proc/self/maps`` reflects
    their ``.so`` files when we read it on Linux. Both are pinned engine
    dependencies; if either import fails the engine cannot run user code at
    all and we let the error surface here instead of silently producing an
    incomplete mount manifest.

    A venv's ``bin/python`` is a symlink to the base interpreter; we keep
    the symlink path (not the resolved target) because Python uses it to
    locate ``pyvenv.cfg`` and pick up the venv's site-packages.
    """
    import numpy  # noqa: F401  pyright: ignore[reportUnusedImport]
    import pandas  # noqa: F401  pyright: ignore[reportUnusedImport]

    executable = python_executable or Path(sys.executable)
    runtime_paths = _collect_runtime_paths()
    library_paths = (
        _collect_loaded_shared_libraries(runtime_roots=runtime_paths)
        if platform.system() == "Linux"
        else ()
    )
    return PythonRuntimeMounts(
        python_executable=executable,
        runtime_paths=runtime_paths,
        library_paths=library_paths,
    )


def _collect_runtime_paths() -> tuple[Path, ...]:
    """Union of Python install + venv + site-packages + ``sys.path`` roots, deduped.

    ``sys.path`` is included because editable installs (``pip install -e .``)
    register the package's source directory there via a ``.pth`` file rather
    than copying it into ``site-packages``. Without ``sys.path`` the
    sandboxed bootstrap script can't import the engine package itself.
    """
    candidates = [
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path(sys.exec_prefix),
        Path(sys.base_exec_prefix),
        *(Path(p) for p in site.getsitepackages()),
        Path(site.getusersitepackages()),
        *(Path(p) for p in sys.path if p),
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved.exists() and resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return tuple(out)


def _collect_loaded_shared_libraries(*, runtime_roots: tuple[Path, ...]) -> tuple[Path, ...]:
    """Read ``/proc/self/maps`` and return loaded ``.so`` files outside ``runtime_roots``.

    Each line of ``/proc/self/maps`` has the form::

        ADDRESS PERMS OFFSET DEV INODE PATHNAME

    ``PATHNAME`` may contain spaces, so we split with ``maxsplit=5`` to keep
    it intact. Pseudo-entries like ``[heap]``, ``[stack]``, ``[vdso]``, and
    ``(deleted)`` filenames are filtered by the leading-slash + ``.so``
    checks. Paths that already live under one of the runtime roots are
    skipped so the bwrap argv stays short.
    """
    discovered: set[Path] = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        pathname = fields[5].strip()
        if not pathname.startswith("/"):
            continue
        if not (pathname.endswith(".so") or ".so." in pathname):
            continue
        try:
            resolved = Path(pathname).resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        if any(resolved.is_relative_to(root) for root in runtime_roots):
            continue
        discovered.add(resolved)
    return tuple(sorted(discovered))
