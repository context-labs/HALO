from __future__ import annotations

import platform
import site
import sys
import sysconfig
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class PythonRuntimeMounts(BaseModel):
    """The narrow set of host paths needed to execute Python inside the sandbox.

    Computed from the running interpreter (``sys`` / ``sysconfig`` / ``site``)
    plus ``/proc/self/maps`` on Linux, so the sandbox only sees the specific
    interpreter binary, stdlib, site-packages, and shared libraries that are
    already loaded by the host process. We deliberately avoid binding broad
    system roots like ``/usr`` or ``/lib``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    python_executable: Path
    runtime_paths: tuple[Path, ...]
    library_paths: tuple[Path, ...]


def discover_python_runtime_mounts(*, python_executable: Path | None = None) -> PythonRuntimeMounts:
    """Compute the minimal mount manifest for running ``python_executable`` in a sandbox.

    On Linux, libraries currently loaded by this process (read from
    ``/proc/self/maps``) are included so the sandboxed interpreter can resolve
    its dynamic dependencies without binding broad system directories. On
    other platforms only the runtime dirs are returned; macOS uses
    ``sandbox-exec`` profiles that allow library resolution via the host.

    ``numpy`` and ``pandas`` are imported best-effort so their native shared
    objects are loaded into this process before ``/proc/self/maps`` is read.
    The bootstrap script depends on both, so ensuring their libraries are in
    the manifest avoids resolution failures inside the sandbox.
    """
    # Do not resolve symlinks here: a venv's ``bin/python`` is a symlink to
    # the base interpreter, and Python uses the symlink path to detect the
    # venv (via ``pyvenv.cfg``). Resolving would point us at the base
    # interpreter and silently skip the venv's site-packages. The venv root
    # and the resolved base install are both included via runtime_paths.
    executable = python_executable or Path(sys.executable)

    if platform.system() == "Linux":
        _force_load_bootstrap_libraries()

    runtime_paths = _collect_runtime_paths(executable=executable)
    library_paths: tuple[Path, ...]
    if platform.system() == "Linux":
        library_paths = _collect_linux_library_paths(runtime_paths=runtime_paths)
    else:
        library_paths = ()

    return PythonRuntimeMounts(
        python_executable=executable,
        runtime_paths=runtime_paths,
        library_paths=library_paths,
    )


def _force_load_bootstrap_libraries() -> None:
    """Import ``numpy`` and ``pandas`` so their native ``.so`` files are mmap'd.

    The runtime mount manifest is read from ``/proc/self/maps``; libraries
    that have not been loaded by the time we read it will not appear there
    and the sandbox will fail to import them.
    """
    try:
        import numpy  # noqa: F401
    except ImportError:
        pass
    try:
        import pandas  # noqa: F401
    except ImportError:
        pass


def _collect_runtime_paths(*, executable: Path) -> tuple[Path, ...]:
    """Collect Python install + venv prefixes and site-packages directories.

    This is the set of directories that contain Python's stdlib, the venv (if
    any), and installed third-party packages required by the bootstrap
    script (``engine``, ``numpy``, ``pandas``, ``pydantic``).
    """
    candidates: list[Path] = [executable.parent]

    for attr in ("prefix", "base_prefix", "exec_prefix", "base_exec_prefix"):
        value = getattr(sys, attr, None)
        if value:
            candidates.append(Path(value))

    for path_name in ("stdlib", "platstdlib", "purelib", "platlib"):
        try:
            value = sysconfig.get_path(path_name)
        except KeyError:
            continue
        if value:
            candidates.append(Path(value))

    candidates.extend(Path(p) for p in site.getsitepackages())
    candidates.append(Path(site.getusersitepackages()))

    for entry in sys.path:
        if entry:
            candidates.append(Path(entry))

    return _normalize_paths(candidates)


def _collect_linux_library_paths(*, runtime_paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Read ``/proc/self/maps`` for shared libraries already loaded by this process.

    Excludes paths that are already covered by ``runtime_paths`` to keep the
    bwrap argv short. Each returned path is an individual file, not a
    directory — the caller binds them one-by-one at their original locations.
    """
    maps_path = Path("/proc/self/maps")
    if not maps_path.exists():
        return ()

    runtime_set = {p.resolve() for p in runtime_paths}
    discovered: set[Path] = set()
    try:
        with maps_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split(maxsplit=5)
                if len(parts) < 6:
                    continue
                pathname = parts[5].strip()
                if not pathname.startswith("/"):
                    continue
                path = Path(pathname)
                if path.suffix not in {".so"} and ".so." not in path.name:
                    continue
                try:
                    resolved = path.resolve()
                except OSError:
                    continue
                if not resolved.is_file():
                    continue
                if any(_is_relative_to(resolved, root) for root in runtime_set):
                    continue
                discovered.add(resolved)
    except OSError:
        return ()

    return tuple(sorted(discovered))


def _normalize_paths(paths: list[Path]) -> tuple[Path, ...]:
    """De-duplicate paths by resolved canonical form, dropping ones that don't exist."""
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not resolved.exists():
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return tuple(result)


def _is_relative_to(child: Path, parent: Path) -> bool:
    """``Path.is_relative_to`` shim that treats unresolved/inaccessible paths as not-related."""
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True
