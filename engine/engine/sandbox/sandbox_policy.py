from __future__ import annotations

from pathlib import Path

from engine.sandbox.runtime_mounts import PythonRuntimeMounts
from engine.sandbox.sandbox_config import SandboxConfig, SandboxPolicy


def compose_policy(
    *,
    trace_path: Path,
    index_path: Path,
    runtime_mounts: PythonRuntimeMounts,
    work_dir: Path,
    sandbox_config: SandboxConfig,
) -> SandboxPolicy:
    """Compose the read-only / writable path lists for a single sandboxed run.

    ``readonly_paths`` is built positionally as
    ``[trace_path, index_path, *runtime_mounts.runtime_paths]`` — the platform
    command builders pull the trace and index by index, then mount the rest
    as the Python runtime. ``library_paths`` carries individual shared library
    files for Linux to bind at their original locations.
    """
    return SandboxPolicy(
        python_executable=runtime_mounts.python_executable,
        readonly_paths=[trace_path, index_path, *runtime_mounts.runtime_paths],
        library_paths=list(runtime_mounts.library_paths),
        writable_paths=[work_dir],
        timeout_seconds=sandbox_config.timeout_seconds,
    )
