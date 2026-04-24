from __future__ import annotations

from pathlib import Path

from engine.sandbox.sandbox_config import SandboxConfig, SandboxPolicy


def compose_policy(
    *,
    trace_path: Path,
    index_path: Path,
    sandbox_venv: Path,
    work_dir: Path,
    sandbox_config: SandboxConfig,
) -> SandboxPolicy:
    return SandboxPolicy(
        readonly_paths=[trace_path, index_path, sandbox_venv],
        writable_paths=[work_dir],
        timeout_seconds=sandbox_config.timeout_seconds,
    )
