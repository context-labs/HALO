from __future__ import annotations

from pathlib import Path

from engine.sandbox.sandbox_config import SandboxPolicy


def build_linux_bubblewrap_command(
    *,
    policy: SandboxPolicy,
    script_path: Path,
) -> list[str]:
    """Build the ``bwrap`` argv that runs user code with read-only trace/index/venv mounts and an isolated network."""
    work_dir = policy.writable_paths[0]
    venv = policy.readonly_paths[-1]
    trace_path = policy.readonly_paths[0]
    index_path = policy.readonly_paths[1]

    return [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--unshare-net",
        "--clearenv",
        "--ro-bind",
        str(trace_path),
        "/mnt/trace/traces.jsonl",
        "--ro-bind",
        str(index_path),
        "/mnt/trace/traces.jsonl.engine-index.jsonl",
        "--ro-bind",
        str(venv),
        "/venv",
        "--bind",
        str(work_dir),
        "/workspace",
        "--setenv",
        "PATH",
        "/venv/bin:/usr/bin:/bin",
        "--setenv",
        "HOME",
        "/workspace",
        "--setenv",
        "LANG",
        "C.UTF-8",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "PYTHONUNBUFFERED",
        "1",
        "--setenv",
        "TMPDIR",
        "/workspace/tmp",
        "--chdir",
        "/workspace",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--",
        "/venv/bin/python",
        str(script_path),
    ]


def render_macos_profile(*, policy: SandboxPolicy) -> str:
    """Render a ``sandbox-exec`` Scheme profile that denies by default, allows reads/writes per policy, and blocks network."""
    allows_read = "\n".join(
        f'(allow file-read* (subpath "{p}"))'
        if p.is_dir()
        else f'(allow file-read* (literal "{p}"))'
        for p in policy.readonly_paths
    )
    allows_write = "\n".join(f'(allow file-write* (subpath "{p}"))' for p in policy.writable_paths)
    return f"""(version 1)
(deny default)
(allow process*)
(deny network*)
{allows_read}
{allows_write}
"""


def build_macos_sandbox_exec_command(
    *,
    policy: SandboxPolicy,
    script_path: Path,
    profile_path: Path,
) -> list[str]:
    """Build the ``sandbox-exec`` argv pointing at a rendered profile, with ``env -i`` host-env scrub."""
    venv = policy.readonly_paths[-1]
    work_dir = policy.writable_paths[0]

    return [
        "sandbox-exec",
        "-f",
        str(profile_path),
        "env",
        "-i",
        f"PATH={venv}/bin:/usr/bin:/bin",
        f"HOME={work_dir}",
        "LANG=C.UTF-8",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUNBUFFERED=1",
        f"TMPDIR={work_dir}/tmp",
        f"{venv}/bin/python",
        str(script_path),
    ]
