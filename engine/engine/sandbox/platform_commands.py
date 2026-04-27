from __future__ import annotations

from pathlib import Path

from engine.sandbox.sandbox_config import SandboxPolicy


def build_linux_bubblewrap_command(
    *,
    bwrap_executable: Path,
    policy: SandboxPolicy,
    script_path: Path,
) -> list[str]:
    """Build the ``bwrap`` argv that runs user code under tight namespace isolation.

    Hardening:
    - ``--unshare-all`` plus an explicit ``--unshare-net`` for redundancy.
    - ``--clearenv`` then a small explicit env set; no host env leaks.
    - ``--die-with-parent`` and ``--new-session`` to bound process lifetime.
    - No ``/proc`` mount; sandboxed analysis code does not need it.
    - ``/dev`` is a fresh ``--dev`` (null/zero/random etc.) rather than the host ``/dev``.

    Mount surface is intentionally narrow:
    - The trace and index files are read-only at their host paths.
    - The Python runtime (interpreter, stdlib, site-packages) is read-only at host paths.
    - Each loaded shared library is read-only at its host path so the dynamic loader resolves it.
    - One writable workspace replaces the host home/tmp.
    """
    work_dir = policy.writable_paths[0]
    trace_path = policy.readonly_paths[0]
    index_path = policy.readonly_paths[1]
    runtime_paths = policy.readonly_paths[2:]

    argv: list[str] = [
        str(bwrap_executable),
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--unshare-net",
        "--clearenv",
        "--dev",
        "/dev",
        "--ro-bind",
        str(trace_path),
        str(trace_path),
        "--ro-bind",
        str(index_path),
        str(index_path),
    ]

    for runtime_path in runtime_paths:
        argv.extend(["--ro-bind", str(runtime_path), str(runtime_path)])

    for library_path in policy.library_paths:
        argv.extend(["--ro-bind", str(library_path), str(library_path)])

    argv.extend(
        [
            "--bind",
            str(work_dir),
            str(work_dir),
            "--setenv",
            "PATH",
            f"{policy.python_executable.parent}:/usr/bin:/bin",
            "--setenv",
            "HOME",
            str(work_dir),
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
            f"{work_dir}/tmp",
            "--chdir",
            str(work_dir),
            "--",
            str(policy.python_executable),
            str(script_path),
        ]
    )

    return argv


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
    sandbox_exec_executable: Path,
    policy: SandboxPolicy,
    script_path: Path,
    profile_path: Path,
) -> list[str]:
    """Build the ``sandbox-exec`` argv pointing at a rendered profile, with ``env -i`` host-env scrub."""
    work_dir = policy.writable_paths[0]

    return [
        str(sandbox_exec_executable),
        "-f",
        str(profile_path),
        "env",
        "-i",
        f"PATH={policy.python_executable.parent}:/usr/bin:/bin",
        f"HOME={work_dir}",
        "LANG=C.UTF-8",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUNBUFFERED=1",
        f"TMPDIR={work_dir}/tmp",
        str(policy.python_executable),
        str(script_path),
    ]
