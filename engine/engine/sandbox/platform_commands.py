from __future__ import annotations

from pathlib import Path

from engine.sandbox.sandbox_config import SandboxPolicy
from engine.sandbox.sandbox_paths import SANDBOX_DEV_PATH, SANDBOX_TMP_DIRNAME


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
    - ``--dev /dev`` mounts a fresh tmpfs containing ``null``/``zero``/
      ``urandom``/``random``/``tty``/``ptmx`` only. It does NOT bind the
      host ``/dev``. We need it because Python reads ``/dev/urandom`` at
      startup for hash randomization and numpy/pandas read it for default
      RNG seeding; without it ``np.random.*`` and ``secrets`` fail.

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
        str(SANDBOX_DEV_PATH),
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
            str(work_dir / SANDBOX_TMP_DIRNAME),
            "--chdir",
            str(work_dir),
            "--",
            str(policy.python_executable),
            str(script_path),
        ]
    )

    return argv


def render_macos_profile(*, policy: SandboxPolicy) -> str:
    """Render a ``sandbox-exec`` Scheme profile for the run.

    Security model on macOS:
      - The default action is ``deny``, but ``file-read*`` is broadly allowed
        because the macOS dynamic loader and CoreFoundation read from many
        opaque locations (``/usr/lib``, ``/System``, ``/private/var/db``,
        framework-specific paths) that are impractical to enumerate and are
        already SIP-protected on the host. Narrowing reads here would mean
        Python could not even start.
      - ``file-write*`` is denied by default and re-allowed only on the
        explicit ``writable_paths``. This is the actual filesystem boundary —
        sandboxed code cannot mutate anything outside the workspace.
      - Network is denied entirely.
      - ``mach-lookup``/``ipc-posix-shm``/``sysctl-read``/``signal`` are
        allowed because Python and CoreFoundation rely on them for normal
        startup; without these, the interpreter aborts before reaching
        user code.

    Reads of ``$HOME`` (typical user-data location) are denied even though
    ``file-read*`` is otherwise broad. The trace and index files are
    re-allowed by literal path so they remain accessible regardless of where
    they live.
    """
    home_dir = _macos_home_dir()
    re_allows_read = "\n".join(
        f'(allow file-read* (subpath "{p}"))'
        if p.is_dir()
        else f'(allow file-read* (literal "{p}"))'
        for p in policy.readonly_paths
    )
    allows_write = "\n".join(f'(allow file-write* (subpath "{p}"))' for p in policy.writable_paths)

    home_clause = f'(deny file-read* (subpath "{home_dir}"))' if home_dir else ""
    return f"""(version 1)
(deny default)
(allow process*)
(allow mach-lookup)
(allow ipc-posix-shm)
(allow sysctl-read)
(allow signal)
(allow file-read*)
{home_clause}
{re_allows_read}
{allows_write}
(deny network*)
"""


def _macos_home_dir() -> str | None:
    """Resolve the current user's home directory; ``None`` if unavailable.

    Returns the absolute path so the rendered profile can deny reads under
    ``$HOME`` while still re-allowing the explicit policy paths.
    """
    try:
        from pathlib import Path

        home = Path.home()
    except (RuntimeError, KeyError):
        return None
    return str(home) if home else None


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
        f"TMPDIR={work_dir / SANDBOX_TMP_DIRNAME}",
        str(policy.python_executable),
        str(script_path),
    ]
