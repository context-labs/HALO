from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from engine.sandbox.log import log_unavailable

HALO_BWRAP_ENV_VAR = "HALO_BWRAP_PATH"


_LINUX_MISSING_REMEDIATION = (
    "Install bubblewrap or the bubblewrap-bin Python package, "
    "or set HALO_BWRAP_PATH to a usable bwrap binary.\n"
    "  Debian/Ubuntu: sudo apt install bubblewrap\n"
    "  Fedora/RHEL:   sudo dnf install bubblewrap\n"
    "  Alpine:        sudo apk add bubblewrap\n"
    "  Or:            pip install bubblewrap-bin"
)

_LINUX_NAMESPACE_REMEDIATION = (
    "bubblewrap is installed but the kernel/runtime denied a sandbox operation.\n"
    "  Bare Linux:    enable unprivileged user namespaces "
    "(sysctl kernel.unprivileged_userns_clone=1).\n"
    "  Ubuntu 24.04+: also relax AppArmor on user namespaces "
    "(sysctl kernel.apparmor_restrict_unprivileged_userns=0); the default policy\n"
    "                 blocks loopback configuration in --unshare-net sandboxes.\n"
    "  Docker:        run with --privileged or relax seccomp/userns restrictions.\n"
    "  Kubernetes:    use securityContext.privileged: true (or equivalent cluster policy)."
)


class LinuxClient:
    """Bubblewrap-backed sandbox client.

    Wraps a single resolved ``bwrap`` executable. Build platform-specific
    argv via :meth:`build_argv`. Resolve a working client (binary present +
    namespace probe passes) via :meth:`resolve`; on failure the resolver
    logs a remediation warning and returns ``None``.
    """

    def __init__(self, *, executable: Path) -> None:
        self.executable = executable

    @staticmethod
    def resolve() -> LinuxClient | None:
        """Find the first bwrap candidate and verify it can create namespaces.

        Resolution order: ``HALO_BWRAP_PATH`` env override → system ``bwrap``
        on ``PATH`` → packaged ``bwrap`` from the optional ``bubblewrap-bin``
        dependency. We pick the highest-priority candidate that exists and
        probe only that one — if a user explicitly set ``HALO_BWRAP_PATH``
        and it's broken, falling through to a different bwrap silently
        would hide their misconfiguration; if the namespace probe fails,
        every bwrap on this kernel will fail the same way.

        On failure, logs a warning with a remediation distinguishing
        "binary missing/broken" from "binary exists but namespace probe
        failed", then returns ``None``.
        """
        env = os.environ.get(HALO_BWRAP_ENV_VAR)
        if env:
            executable = Path(env)
            source = f"{HALO_BWRAP_ENV_VAR} override"
        elif (system := shutil.which("bwrap")) is not None:
            executable = Path(system)
            source = "system PATH"
        elif (packaged := _packaged_bwrap()) is not None:
            executable = packaged
            source = "bubblewrap-bin package"
        else:
            log_unavailable(
                diagnostic="bubblewrap not found via env, PATH, or bubblewrap-bin package",
                remediation=_LINUX_MISSING_REMEDIATION,
            )
            return None

        if not executable.exists():
            log_unavailable(
                diagnostic=f"{source} does not exist: {executable}",
                remediation=_LINUX_MISSING_REMEDIATION,
            )
            return None

        failure = _probe(executable)
        if failure is not None:
            log_unavailable(
                diagnostic=f"{source} ({executable}) probe failed: {failure}",
                remediation=_LINUX_NAMESPACE_REMEDIATION,
            )
            return None

        return LinuxClient(executable=executable)

    def build_argv(
        self,
        *,
        python_executable: Path,
        script_path: Path,
        work_dir: Path,
        readonly_paths: list[Path],
        library_paths: list[Path],
    ) -> list[str]:
        """Build the ``bwrap`` argv that runs ``script_path`` under namespace isolation.

        Hardening:
          - ``--unshare-all`` and explicit ``--unshare-net`` for redundancy.
          - ``--clearenv`` then a small explicit env set; no host env leaks.
          - ``--die-with-parent`` and ``--new-session`` to bound process lifetime.
          - ``--dev /dev`` mounts a fresh tmpfs containing only
            ``null``/``zero``/``urandom``/``random``/``tty``/``ptmx``. It does
            NOT bind the host ``/dev``. Required because Python reads
            ``/dev/urandom`` for hash randomization at startup and numpy
            seeds its default RNG from it; without this ``np.random.*`` and
            ``secrets`` fail.
          - No ``/proc`` mount; sandboxed analysis code does not need it.

        Mount surface:
          - Each ``readonly_paths`` entry is bound read-only at its host path.
          - Each ``library_paths`` entry is bound read-only at its host path
            so the dynamic loader resolves it without binding broad system
            directories.
          - ``work_dir`` is the writable workspace, replacing ``$HOME``/
            ``$TMPDIR`` for the sandboxed process.
        """
        argv: list[str] = [
            str(self.executable),
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--unshare-net",
            "--clearenv",
            "--dev",
            "/dev",
        ]

        for path in readonly_paths:
            argv.extend(["--ro-bind", str(path), str(path)])
        for path in library_paths:
            argv.extend(["--ro-bind", str(path), str(path)])

        argv.extend(
            [
                "--bind",
                str(work_dir),
                str(work_dir),
                "--setenv",
                "PATH",
                f"{python_executable.parent}:/usr/bin:/bin",
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
                str(work_dir / "tmp"),
                "--chdir",
                str(work_dir),
                "--",
                str(python_executable),
                str(script_path),
            ]
        )
        return argv


def _packaged_bwrap() -> Path | None:
    """Return the ``bwrap`` path provided by the optional ``bubblewrap-bin`` dep, if installed."""
    try:
        from bubblewrap_bin import bwrap_path  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        path = bwrap_path()
    except Exception:
        return None
    return Path(path) if path else None


def _probe(executable: Path) -> str | None:
    """Run ``bwrap`` with hardening flags and verify the sandbox is fully usable.

    bwrap writes a JSON status object containing ``"child-pid"`` to the FD
    passed via ``--info-fd`` *after* the user/mount namespaces are created
    but *before* it sets up loopback (for ``--unshare-net``) and execs the
    user command. We deliberately exec a path that does not exist so the
    in-sandbox ``exec(2)`` fails with ENOENT — execvp's ENOENT path makes
    bwrap exit 127. That gives us two independent success signals to check:

      1. ``"child-pid"`` written to ``--info-fd`` (namespaces created), and
      2. exit code 127 (everything between info-fd write and exec succeeded).

    Both are required. If a kernel/runtime restriction blocks loopback
    configuration (Ubuntu 24.04 AppArmor restricts unprivileged user
    namespaces from doing this), bwrap dies *between* (1) and (2) with a
    different exit code, which we treat as probe failure even though
    ``"child-pid"`` is present.

    Returns ``None`` on success, or a diagnostic string on failure.
    """
    read_fd, write_fd = os.pipe()
    write_fd_open = True
    proc: subprocess.Popen[bytes] | None = None

    argv = [
        str(executable),
        "--unshare-all",
        "--unshare-net",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--info-fd",
        str(write_fd),
        "--",
        "/halo-sandbox-probe-no-such-exec-target",
    ]

    try:
        try:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                pass_fds=(write_fd,),
            )
        except OSError as exc:
            return f"failed to spawn bwrap: {type(exc).__name__}: {exc}"

        # Parent never writes to ``--info-fd``; closing here ensures the
        # read end gets EOF as soon as bwrap exits without writing.
        os.close(write_fd)
        write_fd_open = False

        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return "bwrap probe timed out after 5.0s"

        try:
            info = os.read(read_fd, 65_536)
        except OSError:
            info = b""

        if b'"child-pid"' in info and proc.returncode == 127:
            return None

        stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
        diagnostic = stderr_bytes.decode("utf-8", errors="replace").strip()
        return diagnostic or f"bwrap exited {proc.returncode}"
    finally:
        os.close(read_fd)
        if write_fd_open:
            os.close(write_fd)
