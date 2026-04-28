from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from engine.sandbox.log import log_unavailable

_LINUX_MISSING_REMEDIATION = (
    "Install bubblewrap from your distro's package manager:\n"
    "  Debian/Ubuntu: sudo apt install bubblewrap\n"
    "  Fedora/RHEL:   sudo dnf install bubblewrap\n"
    "  Alpine:        sudo apk add bubblewrap\n"
    "Then re-run the engine."
)

_LINUX_NAMESPACE_REMEDIATION = (
    "bubblewrap is on the host but the kernel/runtime denied a sandbox operation.\n"
    "Easiest fix: install bubblewrap from your distro's package manager — the\n"
    "system package ships with the right policy hooks (e.g. AppArmor profile\n"
    "on Ubuntu 24.04+) that the bundled binary doesn't have:\n"
    "  Debian/Ubuntu: sudo apt install bubblewrap\n"
    "  Fedora/RHEL:   sudo dnf install bubblewrap\n"
    "If you're already using the system bwrap and still hitting this, the\n"
    "host or container is restricting user namespaces:\n"
    "  Docker:     run with --privileged or relax seccomp/userns restrictions.\n"
    "  Kubernetes: use securityContext.privileged: true."
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
        """Find a working ``bwrap`` and verify it can create namespaces.

        Resolution order: system ``bwrap`` on ``PATH`` (which is
        ``/usr/bin/bwrap`` on distros that ship the package) → packaged
        ``bwrap`` from the optional ``bubblewrap-bin`` dependency. If the
        system probe fails (e.g. AppArmor on Ubuntu 24.04+ blocking an
        unprivileged user namespace), we fall through to the packaged copy
        rather than give up. On failure, logs a warning and returns ``None``.
        """
        failures: list[str] = []

        system = shutil.which("bwrap")
        if system is not None:
            executable = Path(system)
            failure = _probe(executable)
            if failure is None:
                return LinuxClient(executable=executable)
            failures.append(f"system PATH ({executable}) probe failed: {failure}")

        packaged = _packaged_bwrap()
        if packaged is not None:
            failure = _probe(packaged)
            if failure is None:
                return LinuxClient(executable=packaged)
            failures.append(f"bubblewrap-bin package ({packaged}) probe failed: {failure}")

        if not failures:
            log_unavailable(
                diagnostic="bubblewrap not found via PATH or bubblewrap-bin package",
                remediation=_LINUX_MISSING_REMEDIATION,
            )
        else:
            log_unavailable(
                diagnostic="no working bubblewrap candidate:\n  " + "\n  ".join(failures),
                remediation=_LINUX_NAMESPACE_REMEDIATION,
            )
        return None

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
          - Explicit user, PID, and network namespaces. This matches Codex's
            current bwrap shape more closely than ``--unshare-all`` and avoids
            asking the host for cgroup/IPC/UTS namespaces this sandbox does not
            need.
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
            "--unshare-user",
            "--unshare-pid",
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
    passed via ``--info-fd`` after the core namespaces are created. To know
    that network namespace setup succeeded too, we rely on the
    in-sandbox ``execvp`` reaching the kernel and writing the ENOENT
    diagnostic ``"bwrap: execvp <path>: No such file or directory"`` to
    stderr — bwrap only reaches that point if every preceding setup step
    (namespaces, mounts, loopback) succeeded.

    Probe success requires both signals:
      1. ``"child-pid"`` in ``--info-fd`` output (namespaces created), and
      2. ``"bwrap: execvp"`` in stderr (setup completed all the way to exec).

    bwrap exits with status 1 in both the exec-failure and the
    namespace-internal-failure paths (``die_with_error``), so we cannot
    rely on the exit code to discriminate; the stderr marker is the
    reliable signal.

    Returns ``None`` on success, or a diagnostic string on failure.
    """
    if not executable.is_file():
        return "missing executable"

    read_fd, write_fd = os.pipe()
    write_fd_open = True
    proc: subprocess.Popen[bytes] | None = None

    argv = [
        str(executable),
        "--unshare-user",
        "--unshare-pid",
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

        stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""

        if b'"child-pid"' in info and b"bwrap: execvp" in stderr_bytes:
            return None

        diagnostic = stderr_bytes.decode("utf-8", errors="replace").strip()
        return diagnostic or f"bwrap exited {proc.returncode}"
    finally:
        os.close(read_fd)
        if write_fd_open:
            os.close(write_fd)
