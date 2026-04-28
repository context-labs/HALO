from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
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
    "bubblewrap is installed but the kernel/runtime denied namespace creation.\n"
    "  Bare Linux:  enable unprivileged user namespaces "
    "(sysctl kernel.unprivileged_userns_clone=1).\n"
    "  Docker:      run with --privileged or relax seccomp/userns restrictions.\n"
    "  Kubernetes:  use securityContext.privileged: true (or equivalent cluster policy)."
)


@dataclass(frozen=True)
class _Candidate:
    """One bwrap candidate found during resolution: which path, where it came from."""

    executable: Path
    source: str


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
        """Find a usable ``bwrap`` and verify it can create namespaces.

        Resolution order:
          1. ``HALO_BWRAP_PATH`` env override.
          2. System ``bwrap`` on ``PATH``.
          3. Packaged ``bwrap`` from the optional ``bubblewrap-bin`` dependency.

        On failure, logs a warning whose remediation distinguishes "binary
        is missing entirely" from "binary exists but namespace probe failed",
        then returns ``None``. Callers treat ``None`` as "sandbox unavailable
        for this run" — there's no exception to catch.
        """
        candidates = _candidates()
        if not candidates:
            log_unavailable(
                diagnostic=(
                    "bubblewrap (bwrap) not found via env, PATH, or bubblewrap-bin package"
                ),
                remediation=_LINUX_MISSING_REMEDIATION,
            )
            return None

        missing: list[str] = []
        probe_failures: list[str] = []
        for candidate in candidates:
            if not candidate.executable.exists():
                missing.append(
                    f"bwrap candidate from {candidate.source} does not exist: "
                    f"{candidate.executable}"
                )
                continue
            failure = _probe(candidate.executable)
            if failure is None:
                return LinuxClient(executable=candidate.executable)
            probe_failures.append(
                f"bwrap from {candidate.source} ({candidate.executable}) "
                f"failed namespace probe: {failure}"
            )

        if probe_failures:
            log_unavailable(
                diagnostic="\n".join(probe_failures),
                remediation=_LINUX_NAMESPACE_REMEDIATION,
            )
        else:
            log_unavailable(
                diagnostic="\n".join(missing),
                remediation=_LINUX_MISSING_REMEDIATION,
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


def _candidates() -> list[_Candidate]:
    """Build the ordered list of bwrap candidates from env, PATH, and ``bubblewrap-bin``."""
    out: list[_Candidate] = []
    env_path = os.environ.get(HALO_BWRAP_ENV_VAR)
    if env_path:
        out.append(_Candidate(executable=Path(env_path), source=f"{HALO_BWRAP_ENV_VAR} override"))
    system_bwrap = shutil.which("bwrap")
    if system_bwrap is not None:
        out.append(_Candidate(executable=Path(system_bwrap), source="system PATH"))
    packaged = _packaged_bwrap()
    if packaged is not None:
        out.append(_Candidate(executable=packaged, source="bubblewrap-bin package"))
    return out


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
    """Run ``bwrap`` with hardening flags and detect namespace setup via ``--info-fd``.

    bwrap writes a JSON status object containing ``"child-pid"`` to the FD
    passed via ``--info-fd`` *after* the sandbox is fully set up and *just
    before* it execs the user command. We exec a deliberately nonexistent
    path so the kernel's ``exec(2)`` inside the sandbox fails with ENOENT —
    by then bwrap has already emitted the status object, so we know setup
    worked.

    Net effect: the probe runs with an entirely empty mount namespace and
    distinguishes real namespace failures (``"child-pid"`` never appears)
    from the expected sandbox-set-up-fine-but-exec-failed-as-expected case.

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

        if b'"child-pid"' in info:
            return None

        stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
        diagnostic = stderr_bytes.decode("utf-8", errors="replace").strip()
        return diagnostic or f"bwrap exited {proc.returncode} without writing --info-fd output"
    finally:
        os.close(read_fd)
        if write_fd_open:
            os.close(write_fd)
