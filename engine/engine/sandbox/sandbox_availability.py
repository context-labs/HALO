from __future__ import annotations

import os
import platform
import shutil
import subprocess
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class SandboxBackend(str, Enum):
    """Concrete sandbox backend selected for a run.

    macOS uses Apple's ``sandbox-exec``. Linux uses ``bubblewrap``, sourced
    from either the host or the optional ``bubblewrap-bin`` package.
    """

    MACOS_SANDBOX_EXEC = "macos-sandbox-exec"
    LINUX_BWRAP_SYSTEM = "linux-bwrap-system"
    LINUX_BWRAP_PACKAGED = "linux-bwrap-packaged"


class SandboxUnavailableReason(str, Enum):
    """Why ``run_code`` cannot run; drives the user-facing remediation text."""

    UNSUPPORTED_PLATFORM = "unsupported-platform"
    MISSING_BACKEND = "missing-backend"
    NAMESPACE_DENIED = "namespace-denied"
    PROBE_FAILED = "probe-failed"


class SandboxStatus(BaseModel):
    """Result of probing the host for a working sandbox backend.

    A status either resolves to a usable backend with an executable, or to a
    typed unavailability reason with diagnostic + remediation text. Engine
    startup uses this to decide whether to register the ``run_code`` tool and
    what to print in the unavailability warning.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    available: bool
    backend: SandboxBackend | None
    executable: Path | None
    reason: SandboxUnavailableReason | None
    diagnostic: str
    remediation: str

    @classmethod
    def ok(cls, *, backend: SandboxBackend, executable: Path) -> "SandboxStatus":
        """Build a status for a successfully-probed backend."""
        return cls(
            available=True,
            backend=backend,
            executable=executable,
            reason=None,
            diagnostic="",
            remediation="",
        )

    @classmethod
    def unavailable(
        cls,
        *,
        reason: SandboxUnavailableReason,
        diagnostic: str,
        remediation: str,
    ) -> "SandboxStatus":
        """Build a status describing why no backend resolved."""
        return cls(
            available=False,
            backend=None,
            executable=None,
            reason=reason,
            diagnostic=diagnostic,
            remediation=remediation,
        )


HALO_BWRAP_ENV_VAR = "HALO_BWRAP_PATH"

# Path the probe asks bwrap to exec inside the sandbox. We deliberately
# choose a path that does not exist on any host so the kernel's ``exec(2)``
# inside the sandbox fails with ENOENT — that is fine because by the time
# the kernel reaches ``exec(2)`` the bwrap parent has already finished
# namespace setup and emitted a status object on ``--info-fd``. Choosing a
# nonexistent target lets the probe avoid binding any host filesystem.
_PROBE_EXEC_PATH = "/halo-sandbox-probe-no-such-exec-target"

# Marker we look for in the ``--info-fd`` JSON to confirm bwrap got past
# namespace setup. Stable across all supported bwrap versions.
_PROBE_INFO_FD_MARKER = b'"child-pid"'

# Wall-clock timeout for the probe subprocess. The probe does no I/O and
# exits as soon as the sandboxed exec fails with ENOENT, so a few seconds
# is far more than enough.
_PROBE_TIMEOUT_SECONDS = 5.0


def resolve_sandbox_status() -> SandboxStatus:
    """Resolve the sandbox backend for the current host.

    Returns a status with ``available=True`` if a backend was found and probed
    successfully, otherwise an unavailable status carrying the typed reason
    and human-readable remediation. Never raises.
    """
    system = platform.system()
    if system == "Darwin":
        return _resolve_macos()
    if system == "Linux":
        return _resolve_linux()
    return SandboxStatus.unavailable(
        reason=SandboxUnavailableReason.UNSUPPORTED_PLATFORM,
        diagnostic=f"unsupported platform: {system}",
        remediation="run_code requires Linux (bubblewrap) or macOS (sandbox-exec).",
    )


def _resolve_macos() -> SandboxStatus:
    """Locate ``sandbox-exec`` on PATH; macOS ships it by default."""
    found = shutil.which("sandbox-exec")
    if found is None:
        return SandboxStatus.unavailable(
            reason=SandboxUnavailableReason.MISSING_BACKEND,
            diagnostic="sandbox-exec not found on PATH",
            remediation=("sandbox-exec ships with macOS by default. Ensure /usr/bin is on PATH."),
        )
    return SandboxStatus.ok(backend=SandboxBackend.MACOS_SANDBOX_EXEC, executable=Path(found))


def _resolve_linux() -> SandboxStatus:
    """Find a usable ``bwrap`` and verify it can create namespaces.

    Resolution order:
    1. ``HALO_BWRAP_PATH`` environment override.
    2. System ``bwrap`` on ``PATH``.
    3. Packaged ``bwrap`` from the optional ``bubblewrap-bin`` dependency.

    Failure attribution:
    - If no candidates are found, or every candidate path is missing on disk,
      report ``MISSING_BACKEND`` with install instructions.
    - If at least one candidate exists but its namespace probe failed, report
      ``NAMESPACE_DENIED`` with the kernel/runtime remediation. Reaching the
      probe means the binary is real; the failure is a host policy issue.
    """
    candidates: list[tuple[Path, SandboxBackend, str]] = []

    env_path = os.environ.get(HALO_BWRAP_ENV_VAR)
    if env_path:
        candidates.append(
            (Path(env_path), SandboxBackend.LINUX_BWRAP_SYSTEM, f"{HALO_BWRAP_ENV_VAR} override")
        )

    system_bwrap = shutil.which("bwrap")
    if system_bwrap is not None:
        candidates.append((Path(system_bwrap), SandboxBackend.LINUX_BWRAP_SYSTEM, "system PATH"))

    packaged = _packaged_bwrap()
    if packaged is not None:
        candidates.append((packaged, SandboxBackend.LINUX_BWRAP_PACKAGED, "bubblewrap-bin package"))

    if not candidates:
        return SandboxStatus.unavailable(
            reason=SandboxUnavailableReason.MISSING_BACKEND,
            diagnostic="bubblewrap (bwrap) not found via env, PATH, or bubblewrap-bin package",
            remediation=_LINUX_MISSING_REMEDIATION,
        )

    missing_diagnostics: list[str] = []
    probe_diagnostics: list[str] = []
    for executable, backend, source in candidates:
        if not executable.exists():
            missing_diagnostics.append(
                f"bwrap candidate from {source} does not exist: {executable}"
            )
            continue
        ok, diagnostic = _probe_bwrap(executable)
        if ok:
            return SandboxStatus.ok(backend=backend, executable=executable)
        probe_diagnostics.append(
            f"bwrap from {source} ({executable}) failed namespace probe: {diagnostic}"
        )

    if probe_diagnostics:
        return SandboxStatus.unavailable(
            reason=SandboxUnavailableReason.NAMESPACE_DENIED,
            diagnostic="\n".join(probe_diagnostics),
            remediation=_LINUX_NAMESPACE_REMEDIATION,
        )
    return SandboxStatus.unavailable(
        reason=SandboxUnavailableReason.MISSING_BACKEND,
        diagnostic="\n".join(missing_diagnostics),
        remediation=_LINUX_MISSING_REMEDIATION,
    )


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


def _probe_bwrap(executable: Path) -> tuple[bool, str]:
    """Run ``bwrap`` with hardening flags and detect namespace setup via ``--info-fd``.

    bwrap writes a JSON status object containing ``"child-pid"`` to the FD
    passed via ``--info-fd`` *after* the sandbox is fully set up and *just
    before* it execs the user command. We read that pipe to confirm
    namespace setup; the in-sandbox exec target is a deliberately
    nonexistent path so the kernel's ``exec(2)`` inside the sandbox fails
    immediately with ENOENT — but by then bwrap has already emitted the
    status object, so we know setup worked.

    Net effect: the probe runs with an entirely empty mount namespace
    (no ``/usr``, no ``/bin``, no host filesystem at all) and still
    distinguishes real namespace failures (``"child-pid"`` never appears)
    from sandbox-set-up-fine-but-exec-failed-as-expected.
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
        _PROBE_EXEC_PATH,
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
            return False, f"failed to spawn bwrap: {type(exc).__name__}: {exc}"

        # Parent never writes to ``--info-fd``; closing here ensures the
        # read end gets EOF as soon as bwrap exits without writing.
        os.close(write_fd)
        write_fd_open = False

        try:
            proc.wait(timeout=_PROBE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return False, f"bwrap probe timed out after {_PROBE_TIMEOUT_SECONDS}s"

        try:
            info = os.read(read_fd, 65_536)
        except OSError:
            info = b""

        if _PROBE_INFO_FD_MARKER in info:
            return True, ""

        stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
        diagnostic = stderr_bytes.decode("utf-8", errors="replace").strip()
        return (
            False,
            diagnostic or f"bwrap exited {proc.returncode} without writing --info-fd output",
        )
    finally:
        os.close(read_fd)
        if write_fd_open:
            os.close(write_fd)


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


def render_unavailable_warning(status: SandboxStatus) -> str:
    """Render a multi-line warning describing why ``run_code`` is disabled and how to fix it."""
    if status.available:
        raise ValueError("render_unavailable_warning called on an available SandboxStatus")
    return (
        "HALO run_code disabled: sandbox unavailable.\n"
        "\n"
        f"Reason ({status.reason.value if status.reason else 'unknown'}):\n"
        f"  {status.diagnostic}\n"
        "\n"
        "How to fix:\n"
        f"{_indent(status.remediation, '  ')}\n"
        "\n"
        "The engine will continue without exposing run_code to the agent."
    )


def _indent(text: str, prefix: str) -> str:
    """Prefix each line of ``text`` with ``prefix``."""
    return "\n".join(f"{prefix}{line}" if line else line for line in text.splitlines())
