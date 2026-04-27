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

# Probe argv mirrors the hardening flags used by the production Linux command
# in ``platform_commands.build_linux_bubblewrap_command``. If the production
# command changes, update this too — the probe must reject hosts where the
# real command would fail.
_LINUX_PROBE_FLAGS: tuple[str, ...] = (
    "--unshare-all",
    "--unshare-net",
    "--die-with-parent",
    "--new-session",
    "--clearenv",
    "--dev",
    "/dev",
)


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

    last_diagnostic = ""
    for executable, backend, source in candidates:
        if not executable.exists():
            last_diagnostic = f"bwrap candidate from {source} does not exist: {executable}"
            continue
        ok, diagnostic = _probe_bwrap(executable)
        if ok:
            return SandboxStatus.ok(backend=backend, executable=executable)
        last_diagnostic = f"bwrap from {source} ({executable}) failed namespace probe: {diagnostic}"

    return SandboxStatus.unavailable(
        reason=SandboxUnavailableReason.NAMESPACE_DENIED,
        diagnostic=last_diagnostic,
        remediation=_LINUX_NAMESPACE_REMEDIATION,
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
    """Run ``/bin/true`` under ``bwrap`` with production flags; return (ok, diagnostic)."""
    argv = [str(executable), *_LINUX_PROBE_FLAGS, "--", "/bin/true"]
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if result.returncode == 0:
        return True, ""
    stderr = result.stderr.decode("utf-8", errors="replace").strip()
    return False, stderr or f"exit code {result.returncode}"


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
