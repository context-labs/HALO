from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from engine.sandbox.runtime_mounts import PythonRuntimeMounts, discover_python_runtime_mounts


class SandboxBackend(str, Enum):
    """Concrete sandbox backend selected for a run.

    macOS uses Apple's ``sandbox-exec``. Linux uses ``bubblewrap``, sourced
    from either the host or the optional ``bubblewrap-bin`` package.
    """

    MACOS_SANDBOX_EXEC = "macos-sandbox-exec"
    LINUX_BWRAP_SYSTEM = "linux-bwrap-system"
    LINUX_BWRAP_PACKAGED = "linux-bwrap-packaged"


@dataclass(frozen=True)
class SandboxRuntime:
    """A fully-resolved sandbox: backend, executable, and the Python runtime mount manifest.

    The presence of a ``SandboxRuntime`` is the single signal that ``run_code``
    is usable for a run. Engine code stores ``SandboxRuntime | None`` and
    treats ``None`` as "sandbox unavailable" — there is no half-initialized
    state where the backend is known but mounts are missing.
    """

    backend: SandboxBackend
    executable: Path
    runtime_mounts: PythonRuntimeMounts


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


_logger = logging.getLogger(__name__)


def resolve_sandbox_runtime(*, python_executable: Path | None = None) -> SandboxRuntime | None:
    """Probe the host once; return a ready ``SandboxRuntime`` or ``None``.

    On unavailability, emits a multi-line warning describing the cause and
    remediation through both ``logging`` and ``stderr`` so it is visible in
    every common deployment (CLI, library import, container logs).

    ``python_executable`` overrides the interpreter whose runtime is mounted
    into the sandbox; ``None`` uses the active ``sys.executable``.
    """
    runtime = _resolve(python_executable=python_executable)
    if isinstance(runtime, SandboxRuntime):
        return runtime
    _emit_unavailable_warning(runtime)
    return None


def _resolve(*, python_executable: Path | None) -> SandboxRuntime | _SandboxUnavailable:
    system = platform.system()
    if system == "Darwin":
        return _resolve_macos(python_executable=python_executable)
    if system == "Linux":
        return _resolve_linux(python_executable=python_executable)
    return _SandboxUnavailable(
        reason=_Reason.UNSUPPORTED_PLATFORM,
        diagnostic=f"unsupported platform: {system}",
        remediation="run_code requires Linux (bubblewrap) or macOS (sandbox-exec).",
    )


def _resolve_macos(*, python_executable: Path | None) -> SandboxRuntime | _SandboxUnavailable:
    """Locate ``sandbox-exec`` on PATH; macOS ships it by default."""
    found = shutil.which("sandbox-exec")
    if found is None:
        return _SandboxUnavailable(
            reason=_Reason.MISSING_BACKEND,
            diagnostic="sandbox-exec not found on PATH",
            remediation="sandbox-exec ships with macOS by default. Ensure /usr/bin is on PATH.",
        )
    return SandboxRuntime(
        backend=SandboxBackend.MACOS_SANDBOX_EXEC,
        executable=Path(found),
        runtime_mounts=discover_python_runtime_mounts(python_executable=python_executable),
    )


def _resolve_linux(*, python_executable: Path | None) -> SandboxRuntime | _SandboxUnavailable:
    """Find a usable ``bwrap`` and verify it can create namespaces.

    Resolution order:
    1. ``HALO_BWRAP_PATH`` environment override.
    2. System ``bwrap`` on ``PATH``.
    3. Packaged ``bwrap`` from the optional ``bubblewrap-bin`` dependency.

    Failure attribution:
    - If no candidates are found, or every candidate path is missing on disk,
      report ``MISSING_BACKEND`` with install instructions.
    - If at least one candidate exists but its namespace probe failed, report
      ``NAMESPACE_DENIED`` with the kernel/runtime remediation.
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
        return _SandboxUnavailable(
            reason=_Reason.MISSING_BACKEND,
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
            return SandboxRuntime(
                backend=backend,
                executable=executable,
                runtime_mounts=discover_python_runtime_mounts(python_executable=python_executable),
            )
        probe_diagnostics.append(
            f"bwrap from {source} ({executable}) failed namespace probe: {diagnostic}"
        )

    if probe_diagnostics:
        return _SandboxUnavailable(
            reason=_Reason.NAMESPACE_DENIED,
            diagnostic="\n".join(probe_diagnostics),
            remediation=_LINUX_NAMESPACE_REMEDIATION,
        )
    return _SandboxUnavailable(
        reason=_Reason.MISSING_BACKEND,
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


class _Reason(str, Enum):
    """Internal reason taxonomy used only to compose the unavailability warning."""

    UNSUPPORTED_PLATFORM = "unsupported-platform"
    MISSING_BACKEND = "missing-backend"
    NAMESPACE_DENIED = "namespace-denied"


@dataclass(frozen=True)
class _SandboxUnavailable:
    """Internal carrier for the unavailability warning text; never escapes the module."""

    reason: _Reason
    diagnostic: str
    remediation: str


def _emit_unavailable_warning(unavailable: _SandboxUnavailable) -> None:
    """Log + print the unavailability warning so it shows up everywhere users look."""
    warning = (
        "HALO run_code disabled: sandbox unavailable.\n"
        "\n"
        f"Reason ({unavailable.reason.value}):\n"
        f"  {unavailable.diagnostic}\n"
        "\n"
        "How to fix:\n"
        f"{_indent(unavailable.remediation, '  ')}\n"
        "\n"
        "The engine will continue without exposing run_code to the agent."
    )
    _logger.warning("\n%s", warning)
    print(warning, file=sys.stderr, flush=True)


def _indent(text: str, prefix: str) -> str:
    """Prefix each line of ``text`` with ``prefix``."""
    return "\n".join(f"{prefix}{line}" if line else line for line in text.splitlines())
