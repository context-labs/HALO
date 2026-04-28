from __future__ import annotations

import shutil
from pathlib import Path

from engine.sandbox.log import log_unavailable

# Subdirectories under ``$HOME`` that we explicitly deny reads on. These are
# the common locations of secrets/credentials/private user data. We do NOT
# wholesale-deny ``$HOME`` itself because that would block path traversal to
# allowed deeper paths (e.g. a venv living under ``$HOME/work/...``):
# the macOS sandbox checks read permission on directory traversal, not just
# on the final target, so denying a parent of an allowed path makes the
# allowed path unreachable.
_MACOS_HOME_DENY_SUBDIRS = (
    ".ssh",
    ".aws",
    ".gnupg",
    ".config",
    ".docker",
    ".kube",
    ".gcloud",
    "Documents",
    "Desktop",
    "Downloads",
)


class MacosClient:
    """``sandbox-exec``-backed sandbox client.

    Wraps the resolved ``sandbox-exec`` binary. Render a Scheme profile per
    run via :meth:`render_profile` and build the invocation argv via
    :meth:`build_argv`.
    """

    def __init__(self, *, executable: Path) -> None:
        self.executable = executable

    @staticmethod
    def resolve() -> MacosClient | None:
        """Locate ``sandbox-exec`` on PATH; log + return ``None`` when missing."""
        found = shutil.which("sandbox-exec")
        if found is None:
            log_unavailable(
                diagnostic="sandbox-exec not found on PATH",
                remediation=(
                    "sandbox-exec ships with macOS by default. Ensure /usr/bin is on PATH."
                ),
            )
            return None
        return MacosClient(executable=Path(found))

    def render_profile(
        self,
        *,
        readonly_paths: list[Path],
        writable_paths: list[Path],
    ) -> str:
        """Render the Scheme profile for one ``sandbox-exec`` invocation.

        Security model on macOS:
          - ``deny default`` is the floor. ``file-read*`` is then broadly
            allowed because the macOS dynamic loader and CoreFoundation read
            from many opaque locations (``/usr/lib``, ``/System``,
            ``/private/var/db``, framework-specific paths) that are
            impractical to enumerate and are SIP-protected on the host.
            Narrowing reads here would prevent Python from starting.
          - ``file-write*`` is denied by default and re-allowed only on the
            explicit ``writable_paths``. This is the actual filesystem
            boundary — sandboxed code cannot mutate anything outside the
            workspace.
          - Network is denied entirely.
          - ``mach-lookup``/``ipc-posix-shm``/``sysctl-read``/``signal`` are
            allowed because Python and CoreFoundation rely on them for
            normal startup; without these, the interpreter aborts before
            reaching user code.

        Reads of common credential/secret directories under ``$HOME`` are
        denied (``.ssh``, ``.aws``, etc.). We deliberately do NOT
        wholesale-deny ``$HOME``: macOS sandbox-exec checks read permission
        on every component of a path during traversal, so denying the home
        directory itself blocks reach-through to allowed deeper paths
        (e.g. a venv living at ``$HOME/work/.../.venv``).
        """
        lines: list[str] = [
            "(version 1)",
            "(deny default)",
            "(allow process*)",
            "(allow mach-lookup)",
            "(allow ipc-posix-shm)",
            "(allow sysctl-read)",
            "(allow signal)",
            "(allow file-read*)",
        ]

        home_dir = _home_dir()
        if home_dir is not None:
            for subdir in _MACOS_HOME_DENY_SUBDIRS:
                lines.append(f'(deny file-read* (subpath "{home_dir}/{subdir}"))')

        for path in readonly_paths:
            kind = "subpath" if path.is_dir() else "literal"
            lines.append(f'(allow file-read* ({kind} "{path}"))')

        for path in writable_paths:
            lines.append(f'(allow file-write* (subpath "{path}"))')

        lines.append("(deny network*)")
        return "\n".join(lines) + "\n"

    def build_argv(
        self,
        *,
        python_executable: Path,
        script_path: Path,
        profile_path: Path,
        work_dir: Path,
    ) -> list[str]:
        """Build the ``sandbox-exec`` argv: profile flag, ``env -i`` scrub, then python script."""
        return [
            str(self.executable),
            "-f",
            str(profile_path),
            "env",
            "-i",
            f"PATH={python_executable.parent}:/usr/bin:/bin",
            f"HOME={work_dir}",
            "LANG=C.UTF-8",
            "PYTHONDONTWRITEBYTECODE=1",
            "PYTHONUNBUFFERED=1",
            f"TMPDIR={work_dir / 'tmp'}",
            str(python_executable),
            str(script_path),
        ]


def _home_dir() -> str | None:
    """Resolve the current user's home directory; ``None`` if unavailable."""
    try:
        home = Path.home()
    except (RuntimeError, KeyError):
        return None
    return str(home) if home else None
