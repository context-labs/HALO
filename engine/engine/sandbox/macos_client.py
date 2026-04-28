from __future__ import annotations

import shutil
from pathlib import Path

from engine.sandbox.log import log_unavailable


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

        Reads of ``$HOME`` (typical user-data location) are denied even
        though ``file-read*`` is otherwise broad. Each ``readonly_paths``
        entry is then re-allowed by literal/subpath, so trace and index
        files remain accessible regardless of where they live.
        """
        re_allows_read = "\n".join(
            f'(allow file-read* (subpath "{p}"))'
            if p.is_dir()
            else f'(allow file-read* (literal "{p}"))'
            for p in readonly_paths
        )
        allows_write = "\n".join(f'(allow file-write* (subpath "{p}"))' for p in writable_paths)
        home_dir = _home_dir()
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
