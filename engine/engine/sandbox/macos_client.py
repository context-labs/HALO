from __future__ import annotations

import shutil
from pathlib import Path

from engine.sandbox.log import log_unavailable

# SIP-protected system roots Python's dynamic loader, CoreFoundation, and
# stdlib load executables/data from. These are read-only on the host and
# not user-controlled. Modeled after Apple's own ``system.sb``.
_MACOS_SYSTEM_ROOT_SUBPATHS = (
    "/usr/lib",
    "/usr/share",
    "/System",
    "/Library/Apple",
    "/private/var/db/timezone",
)

# Specific files Python and CoreFoundation read at startup. ``system.sb``
# enumerates these explicitly because they live in directories where we
# don't want to grant the entire subtree.
_MACOS_SYSTEM_ROOT_LITERALS = (
    "/",
    "/dev/random",
    "/dev/urandom",
    "/dev/null",
    "/private/etc/passwd",
    "/private/etc/protocols",
    "/private/etc/services",
    "/private/etc/localtime",
)

# Cache directory used by CoreFoundation, dyld, and Python ``tempfile``
# fallbacks. Per-user but does not contain credentials/secrets.
_MACOS_FOLDERS_CACHE = "/private/var/folders"


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
        """Render a default-deny ``sandbox-exec`` Scheme profile for one run.

        Security model (mirrors Apple's own ``system.sb`` baseline):

          - ``(deny default)`` is the floor — nothing is reachable unless
            we explicitly allow it.
          - ``(allow file-read-metadata)`` is granted everywhere. This is
            what lets the kernel traverse a path during ``open()`` /
            ``stat()``: the lookup of each directory component requires
            metadata access. It does NOT allow reading file contents and
            does NOT allow listing directory entries — those need
            ``file-read-data`` (a subset of ``file-read*``), which we only
            grant on the explicit allow list below. Net effect: the model
            can resolve ``/Users/x/.../venv/bin/python`` without us having
            to enumerate every ancestor, but ``os.listdir('/Users/x')``
            and ``open('/Users/x/secrets.txt')`` are both denied.
          - Full read on SIP-protected system roots (``/usr/lib``,
            ``/System``, etc.) and a small set of system literal files
            (``/dev/urandom``, ``/private/etc/passwd``, ...) — same set
            ``system.sb`` allows for system daemons. These contain no
            user data.
          - Full read on each ``readonly_paths`` entry (interpreter,
            site-packages, trace, index).
          - Full read+write on each ``writable_paths`` entry.
          - ``mach-lookup``/``ipc-posix-shm``/``sysctl-read``/``signal``/
            ``process*`` are allowed because Python and CoreFoundation
            depend on them for startup.
          - Network is denied wholesale.
        """
        lines: list[str] = [
            "(version 1)",
            "(deny default)",
            "(allow process*)",
            "(allow mach-lookup)",
            "(allow ipc-posix-shm)",
            "(allow sysctl-read)",
            "(allow signal)",
            "(allow file-read-metadata)",
        ]

        for system_root in _MACOS_SYSTEM_ROOT_SUBPATHS:
            lines.append(f'(allow file-read* file-map-executable (subpath "{system_root}"))')
        for system_literal in _MACOS_SYSTEM_ROOT_LITERALS:
            lines.append(f'(allow file-read* (literal "{system_literal}"))')
        lines.append(f'(allow file-read* (subpath "{_MACOS_FOLDERS_CACHE}"))')

        for path in readonly_paths:
            kind = "subpath" if path.is_dir() else "literal"
            lines.append(f'(allow file-read* ({kind} "{path}"))')

        for path in writable_paths:
            lines.append(f'(allow file-read* (subpath "{path}"))')
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
