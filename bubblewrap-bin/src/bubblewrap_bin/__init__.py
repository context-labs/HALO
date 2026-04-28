"""Bundled bubblewrap binary as a Python dependency on Linux.

The package ships a statically-linked ``bwrap`` executable inside the wheel
under ``bubblewrap_bin/_bin/bwrap``. ``bwrap_path()`` returns the absolute
path so callers can invoke ``bwrap`` directly. The Python package itself
is intentionally minimal — anything more complex (AppArmor policy,
sandbox setup) is the responsibility of the host or the consumer.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

__all__ = ["BubblewrapNotBundledError", "bwrap_path"]


class BubblewrapNotBundledError(RuntimeError):
    """Raised when this wheel was not built with a packaged ``bwrap`` binary.

    Pure-source installs (``sdist``) do not contain the binary; this error
    points users at the prebuilt platform wheels.
    """


def bwrap_path() -> Path:
    """Return the absolute path of the bundled ``bwrap`` executable.

    Raises:
        BubblewrapNotBundledError: if no packaged binary is present (for
            example, when installed from sdist on a platform without
            prebuilt wheels).
    """
    candidate = Path(__file__).parent / "_bin" / "bwrap"
    if not candidate.is_file():
        raise BubblewrapNotBundledError(
            "bubblewrap-bin is installed but no packaged bwrap binary was found at "
            f"{candidate}. This usually means the package was installed from sdist; "
            "install a platform wheel built with the binary instead."
        )

    mode = candidate.stat().st_mode
    if not (mode & stat.S_IXUSR):
        # Defensive: hatchling preserves +x via permissions metadata when the
        # source file has it set, but harden against the edge case of the
        # bit being lost in transit.
        os.chmod(candidate, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return candidate
