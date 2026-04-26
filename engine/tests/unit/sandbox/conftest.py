from __future__ import annotations

import shutil
import subprocess


def bwrap_can_sandbox() -> bool:
    """Probe whether bubblewrap is installed AND able to create a user-namespace
    sandbox on this host. Returns False inside containers/kernels that disallow
    user-namespace proc mounts (the typical CI / cloud-shell case)."""
    if shutil.which("bwrap") is None:
        return False
    try:
        result = subprocess.run(
            ["bwrap", "--unshare-all", "--proc", "/proc", "--dev", "/dev", "--", "/bin/true"],
            capture_output=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0
