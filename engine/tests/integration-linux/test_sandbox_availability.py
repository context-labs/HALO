from __future__ import annotations

import pytest

from engine.sandbox.sandbox_availability import (
    SandboxBackend,
    _probe_bwrap,
    render_unavailable_warning,
    resolve_sandbox_status,
)


def test_linux_bwrap_probe_succeeds_with_empty_rootfs() -> None:
    status = resolve_sandbox_status()
    if not status.available or status.executable is None:
        pytest.fail(
            "Linux sandbox unavailable in CI; this must work for release.\n"
            + render_unavailable_warning(status)
        )

    assert status.backend in (
        SandboxBackend.LINUX_BWRAP_SYSTEM,
        SandboxBackend.LINUX_BWRAP_PACKAGED,
    )

    ok, diagnostic = _probe_bwrap(status.executable)
    assert ok is True
    assert diagnostic == ""
