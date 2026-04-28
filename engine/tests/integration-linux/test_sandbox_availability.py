from __future__ import annotations

import pytest

from engine.sandbox.sandbox_availability import (
    SandboxBackend,
    _probe_bwrap,
    resolve_sandbox_runtime,
)


def test_linux_bwrap_probe_succeeds_with_empty_rootfs() -> None:
    sandbox = resolve_sandbox_runtime()
    if sandbox is None:
        pytest.fail("Linux sandbox unavailable in CI; this must work for release.")

    assert sandbox.backend in (
        SandboxBackend.LINUX_BWRAP_SYSTEM,
        SandboxBackend.LINUX_BWRAP_PACKAGED,
    )

    ok, diagnostic = _probe_bwrap(sandbox.executable)
    assert ok is True
    assert diagnostic == ""
