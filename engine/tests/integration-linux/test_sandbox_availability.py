from __future__ import annotations

import pytest

from engine.sandbox.linux_client import LinuxClient
from engine.sandbox.models import SandboxConfig
from engine.sandbox.sandbox import resolve_sandbox


def test_linux_resolve_sandbox_returns_working_bubblewrap_client() -> None:
    """End-to-end probe must succeed in CI; failure here is a release blocker."""
    sandbox = resolve_sandbox(config=SandboxConfig())
    if sandbox is None:
        pytest.fail("Linux sandbox unavailable in CI; this must work for release.")

    assert isinstance(sandbox.client, LinuxClient)
    assert sandbox.client.executable.is_file()
