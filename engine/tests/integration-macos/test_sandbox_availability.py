from __future__ import annotations

import pytest

from engine.sandbox.macos_client import MacosClient
from engine.sandbox.models import SandboxConfig
from engine.sandbox.sandbox import resolve_sandbox


def test_macos_resolve_sandbox_returns_working_sandbox_exec_client() -> None:
    """End-to-end probe must succeed in CI; failure here is a release blocker."""
    sandbox = resolve_sandbox(config=SandboxConfig())
    if sandbox is None:
        pytest.fail("macOS sandbox unavailable in CI; this must work for release.")

    assert isinstance(sandbox.client, MacosClient)
    assert sandbox.client.executable.is_file()
