from __future__ import annotations

import platform

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the entire ``tests/integration/macos`` collection on non-macOS hosts.

    These tests exercise real ``sandbox-exec`` and only run in the macOS
    sandbox CI job. We skip rather than fail on Linux/Windows so contributors
    can run the full integration suite locally without false failures.
    """
    if platform.system() == "Darwin":
        return
    skip_marker = pytest.mark.skip(reason="macos integration tests require Darwin")
    for item in items:
        item.add_marker(skip_marker)
