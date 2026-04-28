from __future__ import annotations

import platform

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the entire ``tests/integration/linux`` collection on non-Linux hosts.

    These tests exercise real bubblewrap and only run in the Linux sandbox CI
    job. We skip rather than fail on macOS/Windows so contributors can run the
    full integration suite locally without false failures.
    """
    if platform.system() == "Linux":
        return
    skip_marker = pytest.mark.skip(reason="linux integration tests require Linux")
    for item in items:
        item.add_marker(skip_marker)
