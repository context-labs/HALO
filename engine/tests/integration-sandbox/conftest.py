from __future__ import annotations

import pytest

from engine.sandbox.pyodide_client import _locate_deno_executable


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the sandbox integration suite when no Deno binary is available.

    Out-of-the-box, the ``deno`` PyPI dependency provides one. We only
    skip if ``_locate_deno_executable`` (which checks both the bundled
    binary and PATH) finds nothing — that indicates an unsupported
    platform (e.g. musl Linux without a manual Deno install) where the
    sandbox can't run regardless. The skip is scoped to this directory
    so it never bleeds into unit tests.
    """
    if _locate_deno_executable() is not None:
        return
    skip_marker = pytest.mark.skip(
        reason="sandbox integration tests require a Deno binary "
        "(install the ``deno`` PyPI dep or place ``deno`` on PATH)"
    )
    suite_dir = str(config.rootpath / "tests" / "integration-sandbox")
    for item in items:
        if str(item.path).startswith(suite_dir):
            item.add_marker(skip_marker)
