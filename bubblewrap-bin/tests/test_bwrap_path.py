from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import bubblewrap_bin
from bubblewrap_bin import BubblewrapNotBundledError, bwrap_path


def _packaged_binary() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "bubblewrap_bin" / "_bin" / "bwrap"


def test_bwrap_path_raises_when_binary_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``bwrap_path`` must raise ``BubblewrapNotBundledError`` when no binary is packaged.

    Simulated by pointing the module's ``__file__`` at a temp dir that has
    no ``_bin/bwrap``. ``bwrap_path`` derives the candidate from ``__file__``
    so the override is enough; no reload required.
    """
    fake_init = tmp_path / "bubblewrap_bin" / "__init__.py"
    fake_init.parent.mkdir()
    fake_init.write_text("")

    monkeypatch.setattr(bubblewrap_bin, "__file__", str(fake_init))

    with pytest.raises(BubblewrapNotBundledError):
        bwrap_path()


@pytest.mark.skipif(
    not _packaged_binary().exists(),
    reason="packaged bwrap binary not built; run scripts/build_bwrap.sh on Linux",
)
def test_bwrap_path_returns_executable_when_bundled() -> None:
    path = bwrap_path()
    assert path.is_file()
    assert os.access(path, os.X_OK)


@pytest.mark.skipif(
    not _packaged_binary().exists(),
    reason="packaged bwrap binary not built; run scripts/build_bwrap.sh on Linux",
)
def test_packaged_bwrap_runs_version() -> None:
    result = subprocess.run(
        [str(bwrap_path()), "--version"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert "bubblewrap" in (result.stdout + result.stderr)
