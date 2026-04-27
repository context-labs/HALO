from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from engine.sandbox import sandbox_availability
from engine.sandbox.sandbox_availability import (
    HALO_BWRAP_ENV_VAR,
    SandboxBackend,
    SandboxStatus,
    SandboxUnavailableReason,
    render_unavailable_warning,
    resolve_sandbox_status,
)


def _make_completed_process(
    returncode: int, stderr: str = ""
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=b"", stderr=stderr.encode()
    )


def test_status_ok_factory(tmp_path: Path) -> None:
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")
    status = SandboxStatus.ok(backend=SandboxBackend.LINUX_BWRAP_SYSTEM, executable=bwrap)
    assert status.available is True
    assert status.backend == SandboxBackend.LINUX_BWRAP_SYSTEM
    assert status.executable == bwrap
    assert status.reason is None


def test_status_unavailable_factory() -> None:
    status = SandboxStatus.unavailable(
        reason=SandboxUnavailableReason.MISSING_BACKEND,
        diagnostic="bwrap not found",
        remediation="install bubblewrap-bin",
    )
    assert status.available is False
    assert status.backend is None
    assert status.executable is None
    assert status.reason == SandboxUnavailableReason.MISSING_BACKEND


def test_render_warning_includes_reason_and_remediation() -> None:
    status = SandboxStatus.unavailable(
        reason=SandboxUnavailableReason.NAMESPACE_DENIED,
        diagnostic="user-namespace creation rejected",
        remediation="enable userns",
    )
    rendered = render_unavailable_warning(status)
    assert "run_code disabled" in rendered
    assert "namespace-denied" in rendered
    assert "user-namespace creation rejected" in rendered
    assert "enable userns" in rendered


def test_render_warning_rejects_available_status(tmp_path: Path) -> None:
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")
    status = SandboxStatus.ok(backend=SandboxBackend.LINUX_BWRAP_SYSTEM, executable=bwrap)
    with pytest.raises(ValueError):
        render_unavailable_warning(status)


@pytest.mark.skipif(platform.system() != "Linux", reason="probe-flow tests target Linux resolution")
def test_resolve_linux_uses_env_var_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_bwrap = tmp_path / "env-bwrap"
    env_bwrap.write_text("")

    monkeypatch.setenv(HALO_BWRAP_ENV_VAR, str(env_bwrap))
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: "/host/bwrap")
    monkeypatch.setattr(sandbox_availability, "_packaged_bwrap", lambda: None)
    monkeypatch.setattr(
        sandbox_availability.subprocess,
        "run",
        lambda *_a, **_kw: _make_completed_process(0),
    )

    status = resolve_sandbox_status()
    assert status.available is True
    assert status.executable == env_bwrap


@pytest.mark.skipif(platform.system() != "Linux", reason="probe-flow tests target Linux resolution")
def test_resolve_linux_falls_back_to_packaged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packaged = tmp_path / "packaged-bwrap"
    packaged.write_text("")

    monkeypatch.delenv(HALO_BWRAP_ENV_VAR, raising=False)
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: None)
    monkeypatch.setattr(sandbox_availability, "_packaged_bwrap", lambda: packaged)
    monkeypatch.setattr(
        sandbox_availability.subprocess,
        "run",
        lambda *_a, **_kw: _make_completed_process(0),
    )

    status = resolve_sandbox_status()
    assert status.available is True
    assert status.backend == SandboxBackend.LINUX_BWRAP_PACKAGED
    assert status.executable == packaged


@pytest.mark.skipif(platform.system() != "Linux", reason="probe-flow tests target Linux resolution")
def test_resolve_linux_missing_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(HALO_BWRAP_ENV_VAR, raising=False)
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: None)
    monkeypatch.setattr(sandbox_availability, "_packaged_bwrap", lambda: None)

    status = resolve_sandbox_status()
    assert status.available is False
    assert status.reason == SandboxUnavailableReason.MISSING_BACKEND
    assert "bubblewrap" in status.remediation.lower()


@pytest.mark.skipif(platform.system() != "Linux", reason="probe-flow tests target Linux resolution")
def test_resolve_linux_namespace_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")

    monkeypatch.delenv(HALO_BWRAP_ENV_VAR, raising=False)
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: str(bwrap))
    monkeypatch.setattr(sandbox_availability, "_packaged_bwrap", lambda: None)
    monkeypatch.setattr(
        sandbox_availability.subprocess,
        "run",
        lambda *_a, **_kw: _make_completed_process(1, "Operation not permitted"),
    )

    status = resolve_sandbox_status()
    assert status.available is False
    assert status.reason == SandboxUnavailableReason.NAMESPACE_DENIED
    assert "Operation not permitted" in status.diagnostic


@pytest.mark.skipif(
    platform.system() != "Darwin", reason="probe-flow tests target macOS resolution"
)
def test_resolve_macos_finds_sandbox_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sandbox_availability.shutil,
        "which",
        lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None,
    )

    status = resolve_sandbox_status()
    assert status.available is True
    assert status.backend == SandboxBackend.MACOS_SANDBOX_EXEC
    assert status.executable == Path("/usr/bin/sandbox-exec")


@pytest.mark.skipif(
    platform.system() != "Darwin", reason="probe-flow tests target macOS resolution"
)
def test_resolve_macos_missing_sandbox_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: None)

    status = resolve_sandbox_status()
    assert status.available is False
    assert status.reason == SandboxUnavailableReason.MISSING_BACKEND


def test_resolve_unsupported_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Windows")
    status = resolve_sandbox_status()
    assert status.available is False
    assert status.reason == SandboxUnavailableReason.UNSUPPORTED_PLATFORM
