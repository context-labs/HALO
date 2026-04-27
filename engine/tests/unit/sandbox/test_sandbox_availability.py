from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from engine.sandbox import sandbox_availability
from engine.sandbox.sandbox_availability import (
    HALO_BWRAP_ENV_VAR,
    SandboxBackend,
    SandboxUnavailableReason,
    render_unavailable_warning,
    resolve_sandbox_status,
)


def _install_fake_bwrap(
    monkeypatch: pytest.MonkeyPatch,
    *,
    info_fd_payload: bytes,
    stderr: bytes = b"",
    captured_argv: list[str] | None = None,
) -> None:
    def _fake_popen(argv, *args, **kwargs):
        if captured_argv is not None:
            captured_argv[:] = list(argv)
        info_fd = int(argv[argv.index("--info-fd") + 1])
        if info_fd_payload:
            os.write(info_fd, info_fd_payload)

        proc = MagicMock(spec=["wait", "kill", "returncode", "stderr"])
        proc.returncode = 127
        proc.wait.return_value = 127
        proc.stderr.read.return_value = stderr
        return proc

    monkeypatch.setattr(sandbox_availability.subprocess, "Popen", _fake_popen)


def test_linux_probe_argv_is_explicit_and_empty_rootfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")
    captured_argv: list[str] = []

    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Linux")
    monkeypatch.delenv(HALO_BWRAP_ENV_VAR, raising=False)
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: str(bwrap))
    monkeypatch.setattr(sandbox_availability, "_packaged_bwrap", lambda: None)
    _install_fake_bwrap(
        monkeypatch,
        info_fd_payload=b'{"child-pid":42}',
        captured_argv=captured_argv,
    )

    status = resolve_sandbox_status()

    assert status.available is True
    info_fd = captured_argv[7]
    assert captured_argv == [
        str(bwrap),
        "--unshare-all",
        "--unshare-net",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--info-fd",
        info_fd,
        "--",
        sandbox_availability._PROBE_EXEC_PATH,
    ]


def test_linux_resolves_env_bwrap_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_bwrap = tmp_path / "env-bwrap"
    env_bwrap.write_text("")

    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Linux")
    monkeypatch.setenv(HALO_BWRAP_ENV_VAR, str(env_bwrap))
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: "/host/bwrap")
    monkeypatch.setattr(sandbox_availability, "_packaged_bwrap", lambda: None)
    _install_fake_bwrap(monkeypatch, info_fd_payload=b'{"child-pid":42}')

    status = resolve_sandbox_status()

    assert status.available is True
    assert status.backend == SandboxBackend.LINUX_BWRAP_SYSTEM
    assert status.executable == env_bwrap


def test_linux_resolves_packaged_bwrap_when_system_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packaged = tmp_path / "packaged-bwrap"
    packaged.write_text("")

    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Linux")
    monkeypatch.delenv(HALO_BWRAP_ENV_VAR, raising=False)
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: None)
    monkeypatch.setattr(sandbox_availability, "_packaged_bwrap", lambda: packaged)
    _install_fake_bwrap(monkeypatch, info_fd_payload=b'{"child-pid":42}')

    status = resolve_sandbox_status()

    assert status.available is True
    assert status.backend == SandboxBackend.LINUX_BWRAP_PACKAGED
    assert status.executable == packaged


def test_linux_missing_bwrap_reports_missing_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Linux")
    monkeypatch.delenv(HALO_BWRAP_ENV_VAR, raising=False)
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: None)
    monkeypatch.setattr(sandbox_availability, "_packaged_bwrap", lambda: None)

    status = resolve_sandbox_status()

    assert status.available is False
    assert status.reason == SandboxUnavailableReason.MISSING_BACKEND


def test_linux_missing_env_path_reports_missing_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nonexistent = tmp_path / "does-not-exist"

    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Linux")
    monkeypatch.setenv(HALO_BWRAP_ENV_VAR, str(nonexistent))
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: None)
    monkeypatch.setattr(sandbox_availability, "_packaged_bwrap", lambda: None)

    status = resolve_sandbox_status()

    assert status.available is False
    assert status.reason == SandboxUnavailableReason.MISSING_BACKEND
    assert (
        status.diagnostic
        == f"bwrap candidate from {HALO_BWRAP_ENV_VAR} override does not exist: {nonexistent}"
    )


def test_linux_probe_failure_reports_namespace_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")

    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Linux")
    monkeypatch.delenv(HALO_BWRAP_ENV_VAR, raising=False)
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: str(bwrap))
    monkeypatch.setattr(sandbox_availability, "_packaged_bwrap", lambda: None)
    _install_fake_bwrap(
        monkeypatch,
        info_fd_payload=b"",
        stderr=b"bwrap: setting up uid map: Operation not permitted\n",
    )

    status = resolve_sandbox_status()

    assert status.available is False
    assert status.reason == SandboxUnavailableReason.NAMESPACE_DENIED
    assert "Operation not permitted" in status.diagnostic


def test_macos_resolves_sandbox_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        sandbox_availability.shutil,
        "which",
        lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None,
    )

    status = resolve_sandbox_status()

    assert status.available is True
    assert status.backend == SandboxBackend.MACOS_SANDBOX_EXEC
    assert status.executable == Path("/usr/bin/sandbox-exec")


def test_macos_missing_sandbox_exec_reports_missing_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: None)

    status = resolve_sandbox_status()

    assert status.available is False
    assert status.reason == SandboxUnavailableReason.MISSING_BACKEND


def test_unsupported_platform_reports_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Windows")

    status = resolve_sandbox_status()

    assert status.available is False
    assert status.reason == SandboxUnavailableReason.UNSUPPORTED_PLATFORM


def test_unavailable_warning_includes_reason_and_remediation() -> None:
    status = sandbox_availability.SandboxStatus.unavailable(
        reason=SandboxUnavailableReason.NAMESPACE_DENIED,
        diagnostic="user namespaces disabled",
        remediation="enable user namespaces",
    )

    rendered = render_unavailable_warning(status)

    assert "Reason (namespace-denied):\n  user namespaces disabled" in rendered
    assert "  enable user namespaces" in rendered
