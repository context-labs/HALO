from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from engine.sandbox import sandbox_availability
from engine.sandbox.sandbox_availability import (
    HALO_BWRAP_ENV_VAR,
    SandboxBackend,
    resolve_sandbox_runtime,
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

    runtime = resolve_sandbox_runtime()

    assert runtime is not None
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

    runtime = resolve_sandbox_runtime()

    assert runtime is not None
    assert runtime.backend == SandboxBackend.LINUX_BWRAP_SYSTEM
    assert runtime.executable == env_bwrap


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

    runtime = resolve_sandbox_runtime()

    assert runtime is not None
    assert runtime.backend == SandboxBackend.LINUX_BWRAP_PACKAGED
    assert runtime.executable == packaged


def test_linux_missing_bwrap_returns_none(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Linux")
    monkeypatch.delenv(HALO_BWRAP_ENV_VAR, raising=False)
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: None)
    monkeypatch.setattr(sandbox_availability, "_packaged_bwrap", lambda: None)

    runtime = resolve_sandbox_runtime()

    assert runtime is None
    err = capsys.readouterr().err
    assert "Reason (missing-backend):" in err
    assert "bubblewrap" in err.lower()


def test_linux_missing_env_path_returns_none_with_path_in_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    nonexistent = tmp_path / "does-not-exist"

    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Linux")
    monkeypatch.setenv(HALO_BWRAP_ENV_VAR, str(nonexistent))
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: None)
    monkeypatch.setattr(sandbox_availability, "_packaged_bwrap", lambda: None)

    runtime = resolve_sandbox_runtime()

    assert runtime is None
    err = capsys.readouterr().err
    assert "Reason (missing-backend):" in err
    assert str(nonexistent) in err


def test_linux_probe_failure_returns_none_with_namespace_remediation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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

    runtime = resolve_sandbox_runtime()

    assert runtime is None
    err = capsys.readouterr().err
    assert "Reason (namespace-denied):" in err
    assert "Operation not permitted" in err


def test_macos_resolves_sandbox_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        sandbox_availability.shutil,
        "which",
        lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None,
    )

    runtime = resolve_sandbox_runtime()

    assert runtime is not None
    assert runtime.backend == SandboxBackend.MACOS_SANDBOX_EXEC
    assert runtime.executable == Path("/usr/bin/sandbox-exec")


def test_macos_missing_sandbox_exec_returns_none(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: None)

    runtime = resolve_sandbox_runtime()

    assert runtime is None
    assert "Reason (missing-backend):" in capsys.readouterr().err


def test_unsupported_platform_returns_none(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Windows")

    runtime = resolve_sandbox_runtime()

    assert runtime is None
    assert "Reason (unsupported-platform):" in capsys.readouterr().err
