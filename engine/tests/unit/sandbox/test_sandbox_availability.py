from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

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


def _install_fake_bwrap(
    monkeypatch: pytest.MonkeyPatch,
    *,
    info_fd_payload: bytes,
    stderr: bytes = b"",
    returncode: int = 127,
    captured_argv: list[list[str]] | None = None,
) -> None:
    """Patch ``subprocess.Popen`` to mimic bwrap writing ``info_fd_payload`` to ``--info-fd``.

    Whatever payload is given is written to the FD passed by the caller via
    ``--info-fd``. Returns the requested exit code (defaults to 127 — what
    real bwrap returns when the sandboxed exec fails with ENOENT, our
    expected case for the probe).
    """

    def _fake_popen(argv, *args, **kwargs):
        if captured_argv is not None:
            captured_argv.append(list(argv))
        # Find the --info-fd value in argv.
        info_fd = None
        for i, token in enumerate(argv):
            if token == "--info-fd" and i + 1 < len(argv):
                info_fd = int(argv[i + 1])
                break
        if info_fd is not None and info_fd_payload:
            os.write(info_fd, info_fd_payload)

        proc = MagicMock(spec=["wait", "kill", "returncode", "stderr"])
        proc.returncode = returncode
        proc.wait.return_value = returncode
        stderr_buf = MagicMock()
        stderr_buf.read.return_value = stderr
        proc.stderr = stderr_buf
        return proc

    monkeypatch.setattr(sandbox_availability.subprocess, "Popen", _fake_popen)


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


def test_resolve_linux_uses_env_var_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_bwrap = tmp_path / "env-bwrap"
    env_bwrap.write_text("")

    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Linux")
    monkeypatch.setenv(HALO_BWRAP_ENV_VAR, str(env_bwrap))
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: "/host/bwrap")
    monkeypatch.setattr(sandbox_availability, "_packaged_bwrap", lambda: None)
    _install_fake_bwrap(monkeypatch, info_fd_payload=b'{"child-pid":42}')

    status = resolve_sandbox_status()
    assert status.available is True
    assert status.executable == env_bwrap


def test_resolve_linux_falls_back_to_packaged(
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


def test_resolve_linux_missing_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Linux")
    monkeypatch.delenv(HALO_BWRAP_ENV_VAR, raising=False)
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: None)
    monkeypatch.setattr(sandbox_availability, "_packaged_bwrap", lambda: None)

    status = resolve_sandbox_status()
    assert status.available is False
    assert status.reason == SandboxUnavailableReason.MISSING_BACKEND
    assert "bubblewrap" in status.remediation.lower()


def test_resolve_linux_namespace_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")

    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Linux")
    monkeypatch.delenv(HALO_BWRAP_ENV_VAR, raising=False)
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: str(bwrap))
    monkeypatch.setattr(sandbox_availability, "_packaged_bwrap", lambda: None)
    # No info-fd payload simulates bwrap exiting before namespace setup
    # completed (e.g., kernel rejected user-namespace creation).
    _install_fake_bwrap(
        monkeypatch,
        info_fd_payload=b"",
        stderr=b"bwrap: setting up uid map: Operation not permitted\n",
        returncode=1,
    )

    status = resolve_sandbox_status()
    assert status.available is False
    assert status.reason == SandboxUnavailableReason.NAMESPACE_DENIED
    assert "Operation not permitted" in status.diagnostic


def test_resolve_linux_env_path_missing_reports_missing_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``HALO_BWRAP_PATH`` points to a non-existent file and there is no
    other candidate, the failure is ``MISSING_BACKEND`` (not
    ``NAMESPACE_DENIED``) and the remediation tells the user to install the
    binary, not loosen kernel policy."""
    nonexistent = tmp_path / "does-not-exist"

    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Linux")
    monkeypatch.setenv(HALO_BWRAP_ENV_VAR, str(nonexistent))
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: None)
    monkeypatch.setattr(sandbox_availability, "_packaged_bwrap", lambda: None)

    status = resolve_sandbox_status()
    assert status.available is False
    assert status.reason == SandboxUnavailableReason.MISSING_BACKEND
    assert str(nonexistent) in status.diagnostic
    assert "does not exist" in status.diagnostic
    assert "install" in status.remediation.lower()


def test_resolve_linux_some_missing_some_probe_failed_reports_namespace_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing env override plus a system bwrap whose probe fails reports
    ``NAMESPACE_DENIED`` — at least one real binary was tried and failed."""
    nonexistent = tmp_path / "no-such-bwrap"
    real_bwrap = tmp_path / "real-bwrap"
    real_bwrap.write_text("")

    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Linux")
    monkeypatch.setenv(HALO_BWRAP_ENV_VAR, str(nonexistent))
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: str(real_bwrap))
    monkeypatch.setattr(sandbox_availability, "_packaged_bwrap", lambda: None)
    _install_fake_bwrap(
        monkeypatch,
        info_fd_payload=b"",
        stderr=b"bwrap: setting up uid map: Operation not permitted\n",
        returncode=1,
    )

    status = resolve_sandbox_status()
    assert status.available is False
    assert status.reason == SandboxUnavailableReason.NAMESPACE_DENIED
    assert "Operation not permitted" in status.diagnostic


def test_probe_argv_has_no_filesystem_binds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe must run with an entirely empty mount namespace.

    The ``--info-fd`` design lets bwrap report namespace setup completion
    before exec, so the probe deliberately exec's a nonexistent path and
    binds nothing from the host filesystem (no ``/usr``, no ``/bin``, no
    ``--dev``, no ``--ro-bind`` of any kind).
    """
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")

    captured_argv: list[list[str]] = []
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

    assert len(captured_argv) == 1
    argv = captured_argv[0]
    assert argv[0] == str(bwrap)
    for flag in (
        "--unshare-all",
        "--unshare-net",
        "--clearenv",
        "--die-with-parent",
        "--new-session",
    ):
        assert flag in argv, argv
    # No filesystem mounts of any kind.
    assert "--ro-bind" not in argv, argv
    assert "--bind" not in argv, argv
    assert "--dev" not in argv, argv
    assert "--proc" not in argv, argv
    # ``--info-fd`` must be present so bwrap can signal setup completion.
    assert "--info-fd" in argv
    # Probe execs a deliberately nonexistent path. It is fine for exec to
    # fail with ENOENT — by then bwrap has already written to ``--info-fd``.
    assert argv[-1] == sandbox_availability._PROBE_EXEC_PATH


def test_probe_returns_namespace_denied_when_no_info_fd_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If bwrap exits without writing to ``--info-fd``, namespace setup
    failed before exec and we must report ``NAMESPACE_DENIED`` carrying
    bwrap's stderr in the diagnostic."""
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")

    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Linux")
    monkeypatch.delenv(HALO_BWRAP_ENV_VAR, raising=False)
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: str(bwrap))
    monkeypatch.setattr(sandbox_availability, "_packaged_bwrap", lambda: None)
    _install_fake_bwrap(
        monkeypatch,
        info_fd_payload=b"",
        stderr=b"bwrap: creating new namespace: Permission denied\n",
        returncode=1,
    )

    status = resolve_sandbox_status()
    assert status.available is False
    assert status.reason == SandboxUnavailableReason.NAMESPACE_DENIED
    assert "Permission denied" in status.diagnostic


def test_resolve_macos_finds_sandbox_exec(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_resolve_macos_missing_sandbox_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sandbox_availability.shutil, "which", lambda *_a, **_kw: None)

    status = resolve_sandbox_status()
    assert status.available is False
    assert status.reason == SandboxUnavailableReason.MISSING_BACKEND


def test_resolve_unsupported_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_availability.platform, "system", lambda: "Windows")
    status = resolve_sandbox_status()
    assert status.available is False
    assert status.reason == SandboxUnavailableReason.UNSUPPORTED_PLATFORM
