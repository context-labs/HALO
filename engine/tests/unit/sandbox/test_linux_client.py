from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from engine.sandbox import linux_client
from engine.sandbox.linux_client import HALO_BWRAP_ENV_VAR, LinuxClient


def _install_fake_bwrap(
    monkeypatch: pytest.MonkeyPatch,
    *,
    info_fd_payload: bytes,
    stderr: bytes = b"",
    returncode: int = 127,
    captured_argv: list[str] | None = None,
) -> None:
    """Replace ``subprocess.Popen`` with a stub that mimics bwrap writing ``--info-fd``.

    ``returncode`` defaults to 127 (the value bwrap returns when execvp hits
    ENOENT for our nonexistent probe target — i.e. the *successful* probe
    case where namespace setup completed). Set 1 to simulate a bwrap-internal
    failure that occurred after info-fd was written (for example loopback
    configuration blocked by Ubuntu 24.04 AppArmor).
    """

    def _fake_popen(argv, *args, **kwargs):
        if captured_argv is not None:
            captured_argv[:] = list(argv)
        info_fd = int(argv[argv.index("--info-fd") + 1])
        if info_fd_payload:
            os.write(info_fd, info_fd_payload)

        proc = MagicMock(spec=["wait", "kill", "returncode", "stderr"])
        proc.returncode = returncode
        proc.wait.return_value = returncode
        proc.stderr.read.return_value = stderr
        return proc

    monkeypatch.setattr(linux_client.subprocess, "Popen", _fake_popen)


def test_resolve_uses_env_override_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_bwrap = tmp_path / "env-bwrap"
    env_bwrap.write_text("")

    monkeypatch.setenv(HALO_BWRAP_ENV_VAR, str(env_bwrap))
    monkeypatch.setattr(linux_client.shutil, "which", lambda *_a, **_kw: "/host/bwrap")
    monkeypatch.setattr(linux_client, "_packaged_bwrap", lambda: None)
    _install_fake_bwrap(monkeypatch, info_fd_payload=b'{"child-pid":42}')

    client = LinuxClient.resolve()
    assert client is not None
    assert client.executable == env_bwrap


def test_resolve_falls_back_to_packaged_when_system_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packaged = tmp_path / "packaged-bwrap"
    packaged.write_text("")

    monkeypatch.delenv(HALO_BWRAP_ENV_VAR, raising=False)
    monkeypatch.setattr(linux_client.shutil, "which", lambda *_a, **_kw: None)
    monkeypatch.setattr(linux_client, "_packaged_bwrap", lambda: packaged)
    _install_fake_bwrap(monkeypatch, info_fd_payload=b'{"child-pid":42}')

    client = LinuxClient.resolve()
    assert client is not None
    assert client.executable == packaged


def test_resolve_returns_none_with_install_remediation_when_no_candidates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(HALO_BWRAP_ENV_VAR, raising=False)
    monkeypatch.setattr(linux_client.shutil, "which", lambda *_a, **_kw: None)
    monkeypatch.setattr(linux_client, "_packaged_bwrap", lambda: None)

    assert LinuxClient.resolve() is None
    err = capsys.readouterr().err
    assert "not found" in err
    assert "install" in err.lower()


def test_resolve_returns_none_with_install_remediation_when_only_candidate_is_nonexistent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    nonexistent = tmp_path / "does-not-exist"

    monkeypatch.setenv(HALO_BWRAP_ENV_VAR, str(nonexistent))
    monkeypatch.setattr(linux_client.shutil, "which", lambda *_a, **_kw: None)
    monkeypatch.setattr(linux_client, "_packaged_bwrap", lambda: None)

    assert LinuxClient.resolve() is None
    err = capsys.readouterr().err
    assert str(nonexistent) in err
    assert "install" in err.lower()


def test_resolve_returns_none_with_namespace_remediation_when_probe_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Namespace setup itself is rejected: nothing is written to --info-fd, bwrap exits 1."""
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")

    monkeypatch.delenv(HALO_BWRAP_ENV_VAR, raising=False)
    monkeypatch.setattr(linux_client.shutil, "which", lambda *_a, **_kw: str(bwrap))
    monkeypatch.setattr(linux_client, "_packaged_bwrap", lambda: None)
    _install_fake_bwrap(
        monkeypatch,
        info_fd_payload=b"",
        stderr=b"bwrap: setting up uid map: Operation not permitted\n",
        returncode=1,
    )

    assert LinuxClient.resolve() is None
    err = capsys.readouterr().err
    assert "Operation not permitted" in err
    assert "user namespaces" in err


def test_resolve_returns_none_when_bwrap_dies_after_info_fd_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """bwrap can write child-pid to --info-fd and *then* die during loopback
    setup (Ubuntu 24.04 AppArmor blocks ``RTM_NEWADDR`` in unprivileged
    user namespaces). The probe must reject this case rather than treat
    the info-fd marker alone as success — actual ``run_python`` calls would
    fail the same way and the model would see a confusing error."""
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")

    monkeypatch.delenv(HALO_BWRAP_ENV_VAR, raising=False)
    monkeypatch.setattr(linux_client.shutil, "which", lambda *_a, **_kw: str(bwrap))
    monkeypatch.setattr(linux_client, "_packaged_bwrap", lambda: None)
    _install_fake_bwrap(
        monkeypatch,
        info_fd_payload=b'{"child-pid":42}',
        stderr=b"bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\n",
        returncode=1,
    )

    assert LinuxClient.resolve() is None
    err = capsys.readouterr().err
    assert "loopback" in err
    assert "Operation not permitted" in err
    assert "AppArmor" in err


def test_probe_argv_is_explicit_and_uses_empty_rootfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")
    captured_argv: list[str] = []

    monkeypatch.delenv(HALO_BWRAP_ENV_VAR, raising=False)
    monkeypatch.setattr(linux_client.shutil, "which", lambda *_a, **_kw: str(bwrap))
    monkeypatch.setattr(linux_client, "_packaged_bwrap", lambda: None)
    _install_fake_bwrap(
        monkeypatch,
        info_fd_payload=b'{"child-pid":42}',
        captured_argv=captured_argv,
    )

    LinuxClient.resolve()

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
        "/halo-sandbox-probe-no-such-exec-target",
    ]


def test_build_argv_exact_shape(tmp_path: Path) -> None:
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")
    python = tmp_path / "runtime" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    work = tmp_path / "work"
    work.mkdir()
    script = work / "bootstrap.py"
    script.write_text("")
    trace = tmp_path / "t.jsonl"
    trace.write_text("")
    index = tmp_path / "t.idx.jsonl"
    index.write_text("")
    runtime_dir = tmp_path / "runtime"
    libc = tmp_path / "lib" / "libc.so.6"
    libc.parent.mkdir()
    libc.write_text("")

    client = LinuxClient(executable=bwrap)
    argv = client.build_argv(
        python_executable=python,
        script_path=script,
        work_dir=work,
        readonly_paths=[trace, index, runtime_dir],
        library_paths=[libc],
    )

    assert argv == [
        str(bwrap),
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--unshare-net",
        "--clearenv",
        "--dev",
        "/dev",
        "--ro-bind",
        str(trace),
        str(trace),
        "--ro-bind",
        str(index),
        str(index),
        "--ro-bind",
        str(runtime_dir),
        str(runtime_dir),
        "--ro-bind",
        str(libc),
        str(libc),
        "--bind",
        str(work),
        str(work),
        "--setenv",
        "PATH",
        f"{python.parent}:/usr/bin:/bin",
        "--setenv",
        "HOME",
        str(work),
        "--setenv",
        "LANG",
        "C.UTF-8",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "PYTHONUNBUFFERED",
        "1",
        "--setenv",
        "TMPDIR",
        str(work / "tmp"),
        "--chdir",
        str(work),
        "--",
        str(python),
        str(script),
    ]
