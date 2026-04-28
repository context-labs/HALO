from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from engine.sandbox import linux_client
from engine.sandbox.linux_client import LinuxClient

# Stderr line bwrap writes when execvp fails with ENOENT. The probe relies
# on this string as the "namespace setup completed all the way to exec"
# signal, so the test fakes mimic real bwrap output.
_PROBE_EXEC_FAIL_STDERR = (
    b"bwrap: execvp /halo-sandbox-probe-no-such-exec-target: No such file or directory\n"
)


@pytest.fixture(autouse=True)
def _fake_unshare(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Pretend util-linux ``unshare`` exists so resolve() doesn't fail at module-level guard."""
    unshare = tmp_path_factory.mktemp("unshare-bin") / "unshare"
    unshare.write_text("")
    return unshare


@pytest.fixture(autouse=True)
def _patch_unshare(monkeypatch: pytest.MonkeyPatch, _fake_unshare: Path) -> None:
    monkeypatch.setattr(linux_client, "_UNSHARE", _fake_unshare)


def _install_fake_bwrap(
    monkeypatch: pytest.MonkeyPatch,
    *,
    info_fd_payload: bytes,
    stderr: bytes = _PROBE_EXEC_FAIL_STDERR,
    returncode: int = 1,
    captured_argv: list[str] | None = None,
) -> None:
    """Replace ``subprocess.Popen`` with a stub that mimics bwrap writing ``--info-fd``.

    Defaults match the *successful* probe case: bwrap writes ``child-pid`` to
    ``--info-fd``, sets up the namespace fully, then dies with exit 1 because
    ``execvp`` could not find our deliberately-nonexistent target. Override
    ``stderr`` and ``info_fd_payload`` to simulate failure modes.
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


def test_resolve_prefers_system_path_when_probe_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = tmp_path / "system-bwrap"
    packaged = tmp_path / "packaged-bwrap"
    system.write_text("")
    packaged.write_text("")

    monkeypatch.setattr(linux_client.shutil, "which", lambda *_a, **_kw: str(system))
    monkeypatch.setattr(linux_client, "_packaged_bwrap", lambda: packaged)
    _install_fake_bwrap(monkeypatch, info_fd_payload=b'{"child-pid":42}')

    client = LinuxClient.resolve()
    assert client is not None
    assert client.executable == system


def test_resolve_falls_back_to_packaged_when_system_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packaged = tmp_path / "packaged-bwrap"
    packaged.write_text("")

    monkeypatch.setattr(linux_client.shutil, "which", lambda *_a, **_kw: None)
    monkeypatch.setattr(linux_client, "_packaged_bwrap", lambda: packaged)
    _install_fake_bwrap(monkeypatch, info_fd_payload=b'{"child-pid":42}')

    client = LinuxClient.resolve()
    assert client is not None
    assert client.executable == packaged


def test_resolve_tries_packaged_when_system_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = tmp_path / "system-bwrap"
    packaged = tmp_path / "packaged-bwrap"
    system.write_text("")
    packaged.write_text("")

    monkeypatch.setattr(linux_client.shutil, "which", lambda *_a, **_kw: str(system))
    monkeypatch.setattr(linux_client, "_packaged_bwrap", lambda: packaged)

    def _fake_popen(argv, *args, **kwargs):
        info_fd = int(argv[argv.index("--info-fd") + 1])
        proc = MagicMock(spec=["wait", "kill", "returncode", "stderr"])
        proc.returncode = 1
        proc.wait.return_value = 1
        # ``argv`` is ``[unshare, ..., --, bwrap, ...]``; locate the bwrap
        # binary by skipping the leading unshare flags.
        bwrap_path = Path(argv[argv.index("--") + 1])
        if bwrap_path == system:
            proc.stderr.read.return_value = b"bwrap: setting up uid map: Operation not permitted\n"
        else:
            os.write(info_fd, b'{"child-pid":42}')
            proc.stderr.read.return_value = _PROBE_EXEC_FAIL_STDERR
        return proc

    monkeypatch.setattr(linux_client.subprocess, "Popen", _fake_popen)

    client = LinuxClient.resolve()

    assert client is not None
    assert client.executable == packaged


def test_resolve_returns_none_when_unshare_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(linux_client, "_UNSHARE", None)

    assert LinuxClient.resolve() is None
    err = capsys.readouterr().err
    assert "unshare" in err
    assert "install" in err.lower()


def test_resolve_returns_none_with_install_remediation_when_no_candidates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(linux_client.shutil, "which", lambda *_a, **_kw: None)
    monkeypatch.setattr(linux_client, "_packaged_bwrap", lambda: None)

    assert LinuxClient.resolve() is None
    err = capsys.readouterr().err
    assert "not found" in err
    assert "install" in err.lower()


def test_resolve_returns_none_with_namespace_remediation_when_probe_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Namespace setup itself is rejected: nothing is written to --info-fd, bwrap exits 1."""
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")

    monkeypatch.setattr(linux_client.shutil, "which", lambda *_a, **_kw: str(bwrap))
    monkeypatch.setattr(linux_client, "_packaged_bwrap", lambda: None)
    _install_fake_bwrap(
        monkeypatch,
        info_fd_payload=b"",
        stderr=b"bwrap: setting up uid map: Operation not permitted\n",
    )

    assert LinuxClient.resolve() is None
    err = capsys.readouterr().err
    assert "Operation not permitted" in err


def test_probe_argv_wraps_bwrap_with_outer_unshare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fake_unshare: Path
) -> None:
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")
    captured_argv: list[str] = []

    monkeypatch.setattr(linux_client.shutil, "which", lambda *_a, **_kw: str(bwrap))
    monkeypatch.setattr(linux_client, "_packaged_bwrap", lambda: None)
    _install_fake_bwrap(
        monkeypatch,
        info_fd_payload=b'{"child-pid":42}',
        captured_argv=captured_argv,
    )

    LinuxClient.resolve()

    info_fd = captured_argv[captured_argv.index("--info-fd") + 1]
    assert captured_argv == [
        str(_fake_unshare),
        "--user",
        "--map-current-user",
        "--net",
        "--",
        str(bwrap),
        "--unshare-user",
        "--unshare-pid",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--info-fd",
        info_fd,
        "--",
        "/halo-sandbox-probe-no-such-exec-target",
    ]


def test_build_argv_exact_shape(tmp_path: Path, _fake_unshare: Path) -> None:
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
        str(_fake_unshare),
        "--user",
        "--map-current-user",
        "--net",
        "--",
        str(bwrap),
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
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
    assert "--unshare-net" not in argv
