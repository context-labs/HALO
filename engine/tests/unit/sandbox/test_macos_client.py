from __future__ import annotations

from pathlib import Path

import pytest

from engine.sandbox import macos_client
from engine.sandbox.macos_client import MacosClient


def test_resolve_finds_sandbox_exec_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        macos_client.shutil,
        "which",
        lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None,
    )

    client = MacosClient.resolve()
    assert client is not None
    assert client.executable == Path("/usr/bin/sandbox-exec")


def test_resolve_returns_none_when_sandbox_exec_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(macos_client.shutil, "which", lambda *_a, **_kw: None)

    assert MacosClient.resolve() is None
    err = capsys.readouterr().err
    assert "sandbox-exec not found on PATH" in err


def test_render_profile_exact_shape(tmp_path: Path) -> None:
    """Profile is default-deny + broad ``file-read-metadata`` for path
    traversal + full read on a fixed set of SIP-protected system roots and
    literals (mirrors Apple's ``system.sb``) + full read on each
    readonly path (subpath for dirs, literal for files) + read+write on
    each writable path. ``file-read-metadata`` does not include
    ``file-read-data``, so listing arbitrary host directories or reading
    arbitrary file contents remains denied."""
    trace = tmp_path / "t.jsonl"
    index = tmp_path / "t.idx.jsonl"
    runtime = tmp_path / "runtime"
    work = tmp_path / "work"
    trace.write_text("")
    index.write_text("")
    runtime.mkdir()
    work.mkdir()

    client = MacosClient(executable=Path("/usr/bin/sandbox-exec"))
    profile = client.render_profile(
        readonly_paths=[trace, index, runtime],
        writable_paths=[work],
    )

    expected_lines = [
        "(version 1)",
        "(deny default)",
        "(allow process*)",
        "(allow mach-lookup)",
        "(allow ipc-posix-shm)",
        "(allow sysctl-read)",
        "(allow signal)",
        "(allow file-read-metadata)",
        '(allow file-read* file-map-executable (subpath "/usr/lib"))',
        '(allow file-read* file-map-executable (subpath "/usr/share"))',
        '(allow file-read* file-map-executable (subpath "/System"))',
        '(allow file-read* file-map-executable (subpath "/Library/Apple"))',
        '(allow file-read* file-map-executable (subpath "/private/var/db/timezone"))',
        '(allow file-read* (literal "/"))',
        '(allow file-read* (literal "/dev/random"))',
        '(allow file-read* (literal "/dev/urandom"))',
        '(allow file-read* (literal "/dev/null"))',
        '(allow file-read* (literal "/private/etc/passwd"))',
        '(allow file-read* (literal "/private/etc/protocols"))',
        '(allow file-read* (literal "/private/etc/services"))',
        '(allow file-read* (literal "/private/etc/localtime"))',
        '(allow file-read* (subpath "/private/var/folders"))',
        f'(allow file-read* (literal "{trace}"))',
        f'(allow file-read* (literal "{index}"))',
        f'(allow file-read* (subpath "{runtime}"))',
        f'(allow file-read* (subpath "{work}"))',
        f'(allow file-write* (subpath "{work}"))',
        "(deny network*)",
    ]
    assert profile == "\n".join(expected_lines) + "\n"


def test_build_argv_exact_shape(tmp_path: Path) -> None:
    sandbox_exec = tmp_path / "bin" / "sandbox-exec"
    sandbox_exec.parent.mkdir()
    sandbox_exec.write_text("")
    python = tmp_path / "runtime" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    work = tmp_path / "work"
    work.mkdir()
    script = work / "bootstrap.py"
    script.write_text("")
    profile_path = tmp_path / "profile.sb"

    client = MacosClient(executable=sandbox_exec)
    argv = client.build_argv(
        python_executable=python,
        script_path=script,
        profile_path=profile_path,
        work_dir=work,
    )

    assert argv == [
        str(sandbox_exec),
        "-f",
        str(profile_path),
        "env",
        "-i",
        f"PATH={python.parent}:/usr/bin:/bin",
        f"HOME={work}",
        "LANG=C.UTF-8",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUNBUFFERED=1",
        f"TMPDIR={work / 'tmp'}",
        str(python),
        str(script),
    ]
