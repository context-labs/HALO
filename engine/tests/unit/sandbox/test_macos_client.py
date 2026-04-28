from __future__ import annotations

from pathlib import Path

import pytest

from engine.sandbox import macos_client
from engine.sandbox.linux_client import SandboxNotAvailable
from engine.sandbox.macos_client import MacosClient


def test_resolve_finds_sandbox_exec_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        macos_client.shutil,
        "which",
        lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None,
    )

    client = MacosClient.resolve()
    assert client.executable == Path("/usr/bin/sandbox-exec")


def test_resolve_raises_when_sandbox_exec_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(macos_client.shutil, "which", lambda *_a, **_kw: None)

    with pytest.raises(SandboxNotAvailable) as exc_info:
        MacosClient.resolve()
    assert "sandbox-exec" in exc_info.value.diagnostic


def test_render_profile_exact_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(macos_client, "_home_dir", lambda: "/Users/tester")

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

    assert (
        profile
        == f"""(version 1)
(deny default)
(allow process*)
(allow mach-lookup)
(allow ipc-posix-shm)
(allow sysctl-read)
(allow signal)
(allow file-read*)
(deny file-read* (subpath "/Users/tester"))
(allow file-read* (literal "{trace}"))
(allow file-read* (literal "{index}"))
(allow file-read* (subpath "{runtime}"))
(allow file-write* (subpath "{work}"))
(deny network*)
"""
    )


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
