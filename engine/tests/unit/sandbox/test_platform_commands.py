from __future__ import annotations

from pathlib import Path

import pytest

from engine.sandbox import platform_commands
from engine.sandbox.platform_commands import (
    build_linux_bubblewrap_command,
    build_macos_sandbox_exec_command,
    render_macos_profile,
)
from engine.sandbox.sandbox_config import SandboxPolicy


def _policy(tmp_path: Path) -> SandboxPolicy:
    trace = tmp_path / "t.jsonl"
    index = tmp_path / "t.idx.jsonl"
    runtime = tmp_path / "runtime"
    work = tmp_path / "work"
    python = runtime / "bin" / "python"
    libc = tmp_path / "lib" / "libc.so.6"

    trace.write_text("")
    index.write_text("")
    runtime.mkdir()
    work.mkdir()
    python.parent.mkdir()
    python.write_text("")
    libc.parent.mkdir()
    libc.write_text("")

    return SandboxPolicy(
        python_executable=python,
        readonly_paths=[trace, index, runtime],
        library_paths=[libc],
        writable_paths=[work],
        timeout_seconds=10.0,
    )


def test_linux_command_exact_shape(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    script = tmp_path / "work" / "bootstrap.py"
    script.write_text("")
    bwrap = tmp_path / "bin" / "bwrap"
    bwrap.parent.mkdir()
    bwrap.write_text("")

    argv = build_linux_bubblewrap_command(
        bwrap_executable=bwrap,
        policy=policy,
        script_path=script,
    )

    trace = policy.readonly_paths[0]
    index = policy.readonly_paths[1]
    runtime = policy.readonly_paths[2]
    library = policy.library_paths[0]
    work = policy.writable_paths[0]
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
        str(runtime),
        str(runtime),
        "--ro-bind",
        str(library),
        str(library),
        "--bind",
        str(work),
        str(work),
        "--setenv",
        "PATH",
        f"{policy.python_executable.parent}:/usr/bin:/bin",
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
        f"{work}/tmp",
        "--chdir",
        str(work),
        "--",
        str(policy.python_executable),
        str(script),
    ]


def test_macos_profile_exact_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy(tmp_path)
    monkeypatch.setattr(platform_commands, "_macos_home_dir", lambda: "/Users/tester")

    profile = render_macos_profile(policy=policy)

    trace = policy.readonly_paths[0]
    index = policy.readonly_paths[1]
    runtime = policy.readonly_paths[2]
    work = policy.writable_paths[0]
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


def test_macos_command_exact_shape(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    script = tmp_path / "work" / "bootstrap.py"
    script.write_text("")
    profile_path = tmp_path / "profile.sb"
    sandbox_exec = tmp_path / "bin" / "sandbox-exec"
    sandbox_exec.parent.mkdir()
    sandbox_exec.write_text("")

    argv = build_macos_sandbox_exec_command(
        sandbox_exec_executable=sandbox_exec,
        policy=policy,
        script_path=script,
        profile_path=profile_path,
    )

    work = policy.writable_paths[0]
    assert argv == [
        str(sandbox_exec),
        "-f",
        str(profile_path),
        "env",
        "-i",
        f"PATH={policy.python_executable.parent}:/usr/bin:/bin",
        f"HOME={work}",
        "LANG=C.UTF-8",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUNBUFFERED=1",
        f"TMPDIR={work}/tmp",
        str(policy.python_executable),
        str(script),
    ]
