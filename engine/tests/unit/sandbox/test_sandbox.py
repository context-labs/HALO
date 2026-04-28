from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from engine.sandbox import sandbox as sandbox_module
from engine.sandbox.linux_client import LinuxClient, SandboxNotAvailable
from engine.sandbox.macos_client import MacosClient
from engine.sandbox.models import PythonRuntimeMounts, SandboxConfig
from engine.sandbox.sandbox import Sandbox, resolve_sandbox


def _runtime_mounts(tmp_path: Path) -> PythonRuntimeMounts:
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.write_text("")
    return PythonRuntimeMounts(
        python_executable=python,
        runtime_paths=(),
        library_paths=(),
    )


# -- resolve_sandbox -----------------------------------------------------------


def test_resolve_sandbox_returns_none_on_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sandbox_module.platform, "system", lambda: "Windows")

    sandbox = resolve_sandbox(config=SandboxConfig())

    assert sandbox is None
    err = capsys.readouterr().err
    assert "unsupported platform: Windows" in err
    assert "How to fix:" in err


def test_resolve_sandbox_returns_none_when_client_resolve_raises(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sandbox_module.platform, "system", lambda: "Linux")

    def _raise() -> LinuxClient:
        raise SandboxNotAvailable(
            diagnostic="bwrap missing",
            remediation="install bubblewrap",
        )

    monkeypatch.setattr(LinuxClient, "resolve", staticmethod(_raise))

    sandbox = resolve_sandbox(config=SandboxConfig())

    assert sandbox is None
    err = capsys.readouterr().err
    assert "bwrap missing" in err
    assert "install bubblewrap" in err


def test_resolve_sandbox_returns_sandbox_when_client_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")
    fake_runtime = _runtime_mounts(tmp_path)

    monkeypatch.setattr(sandbox_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(LinuxClient, "resolve", staticmethod(lambda: LinuxClient(executable=bwrap)))
    monkeypatch.setattr(
        sandbox_module, "discover_python_runtime_mounts", lambda **_kwargs: fake_runtime
    )

    sandbox = resolve_sandbox(config=SandboxConfig())

    assert sandbox is not None
    assert isinstance(sandbox.client, LinuxClient)
    assert sandbox.client.executable == bwrap


def test_resolve_sandbox_passes_python_executable_to_runtime_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("")
    override = tmp_path / "bin" / "custompython"
    override.parent.mkdir()
    override.write_text("")

    captured: dict[str, Path | None] = {}

    def _fake_discover(*, python_executable: Path | None) -> PythonRuntimeMounts:
        captured["python_executable"] = python_executable
        return PythonRuntimeMounts(
            python_executable=override,
            runtime_paths=(),
            library_paths=(),
        )

    monkeypatch.setattr(sandbox_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(LinuxClient, "resolve", staticmethod(lambda: LinuxClient(executable=bwrap)))
    monkeypatch.setattr(sandbox_module, "discover_python_runtime_mounts", _fake_discover)

    resolve_sandbox(config=SandboxConfig(python_executable=override))

    assert captured["python_executable"] == override


# -- Sandbox.run_python: argv routing -----------------------------------------


@pytest.mark.asyncio
async def test_run_python_routes_to_linux_client_build_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_python`` must call into ``LinuxClient.build_argv`` (not macOS) when client is Linux."""
    runtime = _runtime_mounts(tmp_path)
    client = LinuxClient(executable=tmp_path / "bwrap")
    sandbox = Sandbox(client=client, runtime_mounts=runtime, config=SandboxConfig())

    captured: dict[str, list[str]] = {}

    def _fake_build_argv(self, **kwargs):
        captured["readonly_paths"] = [str(p) for p in kwargs["readonly_paths"]]
        return ["/bin/true"]

    async def _stub_run_capped(self, *, argv):
        return MagicMock(spec=["exit_code"], exit_code=0)

    monkeypatch.setattr(LinuxClient, "build_argv", _fake_build_argv)
    monkeypatch.setattr(Sandbox, "_run_capped", _stub_run_capped)

    trace = tmp_path / "t.jsonl"
    trace.write_text("")
    index = tmp_path / "i.jsonl"
    index.write_text("")

    await sandbox.run_python(code="x=1", trace_path=trace, index_path=index)

    assert str(trace) in captured["readonly_paths"]
    assert str(index) in captured["readonly_paths"]


@pytest.mark.asyncio
async def test_run_python_routes_to_macos_client_with_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_python`` must render a profile, write it to disk, and pass the path to ``MacosClient.build_argv``."""
    runtime = _runtime_mounts(tmp_path)
    client = MacosClient(executable=tmp_path / "sandbox-exec")
    sandbox = Sandbox(client=client, runtime_mounts=runtime, config=SandboxConfig())

    captured: dict[str, str] = {}

    def _fake_render_profile(self, *, readonly_paths, writable_paths):
        return "(version 1)\n;; rendered for test"

    def _fake_build_argv(self, *, python_executable, script_path, profile_path, work_dir):
        # Read the file inside the sandbox's temp dir before run_python's
        # cleanup deletes it.
        captured["profile_filename"] = profile_path.name
        captured["profile_contents"] = profile_path.read_text()
        return ["/bin/true"]

    async def _stub_run_capped(self, *, argv):
        return MagicMock(spec=["exit_code"], exit_code=0)

    monkeypatch.setattr(MacosClient, "render_profile", _fake_render_profile)
    monkeypatch.setattr(MacosClient, "build_argv", _fake_build_argv)
    monkeypatch.setattr(Sandbox, "_run_capped", _stub_run_capped)

    trace = tmp_path / "t.jsonl"
    trace.write_text("")
    index = tmp_path / "i.jsonl"
    index.write_text("")

    await sandbox.run_python(code="x=1", trace_path=trace, index_path=index)

    assert captured["profile_filename"] == "profile.sb"
    assert captured["profile_contents"] == "(version 1)\n;; rendered for test"


# -- _run_capped: process spawning + caps + timeout ---------------------------


@pytest.mark.asyncio
async def test_run_capped_simple_stdout_capture(tmp_path: Path) -> None:
    """``_run_capped`` returns stdout up to the cap when the process exits cleanly."""
    sandbox = Sandbox(
        client=LinuxClient(executable=tmp_path / "bwrap"),
        runtime_mounts=_runtime_mounts(tmp_path),
        config=SandboxConfig(timeout_seconds=5.0),
    )
    result = await sandbox._run_capped(argv=["/bin/sh", "-c", "printf hello"])
    assert result.exit_code == 0
    assert result.stdout == "hello"
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_run_capped_stdout_cap_truncates(tmp_path: Path) -> None:
    sandbox = Sandbox(
        client=LinuxClient(executable=tmp_path / "bwrap"),
        runtime_mounts=_runtime_mounts(tmp_path),
        config=SandboxConfig(timeout_seconds=5.0, maximum_stdout_bytes=64),
    )
    result = await sandbox._run_capped(
        argv=["/bin/sh", "-c", "for i in $(seq 1 1000); do echo aaaaaaaaaa; done"]
    )
    assert result.exit_code == 0
    assert "[... output truncated ...]" in result.stdout
    assert len(result.stdout.encode()) <= 64


@pytest.mark.asyncio
async def test_run_capped_under_cap_has_no_truncation_marker(tmp_path: Path) -> None:
    sandbox = Sandbox(
        client=LinuxClient(executable=tmp_path / "bwrap"),
        runtime_mounts=_runtime_mounts(tmp_path),
        config=SandboxConfig(timeout_seconds=5.0, maximum_stdout_bytes=10_000),
    )
    result = await sandbox._run_capped(argv=["/bin/sh", "-c", "echo ok"])
    assert "[... output truncated ...]" not in result.stdout
    assert "ok" in result.stdout


@pytest.mark.asyncio
async def test_run_capped_timeout_kills_and_reports(tmp_path: Path) -> None:
    sandbox = Sandbox(
        client=LinuxClient(executable=tmp_path / "bwrap"),
        runtime_mounts=_runtime_mounts(tmp_path),
        config=SandboxConfig(timeout_seconds=0.5),
    )
    result = await sandbox._run_capped(argv=["/bin/sh", "-c", "sleep 5"])
    assert result.timed_out is True
    assert result.exit_code != 0
