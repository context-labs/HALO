from __future__ import annotations

from pathlib import Path

import pytest

from engine.sandbox import sandbox as sandbox_module
from engine.sandbox.models import SandboxConfig
from engine.sandbox.pyodide_client import PyodideAssets, PyodideClient
from engine.sandbox.sandbox import Sandbox, resolve_sandbox


def _fake_assets(tmp_path: Path) -> PyodideAssets:
    runner = tmp_path / "runner.js"
    runner.write_text("// stub")
    deno_dir = tmp_path / "deno-cache"
    deno_dir.mkdir()
    pyodide_dir = deno_dir / "npm" / "pyodide" / "0.29.3"
    pyodide_dir.mkdir(parents=True)
    return PyodideAssets(
        runner_path=runner,
        deno_dir=deno_dir,
        pyodide_npm_dir=pyodide_dir,
    )


# -- resolve_sandbox -----------------------------------------------------------


def test_resolve_sandbox_returns_none_when_client_resolve_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``resolve_sandbox`` propagates ``None`` from the client without inspecting why.

    The Pyodide client emits its own remediation warning before returning
    ``None``. This test only checks that ``resolve_sandbox`` does not
    fabricate a ``Sandbox`` when the client bails out.
    """
    monkeypatch.setattr(PyodideClient, "resolve", staticmethod(lambda: None))
    assert resolve_sandbox(config=SandboxConfig()) is None


def test_resolve_sandbox_returns_sandbox_when_client_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deno = tmp_path / "deno"
    deno.write_text("")
    assets = _fake_assets(tmp_path)
    client = PyodideClient(deno_executable=deno, assets=assets)

    monkeypatch.setattr(PyodideClient, "resolve", staticmethod(lambda: client))

    sandbox = resolve_sandbox(config=SandboxConfig())

    assert sandbox is not None
    assert sandbox.client is client


def test_resolve_sandbox_returns_none_when_compat_shim_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """If the trace compat shim is missing from the package, sandboxing is unavailable.

    The shim is shipped as package data; an absent file means a broken
    install and we'd rather refuse to spin up a Sandbox that crashes on
    first use than silently misbehave.
    """
    deno = tmp_path / "deno"
    deno.write_text("")
    assets = _fake_assets(tmp_path)
    client = PyodideClient(deno_executable=deno, assets=assets)
    missing_path = tmp_path / "nope.py"

    monkeypatch.setattr(PyodideClient, "resolve", staticmethod(lambda: client))
    monkeypatch.setattr(sandbox_module, "_TRACE_COMPAT_HOST_PATH", missing_path)

    assert resolve_sandbox(config=SandboxConfig()) is None
    err = capsys.readouterr().err
    assert "trace compat shim missing" in err


# -- Sandbox.run_python: argv + mount routing ---------------------------------


@pytest.mark.asyncio
async def test_run_python_includes_trace_and_index_in_allow_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_python`` adds the trace + index files to ``--allow-read``.

    Permissions are scoped per-call: the runner script and Deno cache are
    constants, but the trace/index files are caller-supplied. Both must
    appear in the resolved Deno argv so the runner can read them once at
    mount time.
    """
    deno = tmp_path / "deno"
    deno.write_text("")
    assets = _fake_assets(tmp_path)
    client = PyodideClient(deno_executable=deno, assets=assets)
    sandbox = Sandbox(client=client, config=SandboxConfig(timeout_seconds=5.0))

    trace = tmp_path / "t.jsonl"
    trace.write_text("")
    index = tmp_path / "i.jsonl"
    index.write_text("")

    captured: dict[str, list[str]] = {}

    class _StubSession:
        def __init__(self, *, argv, **_kwargs):
            captured["argv"] = list(argv)

        async def run(self, **_kwargs):
            from engine.sandbox.pyodide_client import _ExecutionOutcome

            return _ExecutionOutcome(exit_code=0, stdout="ok", stderr="", timed_out=False)

    monkeypatch.setattr(sandbox_module, "_RunnerSession", _StubSession)

    result = await sandbox.run_python(code="x=1", trace_path=trace, index_path=index)
    assert result.exit_code == 0
    assert result.stdout == "ok"

    allow_read_arg = next(a for a in captured["argv"] if a.startswith("--allow-read="))
    allow = allow_read_arg.split("=", 1)[1].split(",")
    assert str(trace.resolve()) in allow
    assert str(index.resolve()) in allow


@pytest.mark.asyncio
async def test_run_python_does_not_pass_unsafe_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Deno argv must never carry ``--allow-net``, ``--allow-write``, ``--allow-env``, or ``--allow-run``.

    These flags would lift exactly the constraints HALO is enforcing
    (network, host writes, host env vars, subprocess spawn). A regression
    that adds any of them silently weakens the sandbox.
    """
    deno = tmp_path / "deno"
    deno.write_text("")
    assets = _fake_assets(tmp_path)
    client = PyodideClient(deno_executable=deno, assets=assets)
    sandbox = Sandbox(client=client, config=SandboxConfig(timeout_seconds=5.0))

    trace = tmp_path / "t.jsonl"
    trace.write_text("")
    index = tmp_path / "i.jsonl"
    index.write_text("")

    captured: dict[str, list[str]] = {}

    class _StubSession:
        def __init__(self, *, argv, **_kwargs):
            captured["argv"] = list(argv)

        async def run(self, **_kwargs):
            from engine.sandbox.pyodide_client import _ExecutionOutcome

            return _ExecutionOutcome(exit_code=0, stdout="", stderr="", timed_out=False)

    monkeypatch.setattr(sandbox_module, "_RunnerSession", _StubSession)

    await sandbox.run_python(code="x=1", trace_path=trace, index_path=index)
    forbidden = ("--allow-net", "--allow-write", "--allow-env", "--allow-run", "--allow-all")
    for flag in forbidden:
        assert not any(arg.startswith(flag) for arg in captured["argv"]), (
            f"sandbox argv must not contain {flag}: {captured['argv']}"
        )


@pytest.mark.asyncio
async def test_run_python_translates_runner_outcome_to_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A runner outcome with ``timed_out`` and a non-zero exit must round-trip into ``CodeExecutionResult``."""
    deno = tmp_path / "deno"
    deno.write_text("")
    assets = _fake_assets(tmp_path)
    client = PyodideClient(deno_executable=deno, assets=assets)
    sandbox = Sandbox(client=client, config=SandboxConfig(timeout_seconds=5.0))

    trace = tmp_path / "t.jsonl"
    trace.write_text("")
    index = tmp_path / "i.jsonl"
    index.write_text("")

    class _StubSession:
        def __init__(self, **_kwargs):
            pass

        async def run(self, **_kwargs):
            from engine.sandbox.pyodide_client import _ExecutionOutcome

            return _ExecutionOutcome(
                exit_code=137,
                stdout="partial",
                stderr="boom",
                timed_out=True,
            )

    monkeypatch.setattr(sandbox_module, "_RunnerSession", _StubSession)

    result = await sandbox.run_python(code="x=1", trace_path=trace, index_path=index)
    assert result.exit_code == 137
    assert result.stdout == "partial"
    assert result.stderr == "boom"
    assert result.timed_out is True
