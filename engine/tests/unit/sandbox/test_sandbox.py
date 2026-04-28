from __future__ import annotations

from pathlib import Path

import pytest

from engine.sandbox import sandbox as sandbox_module
from engine.sandbox.models import CodeExecutionResult
from engine.sandbox.sandbox import Sandbox


def _fake_sandbox(tmp_path: Path) -> Sandbox:
    """Build a ``Sandbox`` with stub paths for argv-shape tests.

    Bypasses the discovery path: tests at this level care about
    ``run_python`` plumbing, not Deno detection. The actual subprocess
    is replaced by stubbing ``_run_session`` in the test bodies.
    """
    deno = tmp_path / "deno"
    deno.write_text("")
    runner = tmp_path / "runner.js"
    runner.write_text("// stub")
    runtime = tmp_path / "pyodide_runtime.py"
    runtime.write_text("# stub")
    trace_compat = tmp_path / "pyodide_trace_compat.py"
    trace_compat.write_text("# stub")
    deno_dir = tmp_path / "deno-cache"
    deno_dir.mkdir()
    return Sandbox(
        deno_executable=deno,
        runner_path=runner,
        runtime_path=runtime,
        trace_compat_path=trace_compat,
        deno_dir=deno_dir,
    )


# -- Sandbox.resolve -----------------------------------------------------------


def test_resolve_returns_none_when_deno_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Deno on PATH or via the PyPI dep → ``Sandbox.resolve`` returns ``None``.

    Discovery emits its own remediation warning before bailing out;
    ``Sandbox.resolve`` just propagates the ``None`` so callers can
    silently drop ``run_code`` from the agent surface.
    """
    monkeypatch.setattr(sandbox_module, "_locate_deno_executable", lambda: None)
    assert Sandbox.resolve() is None


def test_resolve_returns_none_when_required_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If runner.js or its sibling .py files vanish, resolve must refuse to fabricate a Sandbox."""
    deno = tmp_path / "deno"
    deno.write_text("")
    monkeypatch.setattr(sandbox_module, "_locate_deno_executable", lambda: deno)
    # Point ``__file__``-derived parent at an empty dir so the required
    # sibling files don't exist relative to it.
    monkeypatch.setattr(sandbox_module, "__file__", str(tmp_path / "sandbox.py"))
    assert Sandbox.resolve() is None


# -- Sandbox.run_python: argv shape -------------------------------------------


@pytest.mark.asyncio
async def test_run_python_includes_trace_and_index_in_allow_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_python`` adds the trace + index files to ``--allow-read``.

    Permissions are scoped per-call: the runner script and Deno cache
    are constants, but the trace/index files are caller-supplied. Both
    must appear in the resolved Deno argv so the runner can read them
    once at mount time.
    """
    sandbox = _fake_sandbox(tmp_path)
    trace = tmp_path / "t.jsonl"
    trace.write_text("")
    index = tmp_path / "i.jsonl"
    index.write_text("")

    captured: dict[str, list[str]] = {}

    async def _stub_run_session(*, argv, **_kwargs):
        captured["argv"] = list(argv)
        return CodeExecutionResult(exit_code=0, stdout="ok", stderr="", timed_out=False)

    monkeypatch.setattr(sandbox_module, "_run_session", _stub_run_session)

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
    sandbox = _fake_sandbox(tmp_path)
    trace = tmp_path / "t.jsonl"
    trace.write_text("")
    index = tmp_path / "i.jsonl"
    index.write_text("")

    captured: dict[str, list[str]] = {}

    async def _stub_run_session(*, argv, **_kwargs):
        captured["argv"] = list(argv)
        return CodeExecutionResult(exit_code=0, stdout="", stderr="", timed_out=False)

    monkeypatch.setattr(sandbox_module, "_run_session", _stub_run_session)

    await sandbox.run_python(code="x=1", trace_path=trace, index_path=index)
    forbidden = ("--allow-net", "--allow-write", "--allow-env", "--allow-run", "--allow-all")
    for flag in forbidden:
        assert not any(arg.startswith(flag) for arg in captured["argv"]), (
            f"sandbox argv must not contain {flag}: {captured['argv']}"
        )


@pytest.mark.asyncio
async def test_run_python_returns_runner_result_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A runner result with ``timed_out`` and a non-zero exit must round-trip unchanged."""
    sandbox = _fake_sandbox(tmp_path)
    trace = tmp_path / "t.jsonl"
    trace.write_text("")
    index = tmp_path / "i.jsonl"
    index.write_text("")

    expected = CodeExecutionResult(
        exit_code=137,
        stdout="partial",
        stderr="boom",
        timed_out=True,
    )

    async def _stub_run_session(**_kwargs):
        return expected

    monkeypatch.setattr(sandbox_module, "_run_session", _stub_run_session)

    result = await sandbox.run_python(code="x=1", trace_path=trace, index_path=index)
    assert result == expected
