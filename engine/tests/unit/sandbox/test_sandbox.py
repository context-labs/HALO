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


# -- _truncate: byte-aware caps -----------------------------------------------


def test_truncate_below_cap_passthrough() -> None:
    """A string under the cap must round-trip unchanged."""
    assert sandbox_module._truncate("hello", 100) == "hello"


def test_truncate_ascii_above_cap_emits_marker() -> None:
    """ASCII content above the cap is replaced past the head with the truncation marker."""
    text = "x" * 200
    result = sandbox_module._truncate(text, 100)
    assert len(result.encode("utf-8")) <= 100
    assert sandbox_module._TRUNCATION_MARKER in result
    assert result.startswith("x")


def test_truncate_multibyte_respects_byte_cap_not_char_cap() -> None:
    """Cap is named ``_MAX_STDOUT_BYTES`` so it must be enforced in bytes.

    Regression: ``_truncate`` previously sliced on ``len(text)`` (character
    count). 200 emoji are 200 chars but 800 UTF-8 bytes; under the old
    code a 100-byte cap would let through ~800 bytes — silently breaking
    the byte-named contract. The fix encodes first, slices on bytes,
    then decodes with ``errors="ignore"`` so a partial UTF-8 sequence at
    the cut never becomes a U+FFFD replacement character.
    """
    # 4-byte emoji × 200 = 800 UTF-8 bytes.
    text = "🔥" * 200
    cap_bytes = 100
    result = sandbox_module._truncate(text, cap_bytes)
    assert len(result.encode("utf-8")) <= cap_bytes, (
        f"truncated output is {len(result.encode('utf-8'))} bytes, exceeds cap {cap_bytes}"
    )
    assert sandbox_module._TRUNCATION_MARKER in result
    # No replacement character: the partial 🔥 bytes at the cut were
    # dropped cleanly by ``errors='ignore'``.
    assert "�" not in result


def test_truncate_drops_partial_utf8_at_cut() -> None:
    """When the byte cap lands mid-character, the partial sequence is dropped, not replaced."""
    # ``é`` is 2 UTF-8 bytes. With a head budget of 5 bytes (= 2 full
    # ``é`` + 1 partial byte), the third ``é`` is half-cut and must be
    # dropped via ``errors="ignore"``. Input length must exceed the cap
    # to actually trigger truncation, hence ``é * 100`` not ``ééé``.
    text = "é" * 100
    cap_bytes = len(sandbox_module._TRUNCATION_MARKER) + 5
    result = sandbox_module._truncate(text, cap_bytes)
    assert len(result.encode("utf-8")) <= cap_bytes
    assert "�" not in result
    assert result.endswith(sandbox_module._TRUNCATION_MARKER)
    # Head holds exactly the two full ``é`` that fit in 5 bytes.
    assert result.startswith("éé")
    assert not result.startswith("ééé")


# -- mount_file error surface --------------------------------------------------


@pytest.mark.asyncio
async def test_run_python_surfaces_mount_file_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``mount_file`` JSON-RPC error must surface immediately, not be swallowed.

    Regression: ``_drive`` discarded the result of every ``mount_file``
    call. A failed mount (host file missing, Deno --allow-read denial)
    would let bootstrap run anyway, where the absent file produced a
    confusing ``FileNotFoundError`` deep inside Pyodide. The fix returns
    early with a clear ``mount_file`` error.
    """
    sandbox = _fake_sandbox(tmp_path)
    trace = tmp_path / "t.jsonl"
    trace.write_text("")
    index = tmp_path / "i.jsonl"
    index.write_text("")

    captured: dict[str, list[str]] = {"phases": []}

    async def _stub_call(_proc, method, _params, _request_id):
        captured["phases"].append(method)
        if method == "mount_file":
            return sandbox_module._RpcResult(
                result=None,
                error={"code": -32008, "message": "Failed to mount file: missing"},
            )
        # If the driver kept going past the mount error, bootstrap would
        # be the next call — fail loud here so the test can assert that
        # the early return prevented it.
        raise AssertionError(
            f"driver should have returned after mount_file error; reached {method!r}"
        )

    async def _stub_read_ready(_stdout):
        return None

    async def _stub_send(_proc, _payload):
        return None

    class _StubProc:
        stdin = object()
        stdout = object()
        stderr = None
        pid = 0
        returncode = 0

        async def wait(self):
            return 0

    async def _stub_create_subprocess(*_args, **_kwargs):
        return _StubProc()

    async def _stub_drain_capped(_stream, _cap):
        return b""

    monkeypatch.setattr(sandbox_module, "_call", _stub_call)
    monkeypatch.setattr(sandbox_module, "_read_ready", _stub_read_ready)
    monkeypatch.setattr(sandbox_module, "_send", _stub_send)
    monkeypatch.setattr(sandbox_module, "_drain_capped", _stub_drain_capped)
    monkeypatch.setattr(sandbox_module.asyncio, "create_subprocess_exec", _stub_create_subprocess)

    result = await sandbox.run_python(code="x=1", trace_path=trace, index_path=index)
    assert result.exit_code == 1
    assert result.timed_out is False
    assert "mount_file" in result.stderr
    assert "Failed to mount file" in result.stderr
    # Driver stopped at the failed mount; bootstrap/execute never ran.
    assert captured["phases"] == ["mount_file"]
