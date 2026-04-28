from __future__ import annotations

from pathlib import Path

from engine.sandbox.models import CodeExecutionResult
from engine.sandbox.pyodide_client import (
    PyodideClient,
    PyodideError,
    _ExecutionOutcome,
    _RunnerSession,
)

# Where the trace and index files live inside the Pyodide FS. Hardcoded so
# the sandbox and the in-Pyodide ``halo_bootstrap`` stay aligned without
# leaking host paths through the WASM filesystem.
_TRACE_VIRTUAL_PATH = "/input/traces.jsonl"
_INDEX_VIRTUAL_PATH = "/input/index.jsonl"

# Defensive caps on captured output. Constants rather than per-call config:
# the agent should not be able to provoke arbitrarily large prompt growth by
# emitting a multi-megabyte stdout from inside the sandbox, and there's no
# realistic use case for a caller wanting to *raise* the cap (any analysis
# that needs more than this should be summarizing in code, not in stdout).
_MAX_STDOUT_BYTES = 64_000
_MAX_STDERR_BYTES = 64_000


class Sandbox:
    """Single front door for running Python under the Deno+Pyodide WASM sandbox.

    A ``Sandbox`` always represents a working backend: ``resolve_sandbox()``
    is the only blessed factory and returns ``None`` when the host cannot
    provide one. Holds the resolved client and the per-run timeout; each
    ``run_python`` call spawns a fresh subprocess so WASM filesystem state
    does not leak between runs.
    """

    def __init__(self, *, client: PyodideClient, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds!r}")
        self._client = client
        self._timeout_seconds = timeout_seconds

    @property
    def client(self) -> PyodideClient:
        """Return the underlying Pyodide client (read-only test seam)."""
        return self._client

    @property
    def timeout_seconds(self) -> float:
        """Wall-clock budget for one ``run_python`` call."""
        return self._timeout_seconds

    async def run_python(
        self,
        *,
        code: str,
        trace_path: Path,
        index_path: Path,
    ) -> CodeExecutionResult:
        """Run ``code`` in the WASM sandbox; returns a typed result regardless of pass/fail/timeout.

        Mounting policy:
          - The runner.js + its sibling Python files + the Deno cache are
            read-only (covered by the client's ``--allow-read`` set).
          - The host trace + index are added to ``--allow-read`` for this
            invocation only, so Deno can read them once to copy bytes into
            Pyodide's virtual FS.
          - Inside Pyodide, files are visible at fixed virtual paths. The
            runner stages the trace compat shim itself; the host only
            tells the bootstrap which mount points to load.
        """
        argv = self._client.build_argv(
            extra_read_paths=[trace_path.resolve(), index_path.resolve()],
        )
        session = _RunnerSession(
            argv=argv,
            timeout_seconds=self._timeout_seconds,
            max_stdout=_MAX_STDOUT_BYTES,
            max_stderr=_MAX_STDERR_BYTES,
        )
        try:
            outcome = await session.run(
                mounts=[
                    (trace_path.resolve(), _TRACE_VIRTUAL_PATH),
                    (index_path.resolve(), _INDEX_VIRTUAL_PATH),
                ],
                bootstrap_paths=(_TRACE_VIRTUAL_PATH, _INDEX_VIRTUAL_PATH),
                user_code=code,
            )
        except PyodideError as exc:
            return CodeExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"sandbox runner failure: {exc}",
                timed_out=False,
            )

        return _outcome_to_result(outcome)


def resolve_sandbox(*, timeout_seconds: float = 10.0) -> Sandbox | None:
    """Probe the host once; return a ready ``Sandbox`` or ``None``.

    ``timeout_seconds`` is the only per-run knob — output caps are fixed
    module constants and Deno discovery is fully encapsulated by
    ``PyodideClient.resolve()``. The client logs its own unavailability
    warning before returning ``None``; this function just propagates that
    ``None`` back without inspecting why.
    """
    client = PyodideClient.resolve()
    if client is None:
        return None
    return Sandbox(client=client, timeout_seconds=timeout_seconds)


def _outcome_to_result(outcome: _ExecutionOutcome) -> CodeExecutionResult:
    """Translate the runner's outcome (incl. captured stderr noise) into the public result model."""
    stderr = outcome.stderr
    if outcome.stderr_extra:
        extra = outcome.stderr_extra.decode("utf-8", errors="replace").strip()
        if extra:
            sep = "\n" if stderr else ""
            stderr = stderr + sep + f"[deno stderr] {extra}"
    return CodeExecutionResult(
        exit_code=outcome.exit_code,
        stdout=outcome.stdout,
        stderr=stderr,
        timed_out=outcome.timed_out,
    )


__all__ = [
    "Sandbox",
    "resolve_sandbox",
]
