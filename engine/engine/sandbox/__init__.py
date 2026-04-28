"""HALO sandbox module: run user Python under a Deno+Pyodide WASM sandbox.

Public surface:
    - ``Sandbox``: front-door class; one instance per probed host.
    - ``resolve_sandbox(config=...)``: probe the host; return ``Sandbox | None``.
    - ``SandboxConfig``: caller-tunable timeouts + output caps.
    - ``CodeExecutionResult`` / ``RunCodeArguments``: run-time IO models for ``run_code``.
"""

from engine.sandbox.models import (
    CodeExecutionResult,
    RunCodeArguments,
    SandboxConfig,
)
from engine.sandbox.pyodide_client import PyodideClient
from engine.sandbox.sandbox import Sandbox, resolve_sandbox

__all__ = [
    "CodeExecutionResult",
    "PyodideClient",
    "RunCodeArguments",
    "Sandbox",
    "SandboxConfig",
    "resolve_sandbox",
]
