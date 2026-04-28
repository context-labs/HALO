"""HALO sandbox module: pick a per-platform client and run user Python under isolation.

Public surface:
    - ``Sandbox``: front-door class; one instance per run when sandboxing is available.
    - ``resolve_sandbox(config=...)``: probe the host; return ``Sandbox | None``.
    - ``SandboxConfig``: caller-tunable timeouts/output caps/python-executable override.
    - ``SandboxNotAvailable``: typed exception clients raise when the host can't sandbox.
    - ``CodeExecutionResult`` / ``RunCodeArguments``: run-time IO models for ``run_code``.
"""

from engine.sandbox.linux_client import LinuxClient, SandboxNotAvailable
from engine.sandbox.macos_client import MacosClient
from engine.sandbox.models import (
    CodeExecutionResult,
    PythonRuntimeMounts,
    RunCodeArguments,
    SandboxConfig,
)
from engine.sandbox.sandbox import Sandbox, resolve_sandbox

__all__ = [
    "CodeExecutionResult",
    "LinuxClient",
    "MacosClient",
    "PythonRuntimeMounts",
    "RunCodeArguments",
    "Sandbox",
    "SandboxConfig",
    "SandboxNotAvailable",
    "resolve_sandbox",
]
