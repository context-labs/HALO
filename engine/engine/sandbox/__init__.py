"""HALO sandbox module: run user Python under a Deno+Pyodide WASM sandbox.

Public surface:
    - ``Sandbox``: resolved sandbox; obtained via ``Sandbox.resolve()``.
    - ``CodeExecutionResult`` / ``RunCodeArguments``: tool IO models.
"""

from engine.sandbox.models import CodeExecutionResult, RunCodeArguments
from engine.sandbox.sandbox import Sandbox

__all__ = [
    "CodeExecutionResult",
    "RunCodeArguments",
    "Sandbox",
]
