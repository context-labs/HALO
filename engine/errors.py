from __future__ import annotations


class EngineError(Exception):
    """Base engine error."""


class EngineAgentExhaustedError(EngineError):
    """Raised when an agent hits the consecutive-LLM-failure circuit breaker."""


class EngineMaxDepthExceededError(EngineError):
    """Raised when subagent spawn attempted beyond maximum_depth."""


class EngineSandboxDeniedError(EngineError):
    """Raised when sandbox execution was blocked by policy."""


class EngineToolError(EngineError):
    """Raised from a tool adapter when returning a typed error to the caller."""
