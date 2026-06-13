from __future__ import annotations


class EngineError(Exception):
    """Base engine error."""


class EngineAgentExhaustedError(EngineError):
    """Raised when an agent hits the consecutive-LLM-failure circuit breaker."""


class EngineAgentRefusedError(EngineError):
    """Raised when an agent exhausts configured model-refusal retries."""


class EngineMaxDepthExceededError(EngineError):
    """Raised when subagent spawn attempted beyond maximum_depth."""


class EngineSandboxDeniedError(EngineError):
    """Raised when sandbox execution was blocked by policy."""


class EngineToolError(EngineError):
    """Raised from a tool adapter when returning a typed error to the caller."""


class EngineDependencyError(EngineError):
    """Raised when a required external dependency (e.g. ripgrep) is missing."""
