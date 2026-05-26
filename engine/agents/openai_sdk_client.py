"""Single boundary module for the OpenAI Agents SDK and the OpenAI client.

Every other engine module imports OpenAI / Agents SDK symbols from here, so
the vendor surface lives in exactly one place. To add a new SDK type, re-export
it from this module — do not import it directly elsewhere in ``engine/``.
"""

from __future__ import annotations

from agents import (
    Agent,
    FunctionTool,
    RunConfig,
    RunContextWrapper,
    Runner,
    Tool,
)
from agents.models.openai_provider import OpenAIProvider
from agents.tool_context import ToolContext as SdkToolContext
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
    omit,
)

from engine.model_provider_config import ModelProviderConfig

__all__ = [
    "Agent",
    "AsyncOpenAI",
    "FunctionTool",
    "OpenAIProvider",
    "RunConfig",
    "RunContextWrapper",
    "Runner",
    "SdkToolContext",
    "Tool",
    "build_async_openai_client",
    "is_retriable_llm_error",
    "omit",
]


def build_async_openai_client(provider_config: ModelProviderConfig) -> AsyncOpenAI:
    """Construct the per-run ``AsyncOpenAI`` client from connection settings.

    Each field is independent — when unset on ``provider_config`` the SDK falls
    back to the matching ``OPENAI_*`` env var.
    """
    return AsyncOpenAI(
        base_url=provider_config.base_url,
        api_key=provider_config.api_key,
        default_headers=provider_config.default_headers,
    )


def is_retriable_llm_error(exc: BaseException) -> bool:
    """Classify an exception as a transient LLM failure worth retrying."""
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    return False
