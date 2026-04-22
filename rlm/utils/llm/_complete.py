"""Main entry point for the utils LLM abstraction."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import openai
from openai.types.chat import ChatCompletionMessageParam

from . import _providers

__all__ = ["CompletionResult", "complete"]

logger = logging.getLogger(__name__)

_PER_REQUEST_ERRORS = (openai.BadRequestError, openai.UnprocessableEntityError)


@dataclass
class CompletionResult:
    """Result of a single LLM completion call.

    Attributes:
        content: Text output from the model, or None if the model returned tool calls only.
        tool_calls: List of tool call dicts (OpenAI format) if the model invoked tools.
        message: Full assistant message dict in OpenAI format, suitable for appending
            back to the conversation history in agent loops.
        tokens: Token usage breakdown: ``{"input", "output", "thinking", "total"}``.
        cost: Dollar cost from the LiteLLM proxy header. Always 0.0 for local models.
        latency: Wall-clock seconds for the API call.
        model: Model identifier that was requested.
        error: Per-request error message (e.g. content filter, context length exceeded),
            or None on success.
    """

    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    message: dict[str, Any] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    cost: float = 0.0
    latency: float = 0.0
    model: str = ""
    error: str | None = None


def complete(
    model: str,
    messages: list[ChatCompletionMessageParam],
    *,
    local: bool = False,
    endpoints: list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    max_retries: int = 3,
    on_complete: Callable[[CompletionResult], None] | None = None,
) -> CompletionResult:
    """Send a chat completion request and return a structured result.

    Routes to either the LiteLLM proxy (cloud models) or a direct OpenAI-compatible
    client (local models) depending on the ``local`` flag.

    Per-request errors (400 Bad Request, 422 Unprocessable Entity) are captured in
    ``result.error`` so batch runs can skip and continue. Infrastructure errors
    (auth, not found, permission denied) raise immediately.

    Args:
        model: Model identifier (e.g. "gpt-5.4", "claude-sonnet-4-20250514").
        messages: Conversation history in OpenAI message format.
        local: If True, route to local endpoints instead of the LiteLLM proxy.
        endpoints: List of base URLs for local model servers (required when local=True).
            Requests are round-robined across endpoints.
        tools: Tool definitions in OpenAI function-calling format.
        tool_choice: Tool selection strategy (e.g. "auto", "required", or a specific tool name).
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens in the completion response.
        response_format: Response format constraint (e.g. {"type": "json_object"}).
        max_retries: Number of retry attempts for local model calls.
        on_complete: Optional callback invoked with the CompletionResult after success.

    Returns:
        CompletionResult with content, tool_calls, token counts, cost, latency,
        and error information.

    Raises:
        ValueError: If local=True but no endpoints are provided.
        openai.AuthenticationError: If the API key is invalid.
        openai.NotFoundError: If the model is not found.
        openai.PermissionDeniedError: If access is denied.
    """
    if local and not endpoints:
        raise ValueError("endpoints required when local=True")

    start = time.perf_counter()

    msgs: list[dict[str, Any]] = list(messages)  # type: ignore[arg-type]

    try:
        if local:
            assert endpoints is not None
            r = _providers.complete_local(
                model=model,
                messages=msgs,
                endpoints=endpoints,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                max_retries=max_retries,
            )
        else:
            r = _providers.complete_litellm(
                model=model,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
            )
    except _PER_REQUEST_ERRORS as e:
        return CompletionResult(model=model, error=str(e), latency=time.perf_counter() - start)

    latency = time.perf_counter() - start
    inp = r.get("input_tokens", 0)
    out = r.get("output_tokens", 0)
    think = r.get("thinking_tokens", 0)

    result = CompletionResult(
        content=r.get("content"),
        tool_calls=r.get("tool_calls"),
        message=r.get("message", {}),
        tokens={"input": inp, "output": out, "thinking": think, "total": inp + out + think},
        cost=r.get("cost", 0.0),
        latency=latency,
        model=model,
        error=r.get("error"),
    )

    if on_complete:
        on_complete(result)

    return result
