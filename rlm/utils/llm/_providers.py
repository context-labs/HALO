"""Provider backends: LiteLLM proxy for cloud models, direct OpenAI for local models."""

from __future__ import annotations

import itertools
import logging
import os
import time
from typing import Any

import httpx
import openai
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

logger = logging.getLogger(__name__)

__all__ = ["complete_litellm", "complete_local", "list_models", "print_models"]

LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "https://litellm.inference.cool/v1")

_litellm_client = None
_local_clients: dict[str, "openai.OpenAI"] = {}
_round_robin = itertools.count()


def _get_litellm_client() -> openai.OpenAI:
    """Get or create the cached LiteLLM proxy client."""
    global _litellm_client
    if _litellm_client is None:
        api_key = os.environ.get("LITELLM_API_KEY")
        if not api_key:
            raise ValueError("Missing env var LITELLM_API_KEY")
        _litellm_client = openai.OpenAI(
            base_url=LITELLM_BASE_URL,
            api_key=api_key,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
    return _litellm_client


def _get_local_client(base_url: str) -> openai.OpenAI:
    """Get or create a cached local endpoint client."""
    if base_url not in _local_clients:
        api_key = os.environ.get("LOCAL_API_KEY", "no-key")
        _local_clients[base_url] = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
    return _local_clients[base_url]


def _parse_chat_response(response: Any) -> dict[str, Any]:
    """Extract content, tool_calls, tokens from a Chat Completions response."""
    msg = response.choices[0].message

    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    reasoning_tokens = 0
    if usage and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0

    content = msg.content
    tool_calls = None
    if msg.tool_calls:
        tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]

    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "content": content,
        "tool_calls": tool_calls,
        "message": message,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "thinking_tokens": reasoning_tokens,
    }


def _build_params(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float | None,
    max_tokens: int | None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the keyword-argument dict for ``chat.completions.create``.

    ``temperature`` and ``max_tokens`` are omitted from the payload when
    ``None`` so the model or proxy can apply its own default. This matters
    for models like gpt-5.x that reject ``temperature=0``.
    """
    params: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if tools:
        params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = tool_choice
    if response_format:
        params["response_format"] = response_format
    return params


def complete_litellm(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call a cloud model via the LiteLLM proxy.

    No client-side retries -- the proxy handles retries, rate limits, and
    fallbacks server-side. Cost is extracted from the ``x-litellm-response-cost``
    response header.

    Args:
        model: Model identifier (e.g. "gpt-5.4", "claude-sonnet-4-20250514").
        messages: Conversation history in OpenAI message format.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the completion.
        tools: Tool definitions in OpenAI function-calling format.
        tool_choice: Tool selection strategy.
        response_format: Response format constraint.

    Returns:
        Dict with keys: content, tool_calls, message, input_tokens, output_tokens,
        thinking_tokens, cost, error.
    """
    client = _get_litellm_client()
    params = _build_params(
        model, messages, temperature, max_tokens, tools, tool_choice, response_format
    )

    raw_response = client.chat.completions.with_raw_response.create(**params)
    cost = float(raw_response.headers.get("x-litellm-response-cost", 0))
    parsed = raw_response.parse()

    result = _parse_chat_response(parsed)
    result["cost"] = cost
    result["error"] = None
    return result


def complete_local(
    model: str,
    messages: list[dict[str, Any]],
    endpoints: list[str],
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    response_format: dict[str, Any] | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call a local model, round-robin across endpoints with retry.

    Requests are distributed across endpoints using a global round-robin counter.
    On transient failures (rate limits, timeouts, generic errors), retries with
    exponential backoff. Per-request errors (400, 422) are raised immediately
    without retry.

    Args:
        model: Model identifier served by the local endpoint.
        messages: Conversation history in OpenAI message format.
        endpoints: List of base URLs (e.g. ["http://10.0.1.28:8011/v1"]).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the completion.
        tools: Tool definitions in OpenAI function-calling format.
        tool_choice: Tool selection strategy.
        response_format: Response format constraint.
        max_retries: Maximum number of retry attempts.

    Returns:
        Dict with keys: content, tool_calls, message, input_tokens, output_tokens,
        thinking_tokens, cost (always 0.0), error.

    Raises:
        openai.BadRequestError: On 400 errors (raised immediately, no retry).
        openai.UnprocessableEntityError: On 422 errors (raised immediately, no retry).
        RuntimeError: If all retry attempts are exhausted.
    """
    params = _build_params(
        model, messages, temperature, max_tokens, tools, tool_choice, response_format
    )

    for attempt in range(max_retries):
        endpoint = endpoints[next(_round_robin) % len(endpoints)]
        client = _get_local_client(endpoint)
        try:
            response = client.chat.completions.create(**params)
            result = _parse_chat_response(response)
            result["cost"] = 0.0
            result["error"] = None
            return result
        except openai.RateLimitError:
            wait = 2 ** (attempt + 1)
            logger.warning(
                "Local rate limited (%s, attempt %d/%d), waiting %ds",
                endpoint,
                attempt + 1,
                max_retries,
                wait,
            )
            time.sleep(wait)
        except openai.APITimeoutError as e:
            logger.warning(
                "Local timeout (%s, attempt %d/%d): %s", endpoint, attempt + 1, max_retries, e
            )
            time.sleep(2)
        except (openai.BadRequestError, openai.UnprocessableEntityError):
            raise
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(
                "Local error (%s, attempt %d/%d): %s", endpoint, attempt + 1, max_retries, e
            )
            time.sleep(2**attempt)

    raise RuntimeError(f"Local model {model} failed after {max_retries} retries")


def list_models(*, filter: str | None = None) -> list[str]:
    """Query the LiteLLM proxy /v1/models endpoint and return model IDs.

    Args:
        filter: Substring filter on model ID (e.g. "claude", "gemini", "gpt").

    Returns:
        Sorted list of model ID strings.
    """
    client = _get_litellm_client()
    response = client.models.list()
    ids = sorted(m.id for m in response.data)
    if filter:
        f = filter.lower()
        ids = [mid for mid in ids if f in mid.lower()]
    return ids


def print_models(*, filter: str | None = None) -> None:
    """Query and pretty-print available models from the LiteLLM proxy."""
    from rich.console import Console
    from rich.table import Table

    ids = list_models(filter=filter)
    console = Console()
    if not ids:
        console.print("[yellow]No models found.[/yellow]")
        return

    title = f"LiteLLM models ({len(ids)})"
    if filter:
        title += f" — filter: {filter!r}"
    table = Table(title=title, show_lines=False)
    table.add_column("Model ID", style="cyan")
    for mid in ids:
        table.add_row(mid)
    console.print(table)
