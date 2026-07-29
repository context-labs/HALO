from __future__ import annotations

from typing import cast

from openai.types.chat import ChatCompletionSystemMessageParam

CACHE_CONTROL_EPHEMERAL: dict[str, str] = {"type": "ephemeral"}
"""Anthropic ``cache_control`` hint placed on the trailing block of a cacheable
prefix. ``ephemeral`` is the only currently-shipped cache type (5-minute TTL by
default, refreshed on each read)."""


def as_cached_system_message(text: str) -> ChatCompletionSystemMessageParam:
    """Build a ``role=system`` chat-completion message with prompt caching enabled.

    HALO talks the OpenAI Chat Completions API surface but the bulk of its
    production traffic routes to Anthropic models through LiteLLM / Catalyst.
    Anthropic exposes explicit prompt caching via a per-content-block
    ``cache_control: {"type": "ephemeral"}`` hint, which LiteLLM passes
    through verbatim. OpenAI does prefix caching automatically on byte-stable
    prefixes ≥1024 tokens and silently ignores the hint, so attaching the
    hint here is a Pareto improvement — it switches Anthropic on without
    regressing the OpenAI path.

    The returned message uses the content-list shape (a list of one text
    block) rather than the plain-string shape so the ``cache_control`` key
    has somewhere to land. The OpenAI Python SDK passes content-block dicts
    through to the wire verbatim, so additional keys reach the upstream
    provider unmolested.

    ``cache_control`` is not in the OpenAI ``ChatCompletionContentPartTextParam``
    TypedDict, so we cast the assembled message at this boundary; callers see
    a properly typed ``ChatCompletionSystemMessageParam`` and the extra key
    rides on the wire as intended.

    Callers should treat ``text`` as a byte-stable prefix: dynamic per-call
    state belongs in subsequent ``user`` messages, not interleaved into the
    system block. A single byte of drift in the prefix invalidates the cache
    on every call.
    """
    message = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": text,
                "cache_control": CACHE_CONTROL_EPHEMERAL,
            }
        ],
    }
    return cast(ChatCompletionSystemMessageParam, message)
