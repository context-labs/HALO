from __future__ import annotations

from typing import Any, cast

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


# Responses API input messages carry ``input_text`` content parts (assistant
# output uses ``output_text``, and ``function_call`` / ``function_call_output``
# items have no content-part list at all). Only ``system`` and ``user`` items
# are rewritten below, so ``input_text`` is always the correct part type.
_CACHEABLE_INPUT_ROLES = frozenset({"system", "user"})


def apply_prompt_cache_breakpoints(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the stable head of a Responses ``input`` array as cacheable.

    The main agent loop replays the entire conversation on every turn, so the
    head of that array — the engine-rendered system prompt plus the task
    prompt — is byte-identical across every turn of a run and across every
    retry. It is also the largest static block HALO sends. Left unmarked,
    Anthropic re-reads it at full price on all of them; a 50-turn run pays for
    the same prefix 50 times.

    Two breakpoints are placed (Anthropic allows four): the leading ``system``
    item, and the last ``user`` item in the head run of input-role items.
    Anthropic caches the longest matching prefix up to a breakpoint, so the
    second one extends the cached region to cover the task prompt as well.
    OpenAI ignores ``cache_control`` and does prefix caching automatically, so
    this is a Pareto improvement rather than a provider trade-off — the same
    reasoning as :func:`as_cached_system_message`.

    Deliberately narrow: only ``system`` / ``user`` message items are rewritten.
    Marking a trailing ``function_call_output`` would extend caching over the
    growing tool-result history — a larger win — but that item's content shape
    is not one this function can verify, and a malformed replay item is exactly
    the failure that burned a whole retry budget deterministically in run
    ba6fc95c. That extension needs a live wire test first.

    Returns a new list; input items are not mutated. Items already carrying
    list-shaped content are left alone rather than double-wrapped.
    """
    result = [dict(item) for item in items]

    cacheable_head: list[int] = []
    for index, item in enumerate(result):
        if item.get("role") not in _CACHEABLE_INPUT_ROLES:
            break
        cacheable_head.append(index)

    if not cacheable_head:
        return result

    breakpoints = {cacheable_head[0], cacheable_head[-1]}
    for index in breakpoints:
        marked = _as_cached_input_item(result[index])
        if marked is not None:
            result[index] = marked
    return result


def _as_cached_input_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Rewrite ``content`` as a single cache-marked ``input_text`` part.

    ``None`` means "leave this item alone": empty content has nothing worth
    caching, and content that is already a list came from somewhere that owns
    its own block structure.
    """
    content = item.get("content")
    if not isinstance(content, str) or not content:
        return None
    return {
        **item,
        "content": [
            {
                "type": "input_text",
                "text": content,
                "cache_control": CACHE_CONTROL_EPHEMERAL,
            }
        ],
    }
