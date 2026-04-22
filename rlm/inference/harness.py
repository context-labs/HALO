"""The HALO RLM agent loop.

Descriptor-driven: the system prompt, available tools, and outcome semantics
all adapt to whichever ``DatasetDescriptor`` is passed in. Each call to
``run_agent`` is a generator of ``AgentEvent``s consumed by the CLI and the
web server.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from loguru import logger
from utils.llm._complete import CompletionResult
from utils.llm._providers import _build_params, _get_litellm_client

from dataset import DatasetDescriptor, IndexStore, TraceReader
from inference.config import InferenceConfig
from inference.tools import (
    TOOL_FUNCTIONS,
    TOOL_SCHEMAS,
    ToolContext,
    summarize_tool_result,
)


# Model pricing cache: fetched once from LiteLLM's /model/info, looked up
# by model name each turn. Streaming responses don't get the per-request
# ``x-litellm-response-cost`` header, so we compute it ourselves from
# (input_cost_per_token, output_cost_per_token) × reported usage.
_PRICING: dict[str, dict[str, float]] | None = None


def _load_pricing() -> dict[str, dict[str, float]]:
    global _PRICING
    if _PRICING is not None:
        return _PRICING
    import os
    import urllib.request
    table: dict[str, dict[str, float]] = {}
    base = os.environ.get("LITELLM_BASE_URL", "https://litellm.inference.cool/v1")
    base = base.rstrip("/").removesuffix("/v1")
    key = os.environ.get("LITELLM_API_KEY")
    if not key:
        _PRICING = table
        return table
    try:
        req = urllib.request.Request(
            f"{base}/model/info",
            headers={
                "Authorization": f"Bearer {key}",
                # Cloudflare in front of LiteLLM blocks the default
                # ``Python-urllib/*`` UA with a 1010 error. Any non-default
                # UA passes.
                "User-Agent": "halo-rlm/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        for m in data.get("data", []):
            name = m.get("model_name")
            info = m.get("model_info") or {}
            inp = info.get("input_cost_per_token")
            out = info.get("output_cost_per_token")
            if name and (inp is not None or out is not None):
                table[name] = {
                    "input": float(inp or 0.0),
                    "output": float(out or 0.0),
                }
    except Exception:
        logger.exception("failed to load LiteLLM pricing table")
    _PRICING = table
    return table


def _compute_cost(model: str, usage: Any) -> float:
    """Compute USD cost from streamed usage + cached LiteLLM pricing."""
    if usage is None:
        return 0.0
    table = _load_pricing()
    rates = table.get(model)
    if not rates:
        return 0.0
    inp = getattr(usage, "prompt_tokens", 0) or 0
    out = getattr(usage, "completion_tokens", 0) or 0
    return round(inp * rates["input"] + out * rates["output"], 8)

def _stream_chat_completion(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> Any:
    """Generator: yields text deltas as they stream; returns a
    ``CompletionResult`` via ``StopIteration.value``.

    Bypasses ``utils.llm.complete`` because the shared helper is non-streaming.
    The top-level agent LLM call is the only place where streaming matters
    for UX — the synthesize sub-agent still uses the non-streaming path.
    """
    client = _get_litellm_client()
    params = _build_params(
        model=model, messages=messages,
        temperature=None, max_tokens=None,
        tools=tools, tool_choice=None, response_format=None,
    )
    # ``stream_options.include_usage`` is OpenAI's way of emitting usage on
    # the final chunk in streaming mode. Many LiteLLM backends honour it.
    params["stream"] = True
    params["stream_options"] = {"include_usage": True}

    start = time.time()
    content_parts: list[str] = []
    tool_accum: dict[int, dict[str, Any]] = {}
    usage: Any = None

    # Coalesce tokens that arrive within ``FLUSH_MS`` of each other so fast
    # streamers don't spam the SSE/thread pipeline with one tiny event per
    # token. The yielded text is byte-identical to the untouched stream —
    # only the event count is reduced.
    FLUSH_MS = 30
    pending_text = ""
    pending_since_ns = 0

    def _drain() -> str:
        nonlocal pending_text, pending_since_ns
        out = pending_text
        pending_text = ""
        pending_since_ns = 0
        return out

    try:
        stream = client.chat.completions.create(**params)
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                if pending_since_ns == 0:
                    pending_since_ns = time.monotonic_ns()
                pending_text += delta.content
                if (time.monotonic_ns() - pending_since_ns) // 1_000_000 >= FLUSH_MS:
                    yield _drain()
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    # Some providers omit ``index`` on streamed tool_calls.
                    # Falling back to a fixed 0 would collapse every call
                    # into one slot and concat their names/arguments. Use
                    # a fresh slot when index is missing AND this chunk
                    # introduces a new id.
                    idx = getattr(tc, "index", None)
                    tc_id = getattr(tc, "id", None)
                    if idx is None:
                        matched = None
                        if tc_id:
                            for i, slot in tool_accum.items():
                                if slot["id"] == tc_id:
                                    matched = i
                                    break
                        idx = matched if matched is not None else len(tool_accum)
                    slot = tool_accum.setdefault(
                        idx,
                        {"id": "", "type": "function",
                         "function": {"name": "", "arguments": ""}},
                    )
                    if getattr(tc, "id", None):
                        slot["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            slot["function"]["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            slot["function"]["arguments"] += fn.arguments
    except Exception as e:
        # Attach an error onto the CompletionResult so the caller can bail
        # cleanly rather than re-raising mid-generator.
        logger.exception("streaming LLM call failed")
        return CompletionResult(
            model=model, error=str(e), latency=time.time() - start,
        )

    # Flush any tail buffered by the 30ms coalescer so the caller sees the
    # full content before the return.
    if pending_text:
        yield _drain()

    content = "".join(content_parts) or None
    tool_calls = [tool_accum[i] for i in sorted(tool_accum)]
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls

    inp = getattr(usage, "prompt_tokens", 0) if usage else 0
    out = getattr(usage, "completion_tokens", 0) if usage else 0
    think = 0
    if usage and getattr(usage, "completion_tokens_details", None):
        think = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0

    return CompletionResult(
        content=content,
        tool_calls=tool_calls or None,
        message=message,
        tokens={"input": inp, "output": out, "thinking": think,
                "total": inp + out + think},
        cost=_compute_cost(model, usage),
        latency=time.time() - start,
        model=model,
        error=None,
    )


def build_system_prompt(
    descriptor: DatasetDescriptor,
    store: IndexStore | None = None,
    *,
    depth: int = 0,
    max_depth: int = 1,
    max_turns: int = 16,
) -> str:
    """Compose a descriptor-specific system prompt.

    If ``store`` is provided, the top values for each label are injected so
    the analyst knows the vocabulary it can filter on. ``depth`` controls
    whether the ``ask_subagent`` decomposition guidance appears (only
    relevant for agents that can still spawn sub-agents). ``max_turns`` is
    surfaced so the analyst (especially a sub-agent at depth > 0) knows
    its budget and can plan accordingly.
    """
    overview = store.overview() if store is not None else None

    # Label vocabulary block.
    label_block_lines: list[str] = []
    if descriptor.labels and overview is not None:
        for name in descriptor.label_names:
            dist = (overview.get("labels") or {}).get(name, {})
            top = list(dist.items())[:6]
            rendered = ", ".join(f"{v} ({n})" for v, n in top) or "(no values)"
            label_block_lines.append(f"- `{name}`: {rendered}")
    elif descriptor.labels:
        label_block_lines = [f"- `{n}`" for n in descriptor.label_names]
    labels_block = "\n".join(label_block_lines) if label_block_lines else "(no categorical labels)"

    # Primary-metric semantics block.
    outcome_block = "(this dataset has no primary metric; ignore min_outcome/max_outcome)"
    if descriptor.primary_metric is not None:
        pm = descriptor.primary_metric
        direction = "higher is better" if pm.higher_is_better else "lower is better"
        outcome_block = (
            f"This dataset's primary metric is `{pm.label}` ({pm.kind}, {direction}). "
            f"Values at or past {pm.perfect_threshold} count as 'perfect'; values past "
            f"{pm.zero_threshold} count as 'zero'. Use `min_outcome` / `max_outcome` "
            f"filters and `sample_by_outcome`."
        )

    paths_block = ""
    if descriptor.has_documents:
        paths_block = (
            "\nEach trace carries a set of documents with file paths. You can filter or sort "
            "on `max_path_length` and `nested_path_count` to focus on deeply-nested code-repo "
            "traces vs. flat ones."
        )

    # Decomposition guidance — only shown when we can still recurse.
    can_recurse = depth < max_depth
    subagent_block = ""
    if can_recurse:
        subagent_block = (
            "\n8. **Use `ask_subagent`** to decompose a multi-faceted question into focused "
            "sub-questions. For example, given a top-level question like *\"when is the harness "
            "failing?\"*, a logical next step is to call `ask_subagent` with *\"where are there "
            "hallucinations in this dataset?\"* and another with *\"where is path handling "
            "breaking?\"*. Each sub-agent runs a full tool loop over the same dataset and returns "
            "a grounded one-paragraph answer. **Prefer decomposition when the question has "
            "genuinely independent parts** — the sub-agents run bounded (≈8 turns) and their "
            "answers are cheaper to integrate than trying to keep all threads in your own "
            "context at once."
        )

    desc_blurb = descriptor.description or ""
    count = f" ({overview['count']:,} traces)" if overview else ""
    source = f" produced by **{descriptor.source_model}**" if descriptor.source_model else ""

    role_line = (
        "You are **halo**, an expert AI analyst that answers questions about agent-trace datasets."
        if depth == 0
        else f"You are **halo** running as a depth-{depth} sub-agent. A parent analyst has delegated a focused sub-question to you. Answer it tightly and concretely — your answer becomes one tool result in the parent's trajectory."
    )
    budget_line = (
        f"You have a budget of **up to {max_turns} tool-calling turns** for this run. "
        f"Plan accordingly — prefer a few high-information tool calls over many shallow ones."
    )

    return f"""{role_line}

{budget_line}

You are currently analyzing: **{descriptor.name}**{count}{source}. {desc_blurb}

A *trace* is one full run of another agent on one task: a user query, a conversation of tool calls, a final answer, and (optionally) an outcome score comparing the agent's answer to ground truth.

You answer by calling tools, never by guessing. Your answers must be grounded in specific traces (cite `id`s) or in explicit aggregate statistics you pulled from the index.

## Available categorical labels for this dataset
{labels_block}

## Outcome signal
{outcome_block}{paths_block}

## How to investigate

1. **Start with `dataset_overview`** to understand scale and distributions. Pass filters to zoom in.
2. **Use `find_traces` / `sample_by_outcome`** to locate exemplars — longest traces, failed traces, traces using a specific tool.
3. **Use `inspect_trace`** to drill into one exemplar. Do this 2-4 times to form hypotheses.
4. **Use `search_trace`** to grep inside one trace (e.g. patterns the agent hallucinated vs. what the documents actually say).
5. **Use `synthesize`** when you need to reason over a population too large to inspect one-by-one. Pass 10-25 `id`s and a sub-question.
6. **Use `compare_groups`** to contrast two slices side-by-side.
7. **Use `run_code`** as an escape hatch when the fixed tools can't express what you need. The subprocess has `store`, `reader`, and `descriptor` pre-bound — you can run ad-hoc `store.filter(...)` aggregations, custom groupings, multi-pass filters, etc. Print what you want back, or assign to `result`/`out`/`ans`.{subagent_block}

## Context compaction

As the conversation grows, older tool results get collapsed to a one-line summary of the form `[compacted · r_N] <tool> → <summary>`. The full JSON is still available — call `inspect_result(key="r_N")` if you need the details back. Prefer re-running the tool with a narrower filter when possible; `inspect_result` is for when a specific earlier result matters to the current step.

## Output format

- Open with a one-sentence **TL;DR**.
- Follow with **3-6 bullets** of specific findings, each citing evidence (`id`s or aggregate numbers).
- For failure / hallucination questions, include a short **"How I'd fix this"** section.
- Never fabricate ids, tool names, or numbers. Only use values you obtained from tool results.

Prefer fewer, higher-quality tool calls over many shallow ones. 5-8 turns is usually enough.
"""


@dataclass
class AgentEvent:
    """One event emitted by the agent loop."""

    kind: str  # "start" | "thinking" | "tool_call" | "tool_result" | "final" | "error" | "usage"
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "data": self.data}


def _make_context(
    config: InferenceConfig,
    descriptor: DatasetDescriptor,
    store: IndexStore,
    reader: TraceReader,
    *,
    result_store: dict[str, Any] | None = None,
    next_result_key: int = 0,
    depth: int = 0,
) -> ToolContext:
    ctx = ToolContext(
        descriptor=descriptor,
        store=store,
        reader=reader,
        synth_model=config.synth_model,
        synth_trace_cap=config.synth_trace_cap,
        synth_chars_per_trace=config.synth_chars_per_trace,
        sample_cap=config.sample_cap,
        depth=depth,
        subagent_model=config.model,  # sub-agents default to parent's model
        subagent_max_turns=config.subagent_max_turns,
        max_depth=config.max_depth,
    )
    # When continuing a conversation, reuse its persisted result_store so
    # ``inspect_result`` resolves keys from earlier turns.
    if result_store is not None:
        ctx.result_store = result_store
        ctx.next_result_key = next_result_key
    return ctx


def _compact_tool_messages(
    messages: list[dict[str, Any]],
    ctx: ToolContext,
    *,
    keep_recent: int,
) -> int:
    """In-place compaction of older tool messages.

    The last ``keep_recent`` tool-role messages keep their full content
    (the LLM may reference them directly on the next turn). Everything
    older is rewritten to a one-line summary plus the ``r_N`` key an
    analyst can pass to ``inspect_result`` to fetch the original back.
    Returns the number of messages compacted this round.
    """
    if keep_recent <= 0:
        return 0
    tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_idxs) <= keep_recent:
        return 0
    to_compact = tool_idxs[:-keep_recent]
    compacted = 0
    for idx in to_compact:
        msg = messages[idx]
        if msg.get("_halo_compacted"):
            continue
        key = msg.get("_halo_result_key")
        name = msg.get("_halo_tool_name", "?")
        full = ctx.result_store.get(key) if key else None
        summary = summarize_tool_result(name, full) if full is not None else "(no summary)"
        new_content = (
            f"[compacted · {key}] {name} → {summary}\n"
            f"(Use inspect_result('{key}') to retrieve the full JSON.)"
        )
        msg["content"] = new_content
        msg["_halo_compacted"] = True
        compacted += 1
    return compacted


def _dispatch_tool(ctx: ToolContext, name: str, raw_args: str) -> Any:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"unknown tool {name!r}"}
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError as e:
        return {"error": f"invalid JSON arguments: {e}"}
    if not isinstance(args, dict):
        return {"error": "tool arguments must be a JSON object"}
    started = time.time()
    try:
        result = fn(ctx, **args)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:
        logger.exception("tool {} raised", name)
        return {"error": f"tool raised: {e}"}
    elapsed = time.time() - started
    ctx.call_log.append({"name": name, "args": args, "latency_s": round(elapsed, 3)})
    return result


def run_agent(
    question: str,
    config: InferenceConfig,
    *,
    descriptor: DatasetDescriptor,
    store: IndexStore,
    reader: TraceReader | None = None,
    messages: list[dict[str, Any]] | None = None,
    result_store: dict[str, Any] | None = None,
    next_result_key: int = 0,
    depth: int = 0,
) -> Iterator[AgentEvent]:
    """Drive the agent loop, yielding ``AgentEvent``s as it progresses.

    When ``messages`` is passed, the agent *continues* that conversation:
    the system prompt and prior turns stay in context, and ``question`` is
    appended as the next user message. The list is mutated in place with
    the assistant's reply + tool exchanges so the caller can persist the
    updated state for the next turn.

    When ``messages`` is ``None``, a fresh conversation starts with a
    system prompt built from the descriptor.

    ``depth`` is the recursion level. Top-level callers leave it at 0;
    ``ask_subagent`` calls pass ``parent.depth + 1``. When depth reaches
    ``config.max_depth``, the ``ask_subagent`` tool is pruned from this
    agent's schema so it can't spawn further sub-agents.
    """
    owns_reader = False
    if reader is None:
        reader = TraceReader(descriptor.source_path)
        reader.__enter__()
        owns_reader = True

    try:
        ctx = _make_context(
            config, descriptor, store, reader,
            result_store=result_store, next_result_key=next_result_key,
            depth=depth,
        )
        # Prune ``ask_subagent`` from the schema at the depth limit. We
        # keep the function in TOOL_FUNCTIONS untouched (global registry)
        # but the LLM only sees the tools in ``tool_schemas``.
        can_recurse = depth < config.max_depth
        if can_recurse:
            tool_schemas = TOOL_SCHEMAS
        else:
            tool_schemas = [
                s for s in TOOL_SCHEMAS
                if s.get("function", {}).get("name") != "ask_subagent"
            ]

        if messages is None:
            messages = []
        # First-ever turn gets a system prompt. Subsequent turns reuse the
        # one already baked in.
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {
                "role": "system",
                "content": build_system_prompt(
                    descriptor, store,
                    depth=depth, max_depth=config.max_depth,
                    max_turns=config.max_turns,
                ),
            })
        messages.append({"role": "user", "content": question})
        yield AgentEvent("start", {
            "question": question,
            "dataset_id": descriptor.id,
            "dataset_name": descriptor.name,
            "model": config.model,
            "synth_model": config.synth_model,
            "max_turns": config.max_turns,
            "indexed_traces": len(store),
            "depth": depth,
        })

        total_cost = 0.0
        for turn in range(1, config.max_turns + 1):
            # Stream the top-level LLM call so the user sees text as it
            # arrives. ``_stream_chat_completion`` is a generator: it yields
            # text deltas, and its return-value (captured from StopIteration)
            # is a ``CompletionResult`` with the full aggregated content +
            # tool_calls + usage.
            stream = _stream_chat_completion(config.model, messages, tool_schemas)
            try:
                while True:
                    try:
                        delta = next(stream)
                    except StopIteration as stop:
                        result = stop.value
                        break
                    yield AgentEvent("delta", {"turn": turn, "content": delta})
            except Exception as e:
                logger.exception("LLM call failed at turn {}", turn)
                yield AgentEvent("error", {"turn": turn, "message": f"LLM call failed: {e}"})
                return

            if result is None or result.error:
                msg = result.error if result else "unknown streaming error"
                yield AgentEvent("error", {"turn": turn, "message": msg})
                return

            total_cost += result.cost or 0.0
            yield AgentEvent("usage", {
                "turn": turn,
                "tokens": result.tokens,
                "cost": result.cost,
                "model": result.model,
                "latency_s": round(result.latency, 2),
            })

            messages.append(result.message)

            tool_calls = result.tool_calls or []

            # Only emit a ``thinking`` event when the turn also invokes
            # tools — in that case the content is intermediate reasoning
            # that accompanies a tool call. When there are no tool calls,
            # the content *is* the final answer; emit only ``final`` so
            # the same text isn't rendered twice (blue thinking block +
            # orange final block).
            if result.content and tool_calls:
                yield AgentEvent("thinking", {"turn": turn, "content": result.content})

            if not tool_calls:
                yield AgentEvent("final", {
                    "content": result.content or "",
                    "total_cost": round(total_cost + ctx.cost, 6),
                    "turns_used": turn,
                    "tool_calls_made": len(ctx.call_log),
                })
                return

            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                raw_args = fn.get("arguments") or "{}"
                yield AgentEvent("tool_call", {
                    "turn": turn,
                    "id": tc.get("id"),
                    "name": name,
                    "arguments": raw_args,
                })
                result_obj = _dispatch_tool(ctx, name, raw_args)

                # Stash the full result in ctx.result_store under a short
                # key so older messages can be compacted without losing
                # access — the analyst can still fetch the full body via
                # ``inspect_result(key)``.
                result_key = ctx.stash_result(result_obj)

                payload = json.dumps(result_obj, ensure_ascii=False, default=str)
                if len(payload) > 40000:
                    payload = payload[:40000] + "... [truncated]"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "content": payload,
                    # Private keys (ignored by the LLM API) used by the
                    # compactor to replace content with a summary later.
                    "_halo_result_key": result_key,
                    "_halo_tool_name": name,
                })
                yield AgentEvent("tool_result", {
                    "turn": turn,
                    "id": tc.get("id"),
                    "name": name,
                    "result": result_obj,
                })

            # Compact now so the next LLM call doesn't pay for a growing
            # backlog of tool-result JSON. Keep the ``keep_recent`` most
            # recent tool messages in full; summarise the rest.
            _compact_tool_messages(
                messages, ctx,
                keep_recent=config.compact_keep_recent,
            )

        yield AgentEvent("final", {
            "content": "(Agent exhausted max_turns without producing a final answer.)",
            "total_cost": round(total_cost + ctx.cost, 6),
            "turns_used": config.max_turns,
            "tool_calls_made": len(ctx.call_log),
            "truncated": True,
        })
    finally:
        if owns_reader and reader is not None:
            reader.close()
