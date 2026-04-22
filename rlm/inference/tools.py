"""Tools the agent can call to interrogate a trace dataset.

Every tool is a pure function over a ``ToolContext`` (store + reader +
descriptor + synth config) and JSON-serialisable arguments. The tool
surface is descriptor-driven: filters accept a generic ``labels`` dict
(keys from ``descriptor.label_names``) and a numeric ``outcome`` range
interpreted per the descriptor's ``OutcomeSpec``.

``synthesize`` is the RLM-style primitive — it hands a sub-population of
traces to a secondary LLM call when the parent agent needs population-scale
reasoning.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from utils.llm import complete

from dataset import DatasetDescriptor, IndexStore, TraceReader, TraceSummary


@dataclass
class ToolContext:
    """Shared state injected into every tool call.

    ``result_store`` holds the full JSON of every tool result under a short
    key (``r_0``, ``r_1``, …). The harness uses this so older tool
    messages in the conversation can be compacted down to a one-line
    summary + reference while the full body stays retrievable via the
    ``inspect_result`` tool. Persists on the ``Conversation``.
    """

    descriptor: DatasetDescriptor
    store: IndexStore
    reader: TraceReader
    synth_model: str
    synth_trace_cap: int
    synth_chars_per_trace: int
    sample_cap: int
    cost: float = 0.0
    call_log: list[dict[str, Any]] = field(default_factory=list)
    result_store: dict[str, Any] = field(default_factory=dict)
    next_result_key: int = 0
    # Recursion depth of this agent. 0 = top-level analyst; 1+ = sub-agent
    # spawned by ``ask_subagent``. The harness prunes ``ask_subagent`` from
    # the tool schema when ``depth >= config.max_depth`` so sub-agents at
    # the depth limit simply don't see the tool.
    depth: int = 0
    # Model / turn limits to propagate to spawned sub-agents. Populated by
    # the harness when it constructs the context so the tool can read them
    # without needing the full InferenceConfig.
    subagent_model: str = "gpt-5.4"
    subagent_max_turns: int = 8
    max_depth: int = 1

    def stash_result(self, result: Any) -> str:
        """Save a tool result and return the key used to look it up.

        Increment-after-write so a failed store assignment doesn't burn the
        key (which would later collide with a fresh stash overwriting an
        existing entry).
        """
        key = f"r_{self.next_result_key}"
        self.result_store[key] = result
        self.next_result_key += 1
        return key


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _summary_to_dict(s: TraceSummary) -> dict[str, Any]:
    return {
        "id": s.id,
        "labels": s.labels,
        "query_preview": s.query_preview,
        "outcome": s.outcome,
        "ground_truth_count": s.ground_truth_count,
        "n_messages": s.n_messages,
        "n_tool_calls": s.n_tool_calls,
        "tool_errors": s.tool_errors,
        "turns_used": s.turns_used,
        "total_tokens": s.total_tokens,
        "tools_used": s.tools_used,
        "max_path_length": s.max_path_length,
        "nested_path_count": s.nested_path_count,
        "sample_paths": s.sample_paths,
        "has_final_answer": s.has_final_answer,
        "final_answer_chars": s.final_answer_chars,
    }


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + f"\n... [truncated {len(text) - limit} chars]"


def _render_trace_for_synth(
    record: dict[str, Any], descriptor: DatasetDescriptor, char_budget: int
) -> str:
    """Render a single trace as a compact string for LLM synthesis.

    Format-agnostic — reads the trace through a ``TraceView`` so this
    works for both HF and OpenInference records.
    """
    from dataset.formats import view_from_record

    view = view_from_record(record, descriptor)
    lines: list[str] = [f"# Trace {view.id}"]

    labels = view.labels
    if labels:
        lines.append("labels: " + ", ".join(f"{k}={v}" for k, v in labels.items()))

    if view.outcome is not None and descriptor.primary_metric is not None:
        lines.append(f"{descriptor.primary_metric.label}: {view.outcome}")
    if descriptor.has_ground_truth:
        lines.append(f"ground_truth: {view.ground_truth}")

    if view.query:
        lines.append(f"query: {view.query}")

    usage = view.usage
    lines.append(
        f"turns_used={view.turns_used} tool_calls={view.tool_calls_total} "
        f"tool_errors={view.tool_errors} tokens={usage.get('total_tokens')}"
    )

    final = view.final_answer
    if isinstance(final, dict):
        lines.append("final_answer: " + json.dumps(final)[: char_budget // 3])
    elif isinstance(final, str):
        lines.append("final_answer: " + final[: char_budget // 3])

    lines.append("tool_calls:")
    for m in view.messages():
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                lines.append(f"  - {tc.name}({tc.arguments})")
        elif m.role == "tool":
            content = m.content or ""
            lines.append(f"    -> {content[:400]!r}")

    body = "\n".join(lines)
    return _clip(body, char_budget)


# ------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------


def tool_dataset_overview(
    ctx: ToolContext,
    *,
    labels: dict[str, Any] | None = None,
    tool_used: list[str] | None = None,
    min_max_path_length: int | None = None,
    min_nested_paths: int | None = None,
) -> dict[str, Any]:
    """Return headline distributions for the dataset (or a filtered subset)."""
    rows = ctx.store.filter(
        labels=labels,
        tool_used=tool_used,
        min_max_path_length=min_max_path_length,
        min_nested_paths=min_nested_paths,
    )
    return ctx.store.overview(rows)


def tool_find_traces(
    ctx: ToolContext,
    *,
    labels: dict[str, Any] | None = None,
    tool_used: list[str] | None = None,
    min_outcome: float | None = None,
    max_outcome: float | None = None,
    outcome_bucket: str | None = None,
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    min_tool_errors: int | None = None,
    has_final_answer: bool | None = None,
    min_max_path_length: int | None = None,
    min_nested_paths: int | None = None,
    sort_by: str = "n_messages",
    sort_desc: bool = True,
    limit: int = 10,
    random_sample: bool = False,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Filter + sort the dataset; return a small batch of trace summaries."""
    rows = ctx.store.filter(
        labels=labels,
        tool_used=tool_used,
        min_outcome=min_outcome,
        max_outcome=max_outcome,
        outcome_bucket=outcome_bucket if outcome_bucket in {"perfect", "partial", "zero"} else None,
        min_messages=min_messages,
        max_messages=max_messages,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        min_tool_errors=min_tool_errors,
        has_final_answer=has_final_answer,
        min_max_path_length=min_max_path_length,
        min_nested_paths=min_nested_paths,
    )
    total = len(rows)
    limit = max(1, min(limit, ctx.sample_cap))

    valid_sorts = {
        "n_messages", "n_tool_calls", "total_tokens", "turns_used",
        "outcome", "tool_errors", "final_answer_chars",
        "ground_truth_count", "max_path_length", "nested_path_count",
    }
    if random_sample:
        rows = ctx.store.sample(rows, limit, seed=seed)
    else:
        key_field = sort_by if sort_by in valid_sorts else "n_messages"

        def _key(r: TraceSummary) -> float:
            v = getattr(r, key_field)
            return float(v) if v is not None else float("-inf" if sort_desc else "inf")

        rows = sorted(rows, key=_key, reverse=sort_desc)[:limit]

    return {
        "total_matching": total,
        "returned": len(rows),
        "sort_by": sort_by,
        "sort_desc": sort_desc,
        "random_sample": random_sample,
        "traces": [_summary_to_dict(r) for r in rows],
    }


def tool_get_trace(ctx: ToolContext, *, id: str, max_chars: int = 20000) -> dict[str, Any]:
    """Fetch the full trace record, clipped to ``max_chars``."""
    summary = ctx.store.lookup(id)
    if summary is None:
        return {"error": f"unknown id {id!r}"}
    record = ctx.reader.read(summary.byte_offset, summary.byte_length)
    text = json.dumps(record, ensure_ascii=False)
    if len(text) <= max_chars:
        return {"id": id, "record": record}
    return {
        "id": id,
        "truncated": True,
        "record_json_preview": text[:max_chars],
        "hint": "Use inspect_trace for a structured view.",
    }


def tool_inspect_trace(
    ctx: ToolContext, *, id: str, content_chars: int = 600
) -> dict[str, Any]:
    """Structured, LLM-friendly view of one trace. Format-agnostic."""
    from dataset.formats import view_from_record

    summary = ctx.store.lookup(id)
    if summary is None:
        return {"error": f"unknown id {id!r}"}
    record = ctx.reader.read(summary.byte_offset, summary.byte_length)
    desc = ctx.descriptor
    view = view_from_record(record, desc)

    steps: list[dict[str, Any]] = []
    for m in view.messages():
        if m.role == "assistant":
            for tc in m.tool_calls:
                steps.append({
                    "role": "assistant",
                    "tool_call": {"name": tc.name, "arguments": tc.arguments},
                })
            if m.content:
                steps.append({"role": "assistant", "content": _clip(m.content, content_chars)})
        elif m.role == "tool":
            steps.append({"role": "tool", "content": _clip(m.content, content_chars)})
        elif m.role == "user":
            steps.append({"role": "user", "content": _clip(m.content, content_chars)})

    return {
        "id": view.id,
        "query": view.query,
        "labels": view.labels,
        "ground_truth": view.ground_truth,
        "outcome_name": desc.primary_metric.label if desc.primary_metric else None,
        "outcome": view.outcome,
        "document_paths": [d.path for d in view.documents()],
        "final_answer": view.final_answer,
        "steps": steps,
        # Flat per-trace stats the UI renders in the header chip row.
        # Fields that don't apply to a given dataset come back as
        # None / 0 and the UI hides them.
        "turns_used": view.turns_used,
        "tool_calls_total": view.tool_calls_total,
        "tokens_total": (view.usage or {}).get("total_tokens"),
    }


def tool_search_trace(
    ctx: ToolContext,
    *,
    id: str,
    pattern: str,
    case_insensitive: bool = True,
    context: int = 80,
    max_matches: int = 20,
) -> dict[str, Any]:
    """Regex-search within a single trace's messages and documents. Format-agnostic."""
    from dataset.formats import view_from_record

    summary = ctx.store.lookup(id)
    if summary is None:
        return {"error": f"unknown id {id!r}"}
    record = ctx.reader.read(summary.byte_offset, summary.byte_length)
    view = view_from_record(record, ctx.descriptor)

    flags = re.IGNORECASE if case_insensitive else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as e:
        return {"error": f"invalid regex: {e}"}

    hits: list[dict[str, Any]] = []

    def _scan(label: str, text: str) -> None:
        for m in rx.finditer(text):
            start = max(0, m.start() - context)
            end = min(len(text), m.end() + context)
            hits.append({"location": label, "match": m.group(0), "snippet": text[start:end]})
            if len(hits) >= max_matches:
                return

    for i, msg in enumerate(view.messages()):
        if len(hits) >= max_matches:
            break
        if msg.content:
            _scan(f"messages[{i}].content", msg.content)
        for j, tc in enumerate(msg.tool_calls):
            if tc.arguments:
                _scan(f"messages[{i}].tool_calls[{j}].arguments", tc.arguments)

    for i, doc in enumerate(view.documents()):
        if len(hits) >= max_matches:
            break
        if doc.content:
            _scan(f"documents[{i}]({doc.path})", doc.content)

    return {"id": id, "pattern": pattern, "matches": hits}


def tool_sample_by_outcome(
    ctx: ToolContext,
    *,
    outcome: str,
    limit: int = 20,
    labels: dict[str, Any] | None = None,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Outcome-bucketed shortcut sampling. ``outcome`` is one of
    ``zero`` | ``partial`` | ``perfect`` | ``tool_errors`` | ``long`` | ``short``.

    Bucket semantics follow the dataset descriptor's ``OutcomeSpec`` and so
    are direction-aware (works for both higher-better and lower-better
    outcome scores) with no boundary overlap between buckets.
    """
    desc = ctx.descriptor
    outcome_bucket: str | None = None
    min_tool_errors: int | None = None
    sort_by = "n_messages"
    sort_desc = True
    random_sample = True

    if outcome in {"zero", "partial", "perfect"}:
        if desc.primary_metric is None:
            return {
                "error": "dataset has no primary metric to bucket on",
                "hint": "use outcome in {tool_errors, long, short} instead.",
            }
        outcome_bucket = outcome
    elif outcome == "tool_errors":
        min_tool_errors = 1
    elif outcome == "long":
        sort_by = "n_messages"
        sort_desc = True
        random_sample = False
    elif outcome == "short":
        sort_by = "n_messages"
        sort_desc = False
        random_sample = False
    else:
        return {"error": f"unknown outcome {outcome!r}"}

    return tool_find_traces(
        ctx,
        labels=labels,
        outcome_bucket=outcome_bucket,
        min_tool_errors=min_tool_errors,
        sort_by=sort_by,
        sort_desc=sort_desc,
        limit=limit,
        random_sample=random_sample,
        seed=seed,
    )


def tool_compare_groups(
    ctx: ToolContext, *, group_a: dict[str, Any], group_b: dict[str, Any]
) -> dict[str, Any]:
    """Parallel overviews over two filter specs."""

    def _filter(spec: dict[str, Any]) -> list[TraceSummary]:
        return ctx.store.filter(
            labels=spec.get("labels"),
            tool_used=spec.get("tool_used"),
            min_outcome=spec.get("min_outcome"),
            max_outcome=spec.get("max_outcome"),
            min_messages=spec.get("min_messages"),
            max_messages=spec.get("max_messages"),
            min_tokens=spec.get("min_tokens"),
            max_tokens=spec.get("max_tokens"),
            min_tool_errors=spec.get("min_tool_errors"),
            has_final_answer=spec.get("has_final_answer"),
            min_max_path_length=spec.get("min_max_path_length"),
            min_nested_paths=spec.get("min_nested_paths"),
        )

    a_rows = _filter(group_a)
    b_rows = _filter(group_b)
    return {
        "group_a": {"filter": group_a, "overview": ctx.store.overview(a_rows)},
        "group_b": {"filter": group_b, "overview": ctx.store.overview(b_rows)},
    }


def tool_synthesize(
    ctx: ToolContext, *, ids: list[str], question: str
) -> dict[str, Any]:
    """RLM-style: render the given traces compactly and ask a second LLM to answer."""
    if not ids:
        return {"error": "ids is empty"}
    picked = list(dict.fromkeys(ids))[: ctx.synth_trace_cap]

    bodies: list[str] = []
    missing: list[str] = []
    for tid in picked:
        summary = ctx.store.lookup(tid)
        if summary is None:
            missing.append(tid)
            continue
        record = ctx.reader.read(summary.byte_offset, summary.byte_length)
        bodies.append(_render_trace_for_synth(record, ctx.descriptor, ctx.synth_chars_per_trace))

    if not bodies:
        return {"error": "no traces available", "missing": missing}

    system = (
        "You are an expert agent-trace analyst. Given a batch of traces and a question, "
        "answer the question grounded *only* in what the traces actually show. Quote tool "
        "calls, paths, and final_answer snippets when useful. Be concrete; prefer evidence "
        "over generalities. If the traces don't support an answer, say so."
    )
    user = (
        f"Question: {question}\n\n"
        f"Number of traces provided: {len(bodies)}\n\n"
        + "\n\n---\n\n".join(bodies)
    )
    try:
        result = complete(
            ctx.synth_model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as e:
        logger.exception("synthesize LLM call failed")
        return {"error": f"synth model call failed: {e}"}
    ctx.cost += result.cost or 0.0
    return {
        "question": question,
        "traces_used": len(bodies),
        "missing": missing,
        "answer": result.content,
        "synth_model": ctx.synth_model,
        "synth_cost": result.cost,
    }


def tool_run_code(
    ctx: ToolContext,
    *,
    code: str,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Execute arbitrary Python in a sandboxed subprocess.

    The runner pre-binds ``store`` (the ``IndexStore``), ``reader``
    (the ``TraceReader`` open on the source JSONL), and ``descriptor``
    as module-level variables. Great for ad-hoc aggregations the fixed
    tool set doesn't cover. ``print`` output + any values assigned to
    ``result``, ``out``, or ``ans`` are returned to the analyst.
    """
    from inference.sandbox import run_code as _run_code

    # ``index_path`` lives next to the descriptor via the registry's
    # default_index_path convention. We reconstruct it here.
    project_root = Path(__file__).resolve().parent.parent
    # ``config.index_dir`` isn't in scope here; callers with a different
    # index location can override via env var, but the default works for
    # the running server.
    index_dir = Path(os.environ.get("HALO_INDEX_DIR", project_root / "data"))
    index_path = ctx.descriptor.default_index_path(index_dir)

    return _run_code(
        code=code,
        descriptor=ctx.descriptor,
        index_path=index_path,
        project_root=project_root,
        timeout_s=timeout_s or 10.0,
    )


def tool_ask_subagent(
    ctx: ToolContext,
    *,
    question: str,
    max_turns: int | None = None,
) -> dict[str, Any]:
    """Spawn a nested ``run_agent`` instance to answer a focused sub-question.

    The sub-agent gets a fresh conversation, fresh result-store, and its own
    tool loop. It reuses the parent's store/reader (same dataset) and
    accumulates cost back into the parent. When the parent is already at
    ``depth == max_depth``, this tool has been pruned from the parent's
    schema — so reaching this function implies a permitted recursion.
    """
    # Import inside the function to avoid a circular import between
    # ``tools.py`` and ``harness.py``.
    from inference.harness import run_agent
    from inference.config import InferenceConfig

    if ctx.depth >= ctx.max_depth:
        # Defensive; the harness should have removed the tool already.
        return {"error": f"max depth {ctx.max_depth} reached — cannot spawn further sub-agents"}

    sub_cfg = InferenceConfig()
    sub_cfg.init()
    sub_cfg.default_dataset_id = ctx.descriptor.id
    sub_cfg.model = ctx.subagent_model
    sub_cfg.synth_model = ctx.synth_model
    sub_cfg.max_turns = max_turns or ctx.subagent_max_turns
    sub_cfg.sample_cap = ctx.sample_cap
    sub_cfg.synth_trace_cap = ctx.synth_trace_cap
    sub_cfg.synth_chars_per_trace = ctx.synth_chars_per_trace
    sub_cfg.max_depth = ctx.max_depth
    sub_cfg.subagent_max_turns = ctx.subagent_max_turns

    events: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    err: str | None = None
    partial_cost = 0.0  # cumulative spend captured on error paths

    try:
        for event in run_agent(
            question, sub_cfg,
            descriptor=ctx.descriptor, store=ctx.store, reader=ctx.reader,
            messages=None,         # fresh conversation
            result_store=None,     # fresh result store — no r_N bleed
            next_result_key=0,
            depth=ctx.depth + 1,
        ):
            events.append({"kind": event.kind, "data": event.data})
            if event.kind == "final":
                final = event.data
            elif event.kind == "error":
                err = event.data.get("message") or "sub-agent errored"
            elif event.kind == "usage":
                # Top-level LLM cost for this turn of the sub-agent.
                partial_cost += float(event.data.get("cost") or 0.0)
            elif event.kind == "tool_result":
                # Sub-agent-internal tool costs — ``synthesize`` runs a
                # secondary LLM and reports ``synth_cost``; a nested
                # ``ask_subagent`` reports ``subagent_cost``. Capture
                # both so error paths don't silently drop the spend
                # those tools incurred via ``ctx.cost``.
                r = (event.data or {}).get("result") or {}
                if isinstance(r, dict):
                    for k in ("synth_cost", "subagent_cost"):
                        v = r.get(k)
                        if isinstance(v, (int, float)):
                            partial_cost += float(v)
    except Exception as e:  # noqa: BLE001
        logger.exception("ask_subagent crashed")
        ctx.cost += partial_cost
        return {"error": f"sub-agent crashed: {e}", "subagent_cost": partial_cost}

    if final is None:
        # Error path: no final event, but turns may have run. Bill the
        # parent for what the sub-agent spent.
        ctx.cost += partial_cost
        return {
            "error": err or "sub-agent produced no final answer",
            "subagent_cost": partial_cost,
        }

    # Roll the sub-agent's cost into the parent's running total so the
    # top-level ``total_cost`` is honest.
    ctx.cost += float(final.get("total_cost") or 0.0)

    return {
        "question": question,
        "answer": final.get("content") or "",
        "subagent_turns": final.get("turns_used"),
        "subagent_tool_calls": final.get("tool_calls_made"),
        "subagent_cost": final.get("total_cost"),
        "depth": ctx.depth + 1,
    }


def tool_inspect_result(ctx: ToolContext, *, key: str) -> dict[str, Any]:
    """Fetch the full JSON of a previously-compacted tool result.

    As a conversation grows, older tool results are collapsed to one-line
    references like ``[compacted · r_3] ...``. Call ``inspect_result(r_3)``
    to retrieve the full JSON that was hidden from the context window.
    """
    stored = ctx.result_store.get(key)
    if stored is None:
        keys = list(ctx.result_store.keys())
        return {
            "error": f"unknown result key {key!r}",
            "available_keys": keys[-20:],
        }
    return {"key": key, "result": stored}


def summarize_tool_result(name: str, result: Any) -> str:
    """One-line textual summary used when compacting older tool messages.

    Mirror of the frontend's ``resultSummary`` so what the LLM sees in
    the compacted message matches what the UI shows.
    """
    if not isinstance(result, dict):
        return str(result)[:160]
    if "error" in result and result["error"]:
        return f"error · {str(result['error'])[:100]}"
    if name == "dataset_overview":
        outcome = result.get("outcome") or {}
        parts = [f"count {result.get('count', '?'):,}"
                 if isinstance(result.get('count'), int) else
                 f"count {result.get('count', '?')}"]
        if outcome:
            parts.append(
                f"{outcome.get('display_name', 'outcome')} μ={outcome.get('mean', '?')}"
            )
            parts.append(
                f"{outcome.get('perfect_count', 0):,} perf / "
                f"{outcome.get('zero_count', 0):,} zero"
            )
        return " · ".join(parts)
    if "traces" in result and isinstance(result["traces"], list):
        return (
            f"returned {result.get('returned', '?')} of "
            f"{result.get('total_matching', '?')}"
        )
    if name == "synthesize" and "answer" in result:
        ans = str(result.get("answer") or "")
        return f"synth over {result.get('traces_used', '?')} traces · {ans[:80]}…"
    if name == "inspect_trace":
        tid = str(result.get("id") or result.get("query_id") or "")[:10]
        rec = result.get("outcome") if result.get("outcome") is not None else result.get("file_recall")
        return (
            f"{tid} · {result.get('outcome_name', 'outcome')} {rec if rec is not None else '–'}"
        )
    if name == "search_trace":
        hits = result.get("matches") or []
        return f"{len(hits)} matches for {result.get('pattern', '')}"
    if name == "compare_groups":
        a = result.get("group_a", {}).get("overview", {}).get("count")
        b = result.get("group_b", {}).get("overview", {}).get("count")
        return f"A: {a} · B: {b}"
    if name == "inspect_result":
        return f"retrieved {result.get('key', '?')}"
    if name == "ask_subagent":
        ans = str(result.get("answer") or "")
        turns = result.get("subagent_turns")
        return f"sub-agent(depth {result.get('depth', '?')}, {turns} turns) · {ans[:80]}…"
    # Fallback: truncate the JSON.
    s = json.dumps(result, ensure_ascii=False, default=str)
    return s[:160] + ("…" if len(s) > 160 else "")


# ------------------------------------------------------------------
# Registry + OpenAI schemas
# ------------------------------------------------------------------

TOOL_FUNCTIONS: dict[str, Any] = {
    "dataset_overview": tool_dataset_overview,
    "find_traces": tool_find_traces,
    "get_trace": tool_get_trace,
    "inspect_trace": tool_inspect_trace,
    "search_trace": tool_search_trace,
    "sample_by_outcome": tool_sample_by_outcome,
    "compare_groups": tool_compare_groups,
    "synthesize": tool_synthesize,
    "inspect_result": tool_inspect_result,
    "run_code": tool_run_code,
    "ask_subagent": tool_ask_subagent,
}


_LABELS_PROP: dict[str, Any] = {
    "type": "object",
    "description": (
        "Filter on categorical labels defined by the dataset (see the system "
        "prompt for this dataset's available labels and their top values). "
        "Keys are label names; values are a string or array of strings to match."
    ),
    "additionalProperties": {
        "oneOf": [
            {"type": "string"},
            {"type": "array", "items": {"type": "string"}},
        ],
    },
}


_COMMON_FILTER_PROPS: dict[str, Any] = {
    "labels": _LABELS_PROP,
    "tool_used": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Match traces that used any of these tool names.",
    },
    "min_outcome": {"type": "number",
                    "description": "Minimum value on the dataset's outcome score (if any)."},
    "max_outcome": {"type": "number",
                    "description": "Maximum value on the dataset's outcome score (if any)."},
    "min_messages": {"type": "integer"},
    "max_messages": {"type": "integer"},
    "min_tokens": {"type": "integer"},
    "max_tokens": {"type": "integer"},
    "min_tool_errors": {"type": "integer"},
    "has_final_answer": {"type": "boolean"},
    "min_max_path_length": {
        "type": "integer",
        "description": "Require the longest document path to be at least this long.",
    },
    "min_nested_paths": {
        "type": "integer",
        "description": "Require at least this many document paths to contain '/'.",
    },
}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "dataset_overview",
            "description": (
                "Dataset-wide distributions and summary stats. Pass filters to compute an "
                "overview over a subset. Start here for most questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "labels": _LABELS_PROP,
                    "tool_used": _COMMON_FILTER_PROPS["tool_used"],
                    "min_max_path_length": _COMMON_FILTER_PROPS["min_max_path_length"],
                    "min_nested_paths": _COMMON_FILTER_PROPS["min_nested_paths"],
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_traces",
            "description": (
                "Filter + sort traces, returning lightweight summary rows. Use this to locate "
                "exemplars — longest traces, traces with low outcome, traces using a specific "
                "tool. Returns at most sample_cap rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    **_COMMON_FILTER_PROPS,
                    "sort_by": {
                        "type": "string",
                        "enum": [
                            "n_messages", "n_tool_calls", "total_tokens", "turns_used",
                            "outcome", "tool_errors", "final_answer_chars",
                            "ground_truth_count", "max_path_length", "nested_path_count",
                        ],
                    },
                    "sort_desc": {"type": "boolean"},
                    "limit": {"type": "integer", "minimum": 1},
                    "random_sample": {
                        "type": "boolean",
                        "description": "Return a random sample instead of a sorted slice.",
                    },
                    "seed": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trace",
            "description": "Fetch the raw trace record for a single id. Expensive — prefer inspect_trace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 20000},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_trace",
            "description": (
                "Structured view of one trace: query, ground truth, outcome, ordered tool "
                "calls + truncated results, and the final_answer. Primary tool for examining "
                "why a specific trace succeeded or failed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "content_chars": {"type": "integer", "default": 600},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_trace",
            "description": "Regex-search the messages and documents of one trace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "pattern": {"type": "string"},
                    "case_insensitive": {"type": "boolean", "default": True},
                    "context": {"type": "integer", "default": 80},
                    "max_matches": {"type": "integer", "default": 20},
                },
                "required": ["id", "pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sample_by_outcome",
            "description": (
                "Shortcut for outcome-bucketed sampling. outcome is one of: "
                "zero, partial, perfect, tool_errors, long, short."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "enum": ["zero", "partial", "perfect", "tool_errors", "long", "short"],
                    },
                    "labels": _LABELS_PROP,
                    "limit": {"type": "integer", "default": 20},
                    "seed": {"type": "integer", "default": 42},
                },
                "required": ["outcome"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_groups",
            "description": (
                "Compute parallel overview() summaries for two filter specs. Each group_* "
                "is a filter spec with the same shape as find_traces filters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_a": {
                        "type": "object",
                        "description": "Filter spec for group A.",
                        "properties": _COMMON_FILTER_PROPS,
                    },
                    "group_b": {
                        "type": "object",
                        "description": "Filter spec for group B.",
                        "properties": _COMMON_FILTER_PROPS,
                    },
                },
                "required": ["group_a", "group_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "synthesize",
            "description": (
                "RLM-style recursive synthesis. Pass a set of trace ids plus a sub-question; "
                "a secondary LLM receives compact renderings of those traces and answers. "
                "Use this when reasoning over a population too large for the top-level context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": {"type": "string"}},
                    "question": {"type": "string"},
                },
                "required": ["ids", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_result",
            "description": (
                "Fetch the full JSON of a previously-compacted tool result by its key "
                "(shown as ``r_N`` in earlier messages that were summarised to save context). "
                "Use this only when you need information that was hidden by compaction; "
                "otherwise prefer re-running the tool with a narrower filter."
            ),
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_subagent",
            "description": (
                "Spawn a nested halo analyst to answer a focused sub-question. "
                "Use this to decompose a multi-faceted question — the sub-agent "
                "gets its own tool loop and returns a grounded one-paragraph "
                "answer. Example: the top-level question 'where is the agent "
                "failing?' can be decomposed into sub-agent calls for 'where "
                "are the hallucinations?' and 'where does path handling break?'. "
                "The sub-agent sees the same dataset but runs a fresh "
                "conversation — frame the question with enough context, and "
                "keep in mind the sub-agent gets up to ``max_turns`` tool "
                "calls (default 8). Sub-agents at the depth limit cannot "
                "spawn further sub-agents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The sub-question to delegate. Include enough context for a fresh agent to ground its investigation.",
                    },
                    "max_turns": {
                        "type": "integer",
                        "description": "Optional override for the sub-agent's turn cap.",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": (
                "Execute arbitrary Python in a sandboxed subprocess with ``store``, "
                "``reader``, and ``descriptor`` pre-loaded. Use this when the other "
                "tools don't express what you need (ad-hoc aggregations, custom "
                "groupings, multi-pass filters). ``print`` output and any value "
                "assigned to ``result``/``out``/``ans`` come back. Timeout: 10s."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source. ``store.filter(...)``, ``store.overview()``, ``reader.read(offset, length)``, etc. are available.",
                    },
                    "timeout_s": {
                        "type": "number",
                        "description": "Optional timeout in seconds (default 10, max enforced by the server).",
                    },
                },
                "required": ["code"],
            },
        },
    },
]
