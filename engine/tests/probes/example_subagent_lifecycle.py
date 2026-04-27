"""Probe: subagent lifecycle (unit-style).

The ``FakeRunner`` seam stops at the LLM, so end-to-end ``run_with_fake``
probes cannot drive an actual ``call_subagent`` invocation through the SDK.
The README points at the correct workaround: call
``_build_subagent_as_tool(...).on_invoke_tool(ctx, raw_arguments)`` directly,
with the kit's ``FakeRunner`` installed on ``run_state.runner`` so the
inner ``OpenAiAgentRunner.run`` finds a scripted child stream.

The SDK's normal tool-dispatch path constructs an ``agents.tool_context.ToolContext``
(with ``tool_call_id``, ``tool_name``, ``agent``, etc.) and passes it as the
first argument to ``on_invoke_tool``. Bypassing the dispatcher means the probe
has to construct that ``ToolContext`` itself; ``_make_fake_tool_ctx`` below
shows the minimum keyword args required.

Pathways probed:

  1. ``on_invoke_tool`` invokes the child runner exactly once and returns a
     ``SubagentToolResult`` JSON string carrying the child's agent_id,
     extracted final answer, turns, and tool_calls counts.
  2. The child execution lands in ``state.executions_by_agent_id`` with the
     expected depth, agent_name, and parent linkage (``parent_agent_id`` and
     ``parent_tool_call_id`` taken from the closure-captured parent execution
     and the SDK-supplied ``ctx.tool_call_id``).
  3. The output bus accumulates the child's emitted items at ``depth=1``
     stamped with the parent's ``agent_id``.
  4. Invoking at ``child_depth > maximum_depth`` raises
     ``EngineMaxDepthExceededError`` (the depth guard runs *before* any
     SDK call or ctx access, so a ``None`` ctx is fine here).

Conventions worth stealing from this file:

- Use ``make_run_state(cfg, runner=FakeRunner(...))`` to get a fully wired
  ``EngineRunState`` without going through ``stream_engine_async``.
- Hand-build a parent ``AgentExecution``, ``state.register(...)`` it, AND
  pass it as ``parent_execution`` to ``_build_subagent_as_tool``. The closure
  captures it for ``parent_agent_id`` stamping on every child invocation.
- Construct an ``SdkToolContext`` with a fixed ``tool_call_id`` to drive the
  ``parent_tool_call_id`` linkage path; ``on_invoke_tool(None, ...)`` works
  only on paths that don't read ``ctx.tool_call_id`` (e.g., the depth guard).
- After ``on_invoke_tool`` returns, call ``await state.output_bus.close()`` and
  drain the bus via the public ``stream()`` async-iterator. Do NOT reach into
  ``output_bus._queue`` — close-and-stream is the supported pattern.
- Use ``check_raises`` to assert that the depth guard fires; that helper
  keeps the script clean of try/except boilerplate.
"""

from __future__ import annotations

import asyncio
import json
import sys

from agents.tool_context import ToolContext as SdkToolContext

from engine.agents.agent_execution import AgentExecution
from engine.errors import EngineMaxDepthExceededError
from engine.models.engine_output import AgentOutputItem
from engine.tools.subagent_tool_factory import _build_subagent_as_tool
from tests.probes.probe_kit import (
    FakeRunner,
    check_raises,
    make_assistant_text,
    make_checker,
    make_default_config,
    make_run_state,
)

check, failures = make_checker()


def _make_fake_tool_ctx(tool_call_id: str, tool_arguments: str = "") -> SdkToolContext:
    """Construct the minimum ``SdkToolContext`` the subagent closure reads.

    The real SDK builds this in ``run_internal/tool_actions.py`` from a
    ``RunContextWrapper`` plus the live tool-call metadata. ``guarded_invoke``
    only consults ``ctx.tool_call_id``, so context/usage/agent are nominal.
    """
    return SdkToolContext(
        context=None,
        tool_name="call_subagent",
        tool_call_id=tool_call_id,
        tool_arguments=tool_arguments,
    )


def _register_root(state, agent_id: str = "root-x") -> AgentExecution:
    """Hand-build and register a root execution; returns it for closure capture."""
    root = AgentExecution(
        agent_id=agent_id,
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )
    state.register(root)
    return root


async def _drain_bus(state) -> list:
    """Close the bus and collect every queued event via the public stream."""
    await state.output_bus.close()
    out: list = []
    async for ev in state.output_bus.stream():
        out.append(ev)
    return out


async def probe_invocation_returns_subagent_result_json() -> None:
    """``on_invoke_tool`` returns a JSON-encoded ``SubagentToolResult`` whose
    ``answer`` field is the child's last assistant text (with trailing
    whitespace stripped)."""
    cfg = make_default_config(maximum_depth=1)
    runner = FakeRunner(
        [make_assistant_text("the subagent's reasoned answer\n", item_id="sub-msg-1")],
    )
    state = await make_run_state(cfg, runner=runner)
    root = _register_root(state)

    semaphore = asyncio.Semaphore(cfg.maximum_parallel_subagents)
    subagent_tool = _build_subagent_as_tool(
        run_state=state,
        child_depth=1,
        semaphore=semaphore,
        parent_execution=root,
    )
    ctx = _make_fake_tool_ctx(tool_call_id="parent-call-aaaa", tool_arguments="what is the answer?")
    result_json = await subagent_tool.on_invoke_tool(ctx, "what is the answer?")
    parsed = json.loads(result_json)

    check(
        parsed.get("child_agent_id", "").startswith("sub-"),
        "result: child_agent_id starts with 'sub-' prefix",
        observed=f"child_agent_id={parsed.get('child_agent_id')!r}",
    )
    check(
        parsed.get("answer") == "the subagent's reasoned answer",
        "result: answer is the child's final assistant text (rstripped)",
        observed=f"answer={parsed.get('answer')!r}",
    )
    check(
        len(runner.calls) == 1,
        "result: FakeRunner.run_streamed called exactly once for the child",
        observed=f"calls={len(runner.calls)}",
    )


async def probe_child_execution_registered_with_correct_metadata() -> None:
    """After invocation, exactly one ``sub-*`` execution lives in
    ``state.executions_by_agent_id`` with depth=1, the configured subagent
    name, and parent linkage taken from the closure + ``ctx.tool_call_id``."""
    cfg = make_default_config(maximum_depth=1)
    runner = FakeRunner(
        [make_assistant_text("ok\n", item_id="sub-msg-1")],
    )
    state = await make_run_state(cfg, runner=runner)
    root = _register_root(state, agent_id="root-bbbb")

    semaphore = asyncio.Semaphore(cfg.maximum_parallel_subagents)
    subagent_tool = _build_subagent_as_tool(
        run_state=state,
        child_depth=1,
        semaphore=semaphore,
        parent_execution=root,
    )
    ctx = _make_fake_tool_ctx(tool_call_id="parent-call-bbbb")
    await subagent_tool.on_invoke_tool(ctx, "delegate this")

    children = [ex for aid, ex in state.executions_by_agent_id.items() if aid.startswith("sub-")]
    check(
        len(children) == 1,
        "register: exactly one subagent execution registered",
        observed=f"count={len(children)}",
    )
    if not children:
        return
    child = children[0]
    check(child.depth == 1, "register: child depth = 1", observed=f"depth={child.depth}")
    check(
        child.agent_name == cfg.subagent.name,
        "register: child agent_name matches config.subagent.name",
        observed=f"agent_name={child.agent_name!r} expected={cfg.subagent.name!r}",
    )
    check(
        child.parent_agent_id == "root-bbbb",
        "register: child.parent_agent_id is the closure-captured root agent_id",
        observed=f"parent_agent_id={child.parent_agent_id!r}",
    )
    check(
        child.parent_tool_call_id == "parent-call-bbbb",
        "register: child.parent_tool_call_id is ctx.tool_call_id",
        observed=f"parent_tool_call_id={child.parent_tool_call_id!r}",
    )
    check(
        state.executions_by_tool_call_id.get("parent-call-bbbb") is child,
        "register: state.executions_by_tool_call_id indexes the child by parent's call id",
        observed=f"keys={list(state.executions_by_tool_call_id.keys())}",
    )


async def probe_child_emits_items_at_depth_1() -> None:
    """The child's assistant message should reach the shared output bus
    stamped with depth=1 and the parent's agent_id."""
    cfg = make_default_config(maximum_depth=1)
    runner = FakeRunner(
        [make_assistant_text("subagent reply\n", item_id="sub-msg-1")],
    )
    state = await make_run_state(cfg, runner=runner)
    root = _register_root(state, agent_id="root-cccc")

    semaphore = asyncio.Semaphore(cfg.maximum_parallel_subagents)
    subagent_tool = _build_subagent_as_tool(
        run_state=state,
        child_depth=1,
        semaphore=semaphore,
        parent_execution=root,
    )
    ctx = _make_fake_tool_ctx(tool_call_id="parent-call-cccc")
    await subagent_tool.on_invoke_tool(ctx, "ask the child")
    events = await _drain_bus(state)

    depth_one_items = [ev for ev in events if isinstance(ev, AgentOutputItem) and ev.depth == 1]
    check(
        len(depth_one_items) == 1,
        "emit: exactly one depth=1 AgentOutputItem on the bus",
        observed=f"count={len(depth_one_items)} all={[(type(e).__name__, getattr(e, 'depth', None)) for e in events]}",
    )
    if not depth_one_items:
        return
    item = depth_one_items[0]
    check(
        item.item.role == "assistant" and "subagent reply" in (item.item.content or ""),
        "emit: depth=1 item is the assistant's reply",
        observed=f"role={item.item.role} content={item.item.content!r}",
    )
    check(
        item.parent_agent_id == "root-cccc",
        "emit: depth=1 item carries parent_agent_id == root agent id",
        observed=f"parent_agent_id={item.parent_agent_id!r}",
    )
    check(
        item.parent_tool_call_id == "parent-call-cccc",
        "emit: depth=1 item carries parent_tool_call_id == ctx.tool_call_id",
        observed=f"parent_tool_call_id={item.parent_tool_call_id!r}",
    )


async def probe_depth_guard_raises_before_any_sdk_call() -> None:
    """Constructing the tool at ``child_depth=2`` against ``maximum_depth=1``
    is fine; *invoking* it must raise before the inner runner or ctx access.
    Passing ``ctx=None`` here is intentional — the depth guard fires before
    ``ctx.tool_call_id`` is read."""
    cfg = make_default_config(maximum_depth=1)
    runner = FakeRunner(
        [make_assistant_text("never reached\n", item_id="x")],
    )
    state = await make_run_state(cfg, runner=runner)
    root = _register_root(state, agent_id="root-dddd")

    semaphore = asyncio.Semaphore(cfg.maximum_parallel_subagents)
    over_depth_tool = _build_subagent_as_tool(
        run_state=state,
        child_depth=2,
        semaphore=semaphore,
        parent_execution=root,
    )

    exc = await check_raises(
        lambda: over_depth_tool.on_invoke_tool(None, "should not run"),
        EngineMaxDepthExceededError,
    )
    check(
        exc is not None,
        "depth-guard: invoking child_depth > maximum_depth raises EngineMaxDepthExceededError",
        observed=f"got={type(exc).__name__ if exc else 'no raise'}",
    )
    check(
        len(runner.calls) == 0,
        "depth-guard: no FakeRunner call was consumed",
        observed=f"calls={len(runner.calls)}",
    )


async def main() -> int:
    await probe_invocation_returns_subagent_result_json()
    await probe_child_execution_registered_with_correct_metadata()
    await probe_child_emits_items_at_depth_1()
    await probe_depth_guard_raises_before_any_sdk_call()
    return failures.report_and_exit_code()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
