"""Probe: depth enforcement via tool-list construction.

Pathways probed:
  1. ``maximum_depth=0`` → root SDK agent has NO ``call_subagent`` tool.
  2. ``maximum_depth=1`` → root SDK agent HAS ``call_subagent`` tool.
  3. ``maximum_depth=2`` → root has ``call_subagent``; depth-2 leaf-only
     tool list (no further subagent tool) — verified by calling
     ``_child_tools_for_depth`` directly at depth=2 with maximum_depth=2.
  4. End-to-end: with ``maximum_depth=0``, the input the SDK sees does NOT
     advertise ``call_subagent`` as an available tool name.

This probe inspects the SDK ``Agent`` object that ``stream_engine_async``
hands to ``runner.run_streamed`` — that's our visibility into "what tools
were registered for this depth".
"""

from __future__ import annotations

import asyncio
import sys

from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.engine_run_state import EngineRunState
from engine.sandbox.sandbox_availability import SandboxStatus, SandboxUnavailableReason
from engine.tools.subagent_tool_factory import (
    _child_tools_for_depth,
    build_root_sdk_agent,
)
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore
from tests.probes.probe_kit import (
    FakeRunner,
    isolated_trace_copy,
    make_assistant_text,
    make_default_config,
    run_with_fake,
)

_FAILURES: list[str] = []


def _check(condition: bool, description: str, observed: str = "") -> None:
    if condition:
        print(f"PASS: {description}")
    else:
        suffix = f" — observed: {observed}" if observed else ""
        print(f"FAIL: {description}{suffix}")
        _FAILURES.append(description)


async def _build_run_state(cfg) -> EngineRunState:
    """Build a minimal run_state for direct tool-construction probes."""
    trace_path = isolated_trace_copy()
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=cfg.trace_index,
    )
    trace_store = TraceStore.load(trace_path=trace_path, index_path=index_path)
    return EngineRunState(
        trace_store=trace_store,
        output_bus=EngineOutputBus(),
        config=cfg,
        sandbox_status=SandboxStatus.unavailable(
            reason=SandboxUnavailableReason.UNSUPPORTED_PLATFORM,
            diagnostic="probe never exercises the real sandbox",
            remediation="not applicable in probes",
        ),
        runtime_mounts=None,
        runner=FakeRunner(),
    )


def _tool_names(tools) -> list[str]:
    return [t.name for t in tools]


async def probe_depth_zero_no_subagent_tool() -> None:
    """maximum_depth=0 means subagents are disabled — root has no
    call_subagent tool."""
    cfg = make_default_config(maximum_depth=0)
    run_state = await _build_run_state(cfg)
    root_exec = AgentExecution(
        agent_id="root-x",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )
    sdk_agent = build_root_sdk_agent(
        engine_config=cfg,
        run_state=run_state,
        agent_execution=root_exec,
    )
    names = _tool_names(sdk_agent.tools)
    _check(
        "call_subagent" not in names,
        "depth=0: root has NO call_subagent tool",
        observed=f"tools={names}",
    )
    _check(len(names) > 0, "depth=0: root still has leaf tools", observed=f"tools={names}")


async def probe_depth_one_root_has_subagent_tool() -> None:
    """maximum_depth=1 means root can dispatch to depth-1 subagents."""
    cfg = make_default_config(maximum_depth=1)
    run_state = await _build_run_state(cfg)
    root_exec = AgentExecution(
        agent_id="root-x",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )
    sdk_agent = build_root_sdk_agent(
        engine_config=cfg,
        run_state=run_state,
        agent_execution=root_exec,
    )
    names = _tool_names(sdk_agent.tools)
    _check(
        "call_subagent" in names, "depth=1: root has call_subagent tool", observed=f"tools={names}"
    )


async def probe_depth_one_subagent_has_no_subagent_tool() -> None:
    """At depth=1 with maximum_depth=1, the subagent's own tool list
    should NOT include another call_subagent (no nested delegation)."""
    cfg = make_default_config(maximum_depth=1)
    run_state = await _build_run_state(cfg)
    semaphore = {
        d: asyncio.Semaphore(cfg.maximum_parallel_subagents)
        for d in range(1, cfg.maximum_depth + 1)
    }
    parent = AgentExecution(
        agent_id="parent-x",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )
    depth1_tools = _child_tools_for_depth(
        depth=1,
        run_state=run_state,
        semaphores_by_depth=semaphore,
        parent_execution=parent,
    )
    names = _tool_names(depth1_tools)
    _check(
        "call_subagent" not in names,
        "depth=1 child: depth-1 subagent has NO call_subagent",
        observed=f"tools={names}",
    )


async def probe_depth_two_intermediate_has_subagent_tool() -> None:
    """With maximum_depth=2, depth-1 subagent should have call_subagent
    (it can spawn depth-2), but depth-2 should not."""
    cfg = make_default_config(maximum_depth=2)
    run_state = await _build_run_state(cfg)
    semaphore = {
        d: asyncio.Semaphore(cfg.maximum_parallel_subagents)
        for d in range(1, cfg.maximum_depth + 1)
    }
    parent = AgentExecution(
        agent_id="parent-x",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )
    d1 = _tool_names(
        _child_tools_for_depth(
            depth=1, run_state=run_state, semaphores_by_depth=semaphore, parent_execution=parent
        )
    )
    d2 = _tool_names(
        _child_tools_for_depth(
            depth=2, run_state=run_state, semaphores_by_depth=semaphore, parent_execution=parent
        )
    )
    _check(
        "call_subagent" in d1, "depth=2 cfg: depth-1 has call_subagent", observed=f"d1_tools={d1}"
    )
    _check(
        "call_subagent" not in d2,
        "depth=2 cfg: depth-2 leaf has NO call_subagent",
        observed=f"d2_tools={d2}",
    )


async def probe_end_to_end_depth_zero_run() -> None:
    """End-to-end: configure maximum_depth=0 and run the engine. Inspect the
    SDK Agent passed into ``runner.run_streamed`` to confirm subagent tool
    is absent at runtime (not just in unit-construction)."""
    cfg = make_default_config(maximum_depth=0)
    runner = FakeRunner(
        [make_assistant_text("done\n<final/>", item_id="m1")],
    )
    result = await run_with_fake(runner, config=cfg)
    _check(
        result.error is None,
        "e2e-depth=0: completes without error",
        observed=f"error={type(result.error).__name__ if result.error else None}",
    )
    _check(
        len(runner.calls) == 1,
        "e2e-depth=0: runner called exactly once (no subagent path)",
        observed=f"calls={len(runner.calls)}",
    )
    if runner.calls:
        sdk_agent = runner.calls[0]["starting_agent"]
        names = _tool_names(sdk_agent.tools)
        _check(
            "call_subagent" not in names,
            "e2e-depth=0: SDK agent at runtime has NO call_subagent",
            observed=f"tools={names}",
        )


async def main() -> int:
    await probe_depth_zero_no_subagent_tool()
    await probe_depth_one_root_has_subagent_tool()
    await probe_depth_one_subagent_has_no_subagent_tool()
    await probe_depth_two_intermediate_has_subagent_tool()
    await probe_end_to_end_depth_zero_run()

    if _FAILURES:
        print(f"\n{len(_FAILURES)} check(s) failed:")
        for desc in _FAILURES:
            print(f"  - {desc}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
