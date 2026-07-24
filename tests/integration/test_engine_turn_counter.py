"""Integration: engine forwards a TurnCounterInputFilter to the SDK runner.

Verifies the wiring (root and subagent both build a fresh per-execution
filter and pass it through ``RunConfig``). The filter's per-call behavior
is covered by ``tests/unit/agents/test_turn_counter.py``; here we just
check the engine constructs and forwards the right object.

``FakeRunner`` does not actually invoke ``call_model_input_filter`` —
that's behavior of the real SDK Runner — so we cannot assert nudges hit
the wire from a probe. The live e2e test in ``tests/e2e/`` does that.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from agents import RunConfig
from agents.tool_context import ToolContext as SdkToolContext

from engine.agents.agent_execution import AgentExecution
from engine.agents.turn_counter import TurnCounterInputFilter
from engine.tools.subagent_tool_factory import _build_subagent_as_tool
from tests.probes.probe_kit import (
    FakeRunner,
    install_fake_runner,
    make_assistant_text,
    make_default_config,
    make_final_answer,
    make_run_state,
    run_with_fake,
)


@pytest.mark.asyncio
async def test_root_agent_run_config_has_turn_counter_filter() -> None:
    """The engine must pass a RunConfig whose call_model_input_filter is a
    TurnCounterInputFilter sized to root_agent.maximum_turns."""
    runner = FakeRunner([*make_final_answer("done")])
    result = await run_with_fake(runner)

    assert result.error is None, type(result.error).__name__
    assert len(runner.calls) == 1, "expected exactly one runner.run_streamed call"

    call = runner.calls[0]
    run_config = call.get("run_config")
    assert isinstance(run_config, RunConfig), f"expected RunConfig, got {type(run_config).__name__}"

    filter_obj = run_config.call_model_input_filter
    assert isinstance(filter_obj, TurnCounterInputFilter), (
        f"expected TurnCounterInputFilter, got {type(filter_obj).__name__}"
    )
    assert filter_obj.max_turns == call["max_turns"]
    assert filter_obj.is_root is True


@pytest.mark.asyncio
async def test_subagent_gets_its_own_filter_instance() -> None:
    """The subagent path must construct a fresh per-execution filter sized to
    ``subagent.maximum_turns`` and forward it via ``RunConfig``. We invoke
    ``_build_subagent_as_tool().on_invoke_tool`` directly because FakeRunner
    does not dispatch SDK FunctionTools — see tests/probes/example_subagent_lifecycle.py."""
    cfg = make_default_config(maximum_depth=1)
    runner = FakeRunner([make_assistant_text("sub answered\n", item_id="sub-msg-1")])
    state = await make_run_state(cfg)

    root = AgentExecution(
        agent_id="root-x",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )
    state.register(root)

    semaphores = {
        d: asyncio.Semaphore(cfg.maximum_parallel_subagents)
        for d in range(1, cfg.maximum_depth + 1)
    }
    subagent_tool = _build_subagent_as_tool(
        run_state=state,
        child_depth=1,
        semaphores_by_depth=semaphores,
        parent_execution=root,
    )

    raw_args = json.dumps({"input": "do the subtask"})
    ctx = SdkToolContext(
        context=None,
        tool_name="call_subagent",
        tool_call_id="parent-call-1",
        tool_arguments=raw_args,
    )
    with install_fake_runner(runner):
        await subagent_tool.on_invoke_tool(ctx, raw_args)

    assert len(runner.calls) == 1, f"expected 1 subagent runner call; got {len(runner.calls)}"
    sub_call = runner.calls[0]
    sub_run_config = sub_call.get("run_config")
    assert sub_run_config is not None, "subagent must forward a RunConfig"

    sub_filter = sub_run_config.call_model_input_filter
    assert isinstance(sub_filter, TurnCounterInputFilter)
    assert sub_filter.is_root is False
    assert sub_filter.max_turns == cfg.subagent.maximum_turns


@pytest.mark.asyncio
async def test_root_and_subagent_filters_are_distinct_instances() -> None:
    """Per-execution isolation: root and subagent must each get their own
    filter instance — a parent's counter must not leak into a child."""
    cfg = make_default_config(maximum_depth=1)

    root_runner = FakeRunner([*make_final_answer("root done")])
    root_result = await run_with_fake(root_runner, config=cfg)
    assert root_result.error is None, type(root_result.error).__name__
    root_filter = root_runner.calls[0]["run_config"].call_model_input_filter

    sub_runner = FakeRunner([make_assistant_text("sub done\n", item_id="sub-msg")])
    state = await make_run_state(cfg)
    root_exec = AgentExecution(
        agent_id="root-y",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )
    state.register(root_exec)
    semaphores = {
        d: asyncio.Semaphore(cfg.maximum_parallel_subagents)
        for d in range(1, cfg.maximum_depth + 1)
    }
    subagent_tool = _build_subagent_as_tool(
        run_state=state,
        child_depth=1,
        semaphores_by_depth=semaphores,
        parent_execution=root_exec,
    )
    raw_args = json.dumps({"input": "child task"})
    ctx = SdkToolContext(
        context=None,
        tool_name="call_subagent",
        tool_call_id="parent-call-2",
        tool_arguments=raw_args,
    )
    with install_fake_runner(sub_runner):
        await subagent_tool.on_invoke_tool(ctx, raw_args)
    sub_filter = sub_runner.calls[0]["run_config"].call_model_input_filter

    assert isinstance(root_filter, TurnCounterInputFilter)
    assert isinstance(sub_filter, TurnCounterInputFilter)
    assert root_filter is not sub_filter, "root and subagent must have distinct filter instances"
    assert root_filter.is_root is True
    assert sub_filter.is_root is False
    assert root_filter.max_turns == cfg.root_agent.maximum_turns
    assert sub_filter.max_turns == cfg.subagent.maximum_turns
