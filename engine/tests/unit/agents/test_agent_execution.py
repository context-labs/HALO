from __future__ import annotations

from engine.agents.agent_execution import AgentExecution


def test_agent_execution_defaults() -> None:
    execution = AgentExecution(
        agent_id="root",
        agent_name="root",
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )
    assert execution.consecutive_llm_failures == 0
    assert execution.tool_calls_made == 0
    assert execution.turns_used == 0


def test_record_and_reset_failures() -> None:
    execution = AgentExecution(
        agent_id="a", agent_name="a", depth=0,
        parent_agent_id=None, parent_tool_call_id=None,
    )
    execution.record_llm_failure()
    execution.record_llm_failure()
    assert execution.consecutive_llm_failures == 2
    execution.record_llm_success()
    assert execution.consecutive_llm_failures == 0
