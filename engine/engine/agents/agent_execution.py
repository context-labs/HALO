from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentExecution:
    """Per-agent run summary tracked across the lifetime of one Engine agent invocation.

    Lineage fields (agent_id, parent_agent_id, parent_tool_call_id, depth) place the
    execution in the parent/child tree. The counters drive the consecutive-failure
    circuit breaker and feed the SubagentToolResult returned to the parent.
    """

    agent_id: str
    agent_name: str
    depth: int
    parent_agent_id: str | None
    parent_tool_call_id: str | None
    consecutive_llm_failures: int = 0
    tool_calls_made: int = 0
    turns_used: int = 0
    output_start_sequence: int | None = None
    output_end_sequence: int | None = None

    def record_llm_failure(self) -> None:
        """Increment the consecutive-failure counter that feeds the runner circuit breaker."""
        self.consecutive_llm_failures += 1

    def record_llm_success(self) -> None:
        """Reset the consecutive-failure counter after any successful LLM call."""
        self.consecutive_llm_failures = 0
