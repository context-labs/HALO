from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentExecution:
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
        self.consecutive_llm_failures += 1

    def record_llm_success(self) -> None:
        self.consecutive_llm_failures = 0
