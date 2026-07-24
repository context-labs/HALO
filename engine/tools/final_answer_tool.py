from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from engine.tools.tool_protocol import ToolContext

FINAL_ANSWER_TOOL_NAME = "final_answer"


class FinalAnswerArguments(BaseModel):
    """Arguments for ``final_answer``: the root agent's complete final answer."""

    model_config = ConfigDict(extra="forbid")

    answer: str


class FinalAnswerResult(BaseModel):
    """Acknowledgement returned to the SDK; the run stops at this tool via ``StopAtTools``."""

    model_config = ConfigDict(extra="forbid")

    acknowledged: bool = True


class FinalAnswerTool:
    """Run-terminating tool the root agent calls to deliver its final answer.

    Finalization used to ride on a text sentinel (``<final/>``) inside the
    root agent's last assistant message — free prose that nothing enforced,
    which some models (DeepSeek-V4-Pro) reliably dropped, ending otherwise
    complete runs without a final-flagged answer. A tool call goes through
    the structured ``tool_calls`` channel instead, and the root SDK agent is
    built with ``tool_use_behavior=StopAtTools([FINAL_ANSWER_TOOL_NAME])``
    so the call itself ends the run.

    The tool body is a no-op acknowledgement: the stop semantics live in the
    root agent's ``tool_use_behavior``, and ``OpenAiEventMapper`` transforms
    the *call* (whose ``answer`` argument carries the final text) into a
    final-flagged plain assistant message for downstream consumers of
    ``AgentOutputItem.final``.

    Registered on the root agent only — subagents return their answers as
    plain messages to their parent.
    """

    name = FINAL_ANSWER_TOOL_NAME
    description = (
        "Deliver your complete final answer and end the run. Call this "
        "exactly once, when your analysis is finished, with the entire "
        "final answer in `answer`, formatted as markdown (headings, bold, "
        "lists) exactly as you would write it in a chat reply. This is the "
        "only way to complete the run."
    )
    arguments_model = FinalAnswerArguments
    result_model = FinalAnswerResult

    async def run(
        self, tool_context: ToolContext, arguments: FinalAnswerArguments
    ) -> FinalAnswerResult:
        """Acknowledge the final answer; termination is handled by ``StopAtTools``."""
        del tool_context, arguments
        return FinalAnswerResult()
