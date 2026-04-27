from __future__ import annotations

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from engine.agents.prompt_templates import SYNTHESIS_SYSTEM_PROMPT
from engine.tools.tool_protocol import ToolContext


class SynthesizeTracesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_ids: list[str]
    focus: str | None = None


class SynthesizeTracesResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str


class SynthesisTool:
    name = "synthesize_traces"
    description = "Summarize findings across a set of traces."
    arguments_model = SynthesizeTracesArguments
    result_model = SynthesizeTracesResult

    def __init__(self, model_name: str, client: AsyncOpenAI | None = None) -> None:
        self._model_name = model_name
        self._client = client or AsyncOpenAI()

    async def run(self, tool_context: ToolContext, arguments: SynthesizeTracesArguments) -> SynthesizeTracesResult:
        user_text_parts = [f"trace_ids: {', '.join(arguments.trace_ids)}"]
        if arguments.focus:
            user_text_parts.append(f"focus: {arguments.focus}")

        store = tool_context.require_trace_store()
        for tid in arguments.trace_ids:
            rendered = store.render_trace(tid, budget=8_000)
            user_text_parts.append(rendered)

        response = await self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": "\n\n".join(user_text_parts)},
            ],
        )
        summary = (response.choices[0].message.content or "").strip()
        return SynthesizeTracesResult(summary=summary)
