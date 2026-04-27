from __future__ import annotations

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from engine.agents.prompt_templates import SYNTHESIS_SYSTEM_PROMPT
from engine.tools.tool_protocol import ToolContext


class SynthesizeTracesArguments(BaseModel):
    """Arguments for ``synthesize_traces``: trace ids to summarize plus an optional focus directive."""

    model_config = ConfigDict(extra="forbid")

    trace_ids: list[str]
    focus: str | None = None


class SynthesizeTracesResult(BaseModel):
    """Plain-text synthesis of the selected traces, suitable for the calling model to consume."""

    model_config = ConfigDict(extra="forbid")

    summary: str


class SynthesisTool:
    """LLM-backed tool that fans rendered trace text through ``synthesis_model`` for a short summary.

    Uses ``TraceStore.render_trace`` for each id (with a per-trace budget) so the
    synthesis call stays within a reasonable prompt size even on large traces.
    """

    name = "synthesize_traces"
    description = "Summarize findings across a set of traces."
    arguments_model = SynthesizeTracesArguments
    result_model = SynthesizeTracesResult

    def __init__(self, model_name: str, client: AsyncOpenAI | None = None) -> None:
        """``client`` is injectable for tests; defaults to a fresh ``AsyncOpenAI``."""
        self._model_name = model_name
        self._client = client

    async def run(self, tool_context: ToolContext, arguments: SynthesizeTracesArguments) -> SynthesizeTracesResult:
        """Render each trace, prepend the focus hint if set, and return the synthesis model's plain-text reply."""
        user_text_parts = [f"trace_ids: {', '.join(arguments.trace_ids)}"]
        if arguments.focus:
            user_text_parts.append(f"focus: {arguments.focus}")

        store = tool_context.require_trace_store()
        for tid in arguments.trace_ids:
            rendered = store.render_trace(tid, budget=8_000)
            user_text_parts.append(rendered)

        if self._client is None:
            self._client = AsyncOpenAI()

        response = await self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": "\n\n".join(user_text_parts)},
            ],
        )
        summary = (response.choices[0].message.content or "").strip()
        return SynthesizeTracesResult(summary=summary)
