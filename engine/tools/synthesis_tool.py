from __future__ import annotations

from openai import AsyncOpenAI, omit
from pydantic import BaseModel, ConfigDict

from engine.agents.llm_retry import call_with_retries
from engine.agents.prompt_caching import as_cached_system_message
from engine.agents.prompt_templates import SYNTHESIS_SYSTEM_PROMPT
from engine.model_config import ModelConfig
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

    def __init__(
        self,
        model: ModelConfig,
        client: AsyncOpenAI,
    ) -> None:
        self._model = model
        self._client = client

    async def run(
        self, tool_context: ToolContext, arguments: SynthesizeTracesArguments
    ) -> SynthesizeTracesResult:
        """Render each trace, prepend the focus hint if set, and return the synthesis model's plain-text reply.

        The system message is built with :func:`as_cached_system_message`
        so the byte-stable :data:`SYNTHESIS_SYSTEM_PROMPT` prefix is
        cacheable on Anthropic (read-back surfaces as
        ``cache_read_input_tokens``) without affecting OpenAI's automatic
        prefix cache. The per-call dynamic trace text goes in the trailing
        ``user`` message to keep the prefix byte-stable across calls.
        """
        user_text_parts = [f"trace_ids: {', '.join(arguments.trace_ids)}"]
        if arguments.focus:
            user_text_parts.append(f"focus: {arguments.focus}")

        store = tool_context.require_trace_store()
        for tid in arguments.trace_ids:
            rendered = store.render_trace(tid, budget=8_000)
            user_text_parts.append(rendered)

        response = await call_with_retries(
            lambda: self._client.chat.completions.create(
                model=self._model.name,
                messages=[
                    as_cached_system_message(SYNTHESIS_SYSTEM_PROMPT),
                    {"role": "user", "content": "\n\n".join(user_text_parts)},
                ],
                reasoning_effort=self._model.effective_reasoning_effort() or omit,
            ),
            description=f"synthesize_traces call (model={self._model.name})",
        )
        summary = (response.choices[0].message.content or "").strip()
        return SynthesizeTracesResult(summary=summary)
