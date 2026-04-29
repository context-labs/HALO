"""Probe helpers: FakeRunner, event builders, run wrapper.

This module is the entire vocabulary a probe script needs. Compose these
primitives; do not reach into engine internals or use third-party mocking
libraries.

The seam is ``RunnerProtocol`` (see ``engine.agents.runner_protocol``).
``FakeRunner`` implements it: each call to ``run_streamed`` consumes one
scripted *program* — either a list of SDK-shaped events to yield, or an
exception to raise. ``OpenAiAgentRunner`` then drives the engine through
those events as if they came from the real SDK.

LIMITATION: ``FakeRunner`` does not invoke registered ``FunctionTool``s in
response to tool_call events — that's behavior of the real SDK Runner. To
test "the engine routed the tool result correctly" you script the
``tool_call_output_item`` event directly. To test "tool function X was
invoked with arguments Y" you call the tool function directly outside the
engine.
"""

from __future__ import annotations

import asyncio
import atexit
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent

from engine.agents.agent_config import AgentConfig
from engine.agents.agent_context import AgentContext
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.engine_run_state import EngineRunState
from engine.agents.runner_protocol import RunnerProtocol
from engine.engine_config import EngineConfig
from engine.main import stream_engine_async
from engine.model_config import ModelConfig
from engine.models.engine_output import (
    AgentOutputItem,
    AgentTextDelta,
    EngineStreamEvent,
)
from engine.models.messages import AgentMessage
from engine.tools.tool_protocol import ToolContext
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore
from tests._sdk_events import (
    assistant_message_event,
    text_delta_event,
    tool_call_event,
    tool_output_event,
)

# --- Fixture paths -----------------------------------------------------------

PROBES_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = PROBES_DIR.parent / "fixtures"
TINY_TRACES_PATH = FIXTURES_DIR / "tiny_traces.jsonl"
REALISTIC_TRACES_PATH = FIXTURES_DIR / "realistic_traces.jsonl"


# --- FakeRunner --------------------------------------------------------------


class _FakeStream:
    """Async-iterable stream returned by ``FakeRunner.run_streamed``.

    Mirrors the shape of ``RunResultStreaming`` used by the engine:
    only ``stream_events()`` is consumed by ``OpenAiAgentRunner``.
    """

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def stream_events(self):
        for event in self._events:
            yield event


class FakeRunner:
    """Scriptable runner. Each invocation of ``run_streamed`` pops one program
    from ``programs`` (FIFO) and either yields its events or raises its
    exception.

    Pass either:
      - ``list[event]`` — the runner yields these events from ``stream_events``
      - ``BaseException`` instance — the runner raises it from ``run_streamed``

    ``calls`` records every invocation for after-the-fact assertions.
    """

    def __init__(self, *programs: list[Any] | BaseException) -> None:
        self._programs: list[list[Any] | BaseException] = list(programs)
        self.calls: list[dict[str, Any]] = []

    def run_streamed(
        self,
        *,
        starting_agent: Any,
        input: Any,
        context: Any = None,
        **kwargs: Any,
    ) -> Any:
        # Record every kwarg the engine forwards. Probes inspect
        # `runner.calls[i]` to verify the engine passed what it should
        # have (e.g. max_turns from AgentConfig).
        self.calls.append(
            {"starting_agent": starting_agent, "input": input, "context": context, **kwargs}
        )
        if not self._programs:
            raise RuntimeError("FakeRunner exhausted; called more times than programs supplied")
        program = self._programs.pop(0)
        if isinstance(program, BaseException):
            raise program
        return _FakeStream(program)


# Static type-check that FakeRunner satisfies RunnerProtocol.
_: RunnerProtocol = FakeRunner()  # type: ignore[assignment]


# --- Event builders ----------------------------------------------------------
# Thin wrappers over the shared real-SDK factories in ``tests/_sdk_events.py``.
# The mapper now dispatches via ``isinstance`` against ``RunItemStreamEvent`` /
# ``RawResponsesStreamEvent`` and the per-item subclasses, so the SimpleNamespace
# duck types these used to return would silently fall through to ``MappedEvent()``
# and every probe event would be dropped. Keeping the public signatures stable
# means existing probes continue to compile unchanged.


def make_assistant_text(
    text: str,
    *,
    item_id: str = "msg-1",
) -> RunItemStreamEvent:
    """Build a ``message_output_item`` event yielding ``text`` as assistant
    content. Use this for the model's natural-language replies, including
    those carrying the ``<final/>`` sentinel for the root agent."""
    return assistant_message_event(item_id=item_id, text=text)


def make_tool_call(
    *,
    name: str,
    arguments: str,
    call_id: str = "call-1",
    item_id: str | None = None,
) -> RunItemStreamEvent:
    """Build a ``tool_call_item`` event for an LLM-issued function call.
    ``arguments`` is the raw JSON string the model produced (validate-it-or-not
    is up to the engine)."""
    return tool_call_event(
        call_id=call_id,
        name=name,
        arguments=arguments,
        raw_id=item_id or call_id,
    )


def make_tool_output(
    *,
    call_id: str,
    output: str,
    name: str | None = None,  # noqa: ARG001 — name lives on the assistant tool_calls block, not the output item
    item_id: str | None = None,
) -> RunItemStreamEvent:
    """Build a ``tool_call_output_item`` event for the *result* of a tool call.
    Use this to simulate what the SDK would emit after invoking a tool — the
    FakeRunner does NOT run real tool functions.

    ``name`` is accepted for backwards compatibility but ignored: the
    Responses-API ``FunctionCallOutput`` has no ``name`` field, and the
    mapper now correlates names by ``call_id`` from the preceding
    ``ToolCallItem``. Probes that need the result message to carry a
    ``name`` should emit a matching ``make_tool_call`` first."""
    return tool_output_event(
        call_id=call_id,
        output=output,
        raw_id=item_id or call_id,
    )


def make_text_delta(*, item_id: str, delta: str) -> RawResponsesStreamEvent:
    """Build a ``raw_response_event`` with a streaming text delta. The engine
    forwards these to the bus as ``AgentTextDelta`` events but does NOT add
    them to ``AgentContext``."""
    return text_delta_event(item_id=item_id, delta=delta)


# --- Config + message helpers ------------------------------------------------


def make_default_config(
    *,
    maximum_depth: int = 0,
    maximum_parallel_subagents: int = 2,
    text_message_compaction_keep_last_messages: int = 12,
    tool_call_compaction_keep_last_turns: int = 3,
    model: str = "gpt-5.4-mini",
) -> EngineConfig:
    """Sensible defaults for an EngineConfig used in probes.
    The model name is irrelevant when ``runner=FakeRunner`` is injected
    (no real LLM call happens), so any string works."""
    agent = AgentConfig(
        name="root",
        instructions="Answer briefly.",
        model=ModelConfig(name=model),
        maximum_turns=4,
    )
    return EngineConfig(
        root_agent=agent,
        subagent=agent.model_copy(update={"name": "sub"}),
        synthesis_model=ModelConfig(name=model),
        compaction_model=ModelConfig(name=model),
        maximum_depth=maximum_depth,
        maximum_parallel_subagents=maximum_parallel_subagents,
        text_message_compaction_keep_last_messages=text_message_compaction_keep_last_messages,
        tool_call_compaction_keep_last_turns=tool_call_compaction_keep_last_turns,
    )


def make_default_messages(content: str = "Probe.") -> list[AgentMessage]:
    """One user message; the engine prepends its own root system prompt."""
    return [AgentMessage(role="user", content=content)]


def make_root_context(
    cfg: EngineConfig,
    *,
    messages: list[AgentMessage] | None = None,
) -> AgentContext:
    """Build a root ``AgentContext`` the way ``stream_engine_async`` does.

    Probes that bypass ``stream_engine_async`` to call ``build_root_sdk_agent``
    or ``_child_tools_for_depth`` directly need an ``AgentContext`` to pass
    through. Going through ``AgentContext.from_input_messages`` keeps the
    probe consistent with production: same compaction settings, same root
    system-prompt prepending. Defaults to ``make_default_messages()``.
    """
    return AgentContext.from_input_messages(
        messages=messages or make_default_messages(),
        engine_config=cfg,
    )


# --- Trace fixture isolation -------------------------------------------------


def isolated_trace_copy(
    source: Path | None = None,
    *,
    cleanup_on_exit: bool = True,
) -> Path:
    """Copy a trace fixture into a tempdir so ``TraceIndexBuilder`` writes its
    sidecar index there instead of polluting ``tests/fixtures/``.

    Default source is ``tiny_traces.jsonl``. Use ``REALISTIC_TRACES_PATH``
    when probing pathways that need TOOL/CHAIN spans or richer content.

    Returns the copied path. If ``cleanup_on_exit`` is True (default), the
    tempdir is removed when the Python process exits.
    """
    src = source or TINY_TRACES_PATH
    tmp_dir = Path(tempfile.mkdtemp(prefix="halo-probe-"))
    dst = tmp_dir / src.name
    shutil.copy(src, dst)
    if cleanup_on_exit:
        atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
    return dst


# --- EngineRunState builder --------------------------------------------------


async def make_run_state(
    config: EngineConfig | None = None,
    *,
    trace_path: Path | None = None,
    runner: RunnerProtocol | None = None,
) -> EngineRunState:
    """Build a fully-loaded ``EngineRunState`` *without* running the engine.

    Use this when a probe needs to call internals directly — e.g.
    ``_child_tools_for_depth`` for depth-enforcement probes, or
    ``_build_subagent_as_tool().on_invoke_tool`` for unit-style subagent
    probes. The state has a real ``TraceStore`` backed by an isolated trace
    copy, so methods that touch the store work normally.

    Defaults: ``config = make_default_config()``,
    ``trace_path = isolated_trace_copy()``. If ``runner`` is given it is
    installed on the state; otherwise the dataclass default (``agents.Runner``)
    stays.
    """
    cfg = config or make_default_config()
    tp = trace_path or isolated_trace_copy()
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=tp,
        config=cfg.trace_index,
    )
    store = TraceStore.load(trace_path=tp, index_path=index_path)
    state = EngineRunState(
        trace_store=store,
        output_bus=EngineOutputBus(),
        config=cfg,
        sandbox=None,
    )
    if runner is not None:
        state.runner = runner
    return state


# --- ToolContext helper ------------------------------------------------------


def make_tool_context(
    state: EngineRunState,
    *,
    agent_context: Any | None = None,
    agent_execution: Any | None = None,
) -> ToolContext:
    """Construct a ``ToolContext`` from an ``EngineRunState``.

    Use this in unit-style tool probes — when you've built a state via
    ``make_run_state`` and want to call ``Tool.run(tool_context, args)``
    directly. ``agent_context`` and ``agent_execution`` are optional
    because most trace tools don't need them, but probes for
    ``GetContextItemTool`` (and other agent-scoped tools) must pass them.
    """
    return ToolContext.model_construct(
        run_state=state,
        trace_store=state.trace_store,
        output_bus=state.output_bus,
        agent_context=agent_context,
        agent_execution=agent_execution,
    )


# --- Checker helper ----------------------------------------------------------


class Checker:
    """Bundles the ``_check`` / ``_FAILURES`` / ``main()`` pattern that every
    probe duplicates. Usage::

        from tests.probes.probe_kit import make_checker

        check, failures = make_checker()

        async def probe_x():
            ...
            check(condition, "x: description", observed=f"actual={actual}")

        async def main() -> int:
            await probe_x()
            return failures.report_and_exit_code()

    ``failures.report_and_exit_code()`` prints the failure summary block (or
    "All checks passed.") and returns 0/1 suitable for ``sys.exit``.
    """

    def __init__(self) -> None:
        self.descriptions: list[str] = []

    def __call__(self, condition: bool, description: str, observed: str = "") -> None:
        if condition:
            print(f"PASS: {description}")
        else:
            suffix = f" — observed: {observed}" if observed else ""
            print(f"FAIL: {description}{suffix}")
            self.descriptions.append(description)

    def report_and_exit_code(self) -> int:
        if self.descriptions:
            print(f"\n{len(self.descriptions)} check(s) failed:")
            for desc in self.descriptions:
                print(f"  - {desc}")
            return 1
        print("\nAll checks passed.")
        return 0


def make_checker() -> tuple[Checker, Checker]:
    """Returns ``(check, failures)`` — both are the same ``Checker`` object,
    aliased so probe scripts can read naturally::

        check, failures = make_checker()
        check(condition, "...", observed="...")
        return failures.report_and_exit_code()
    """
    c = Checker()
    return c, c


# --- Expected-raise helper ---------------------------------------------------


async def check_raises(
    awaitable_or_callable: Any,
    expected_type: type[BaseException],
) -> BaseException | None:
    """Invoke ``awaitable_or_callable`` and assert it raises an instance of
    ``expected_type``. Returns the caught exception (for further inspection)
    or ``None`` if no exception was raised.

    Use this when a probe's *expected* outcome is a specific exception type.
    Distinct from the README's "do not wrap engine calls in try/except" rule
    — that rule forbids hiding *unexpected* failures, not asserting *expected*
    raises. Example::

        exc = await check_raises(lambda: store.view_trace("nope"), KeyError)
        check(exc is not None, "view_trace: unknown id raises KeyError",
              observed=f"got={type(exc).__name__ if exc else 'no raise'}")

    Accepts either a coroutine, an async callable returning a coroutine, or
    a sync callable. Anything else: pass a zero-arg lambda.
    """
    try:
        result = (
            awaitable_or_callable() if callable(awaitable_or_callable) else awaitable_or_callable
        )
        if asyncio.iscoroutine(result):
            await result
    except expected_type as exc:
        return exc
    except BaseException:
        raise
    return None


# --- Run wrapper -------------------------------------------------------------


@dataclass
class RunResult:
    """Outcome of a probe run.

    Use ``output_items`` for assertions about what the engine emitted to its
    stream consumers. Use ``deltas`` to verify streaming text behavior. Use
    ``all_events`` (chronological union) for sequence/ordering assertions.

    ``error`` is the exception raised inside ``stream_engine_async``, or
    ``None`` if the run terminated cleanly. A ``TimeoutError`` here typically
    means the engine deadlocked (e.g., bug where the output bus was never
    closed after a driver-side failure).
    """

    output_items: list[AgentOutputItem] = field(default_factory=list)
    deltas: list[AgentTextDelta] = field(default_factory=list)
    all_events: list[EngineStreamEvent] = field(default_factory=list)
    error: BaseException | None = None


async def run_with_fake(
    fake_runner: FakeRunner,
    *,
    trace_path: Path | None = None,
    config: EngineConfig | None = None,
    messages: list[AgentMessage] | None = None,
    timeout_seconds: float = 5.0,
) -> RunResult:
    """Run the engine end-to-end with a scripted ``FakeRunner``, collecting
    every emitted event.

    Defaults: ``config = make_default_config()``, ``messages = make_default_messages()``,
    ``trace_path = isolated_trace_copy()`` (a temp copy of ``tiny_traces.jsonl``).
    Override any of them when probing a specific pathway.

    Never raises. Failures are captured in ``RunResult.error``. ``TimeoutError``
    indicates the engine took longer than ``timeout_seconds`` to emit the next
    event or terminate — usually a deadlock.
    """
    cfg = config or make_default_config()
    msgs = messages or make_default_messages()
    tp = trace_path or isolated_trace_copy()

    result = RunResult()

    async def _consume() -> None:
        async for event in stream_engine_async(msgs, cfg, tp, runner=fake_runner):
            result.all_events.append(event)
            if isinstance(event, AgentOutputItem):
                result.output_items.append(event)
            elif isinstance(event, AgentTextDelta):
                result.deltas.append(event)

    try:
        await asyncio.wait_for(_consume(), timeout=timeout_seconds)
    except BaseException as exc:
        result.error = exc

    return result
