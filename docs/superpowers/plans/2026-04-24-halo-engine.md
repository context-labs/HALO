# HALO Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new standalone `engine` uv package at `/home/declan/dev/HALO/engine/` that runs a root agent with subagents over OTel trace data using the OpenAI Agents SDK, returning a full ordered stream of messages with per-agent lineage.

**Architecture:** Async-first public entrypoints (`stream_engine_async`, `run_engine_async`) orchestrate a root agent built via the OpenAI Agents SDK. Subagents are wired through `Agent.as_tool(..., on_stream=..., custom_output_extractor=...)` and stream events through a shared run-level `EngineOutputBus`. Per-agent `AgentContext` stores typed `AgentContextItem`s with individual-item compaction; `TraceStore` is a pure, sandbox-importable query API over a build-once sidecar index; `run_code` executes sandboxed Python against `TraceStore` via `bubblewrap` (Linux) or `sandbox-exec` (macOS).

**Tech Stack:** Python 3.12, uv, Pydantic v2, OpenAI Agents SDK (latest), `openai` SDK types, `pytest`, `pytest-asyncio`, `loguru`, `rich`, `typer`. numpy + pandas inside the sandbox venv only.

**Pre-decided constraints (do not rediscover):**
- Package lives at `/home/declan/dev/HALO/engine/`, publishable to PyPI as `halo-engine`.
- The existing `/home/declan/dev/HALO/rlm/` directory is left in place as reference material. No imports from it, no test runs against it — it simply coexists.
- Input to public entrypoints is `list[AgentMessage]` (validate dicts with `AgentMessage.model_validate` at boundary if needed).
- `MessageContent` follows OpenAI Chat Completions shape: `str | list[OpenAIContentPart] | None`. Use `openai.types.chat` content part types directly.
- Streaming yields a discriminated `AgentOutputItem | AgentTextDelta`; deltas never enter `AgentContext`. `run_engine_async` filters deltas out.
- `final=True` is decided by the model: root system prompt instructs it to end its final assistant message with a line containing only `<final/>`. `OpenAiEventMapper` strips that line and sets `final=True`.
- Compaction triggers *after each agent turn*. `AgentContext.compact_old_items()` decides eligibility per item using `text_message_compaction_keep_last_messages` and `tool_call_compaction_keep_last_messages` counts of **items of that type**. Individual messages compact as they become eligible — no grouping requirement across matching tool_call/tool pairs.
- Index is assumed fresh: `ensure_index_exists` only checks for file presence. No mtime/size check.
- Sandbox uses a single pre-built venv that contains `halo-engine` plus `numpy` + `pandas`. Path is `<repo>/engine/.sandbox-venv` (created by `scripts/build_sandbox_venv.sh`).
- `maximum_parallel_subagents` is enforced by an `asyncio.Semaphore` inside `subagent_tool_factory`. Prompt guidance remains as a soft hint.
- `maximum_depth` is enforced structurally: when building a child `Agent` that would be at `depth == maximum_depth`, the subagent tool is **omitted from its tool list**. The child is physically incapable of recursing further.
- Circuit breaker: track consecutive LLM failures per `AgentExecution`. Ten in a row fails the stream with a typed `EngineAgentExhaustedError`. Any success resets the counter.
- TDD throughout: failing test → run to see fail → minimal impl → run to see pass → commit. Small commits.
- Commits MUST NOT include the `Co-Authored-By` trailer (user preference).

---

## File Structure

```text
engine/
  pyproject.toml
  README.md
  .python-version
  scripts/
    build_sandbox_venv.sh          # one-shot script that builds .sandbox-venv
  engine/
    __init__.py
    main.py                        # stream_engine_async, run_engine_async, sync wrappers
    engine_config.py               # EngineConfig
    model_config.py                # AvailableModelName, ModelConfig
    errors.py                      # engine-wide typed exceptions
    models/
      __init__.py
      messages.py                  # AgentMessage, AgentToolCall, AgentToolFunction, MessageContent alias
      engine_output.py             # AgentOutputItem, AgentTextDelta, EngineStreamEvent union
      json_value.py                # JsonValue alias + JsonMapping
    agents/
      __init__.py
      agent_config.py              # AgentConfig
      agent_context_items.py       # AgentContextItem
      agent_context.py             # AgentContext
      agent_execution.py           # AgentExecution
      engine_output_bus.py         # EngineOutputBus
      engine_run_state.py          # EngineRunState
      prompt_templates.py          # root/subagent/synthesis/compaction prompts
      openai_event_mapper.py       # SDK event → AgentContextItem / AgentOutputItem / AgentTextDelta
      openai_agent_runner.py       # builds SDK Agent, drives Runner.run_streamed, circuit breaker
    traces/
      __init__.py
      trace_index_builder.py
      trace_store.py
      models/
        __init__.py
        canonical_span.py          # SpanRecord, SpanResource, SpanScope, SpanStatus
        trace_index_config.py      # TraceIndexConfig
        trace_index_models.py      # TraceIndexRow, TraceIndexMeta
        trace_query_models.py      # filters, query, overview, count, view, search, render models
    tools/
      __init__.py
      tool_protocol.py             # EngineTool Protocol + to_sdk_function_tool adapter
      trace_tools.py               # overview/query/count/view/search tools
      agent_context_tools.py       # get_context_item
      synthesis_tool.py            # summarize_traces
      subagent_tool_factory.py     # Agent.as_tool wiring
      run_code_tool.py             # sandboxed python
    sandbox/
      __init__.py
      sandbox_config.py            # SandboxConfig, SandboxPolicy, CodeExecutionResult, RunCodeArguments
      sandbox_policy.py            # policy assembly helpers
      platform_commands.py         # build_linux_bubblewrap_command, build_macos_sandbox_exec_command
      sandbox_bootstrap.py         # wrapper script template that loads TraceStore
      sandbox_results.py           # output capping + result assembly
      sandbox_runner.py            # SandboxRunner.run_python orchestration
  tests/
    __init__.py
    conftest.py
    fixtures/
      tiny_traces.jsonl            # 3-trace fixture used across unit tests
      e2e_trace.jsonl              # placeholder; user supplies real trace
    unit/
      ... mirrors engine/ layout
    integration/
      test_engine_e2e.py
```

---

## Conventions used in every task

- All new Python files begin with `from __future__ import annotations`.
- Pydantic v2 `BaseModel` with `model_config = ConfigDict(frozen=False, extra="forbid")` on all models unless stated otherwise.
- `pathlib.Path`, never `os.path`.
- No try/except unless a specific recovery path is designed. Let unexpected errors propagate.
- Tests use `pytest-asyncio` with `@pytest.mark.asyncio` where needed. Async fixtures use `pytest_asyncio.fixture`.
- Run tests from the engine package root: `cd /home/declan/dev/HALO/engine && uv run pytest <path> -v`.
- Commits: `git -C /home/declan/dev/HALO commit -m "<message>"`. No co-author trailer.

---

## Phase 0 — Package skeleton

The existing `rlm/` directory stays in place as reference material. The new `engine/` package is a sibling that neither imports from nor depends on `rlm/`.

### Task 0.1: Create `engine/` uv package skeleton

**Files:**
- Create: `engine/pyproject.toml`
- Create: `engine/README.md`
- Create: `engine/.python-version`
- Create: `engine/engine/__init__.py`
- Create: `engine/tests/__init__.py`
- Create: `engine/tests/conftest.py`

- [ ] **Step 1: Write `engine/.python-version`**

```text
3.12
```

- [ ] **Step 2: Write `engine/pyproject.toml`**

```toml
[project]
name = "halo-engine"
version = "0.1.0"
description = "HALO engine: LLM agent runtime over OTel trace data."
requires-python = ">=3.12,<3.13"
readme = "README.md"
dependencies = [
    "pydantic>=2.8",
    "loguru>=0.7.3",
    "rich>=14.3",
    "typer>=0.15",
    "openai>=1.60",
    "openai-agents>=0.1",
    "orjson>=3.10",
]

[dependency-groups]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "ruff>=0.6",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["engine"]

[tool.pytest.ini_options]
pythonpath = ["."]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"
```

- [ ] **Step 3: Write `engine/README.md`**

```markdown
# halo-engine

LLM agent runtime for exploring OTel trace datasets. See
`docs/engine-architecture-plan.md` for design.

Install: `uv sync`

Run tests: `uv run pytest`
```

- [ ] **Step 4: Write `engine/engine/__init__.py`**

```python
from __future__ import annotations

__version__ = "0.1.0"
```

- [ ] **Step 5: Write `engine/tests/__init__.py` as empty file and `engine/tests/conftest.py`**

`engine/tests/__init__.py`: empty file.

`engine/tests/conftest.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR
```

- [ ] **Step 6: Run `uv sync` to materialize the lockfile**

Run: `cd /home/declan/dev/HALO/engine && uv sync`
Expected: `Resolved N packages`, no errors. `.venv/` created.

- [ ] **Step 7: Smoke test pytest discovers nothing but runs cleanly**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest -q`
Expected: `no tests ran` (exit code 5) OR `0 passed`. Either is fine.

- [ ] **Step 8: Commit**

```bash
git -C /home/declan/dev/HALO add engine/
git -C /home/declan/dev/HALO commit -m "feat(engine): scaffold halo-engine uv package"
```

---

### Task 0.2: Add tiny trace fixture used throughout tests

**Files:**
- Create: `engine/tests/fixtures/tiny_traces.jsonl`

- [ ] **Step 1: Write `engine/tests/fixtures/tiny_traces.jsonl`**

Three traces: two spans each, one trace has an error status. Contents (exactly, no trailing newline after last line is fine):

```jsonl
{"trace_id":"t-aaaa","span_id":"s-aaaa-1","parent_span_id":"","trace_state":"","name":"root","kind":"SPAN_KIND_INTERNAL","start_time":"2026-04-23T05:32:00.000000000Z","end_time":"2026-04-23T05:32:01.000000000Z","status":{"code":"STATUS_CODE_OK","message":""},"resource":{"attributes":{"service.name":"agent-a","service.version":"0.1.0","deployment.environment":"prod"}},"scope":{"name":"@test/scope","version":"0.0.1"},"attributes":{"openinference.span.kind":"AGENT","inference.export.schema_version":1,"inference.project_id":"prj_test","inference.observation_kind":"AGENT","inference.agent_name":"agent-a"}}
{"trace_id":"t-aaaa","span_id":"s-aaaa-2","parent_span_id":"s-aaaa-1","trace_state":"","name":"anthropic.messages.create","kind":"SPAN_KIND_CLIENT","start_time":"2026-04-23T05:32:00.100000000Z","end_time":"2026-04-23T05:32:00.900000000Z","status":{"code":"STATUS_CODE_OK","message":""},"resource":{"attributes":{"service.name":"agent-a","service.version":"0.1.0","deployment.environment":"prod"}},"scope":{"name":"@test/scope","version":"0.0.1"},"attributes":{"openinference.span.kind":"LLM","llm.provider":"anthropic","llm.model_name":"claude-sonnet-4-5","llm.token_count.prompt":100,"llm.token_count.completion":50,"inference.export.schema_version":1,"inference.project_id":"prj_test","inference.observation_kind":"LLM","inference.llm.provider":"anthropic","inference.llm.model_name":"claude-sonnet-4-5","inference.llm.input_tokens":100,"inference.llm.output_tokens":50,"inference.llm.cost.total":"0.0012","inference.agent_name":"agent-a"}}
{"trace_id":"t-bbbb","span_id":"s-bbbb-1","parent_span_id":"","trace_state":"","name":"root","kind":"SPAN_KIND_INTERNAL","start_time":"2026-04-23T06:00:00.000000000Z","end_time":"2026-04-23T06:00:02.000000000Z","status":{"code":"STATUS_CODE_ERROR","message":"tool call failed"},"resource":{"attributes":{"service.name":"agent-b","service.version":"0.2.0","deployment.environment":"prod"}},"scope":{"name":"@test/scope","version":"0.0.1"},"attributes":{"openinference.span.kind":"AGENT","inference.export.schema_version":1,"inference.project_id":"prj_test","inference.observation_kind":"AGENT","inference.agent_name":"agent-b"}}
{"trace_id":"t-bbbb","span_id":"s-bbbb-2","parent_span_id":"s-bbbb-1","trace_state":"","name":"openai.chat.completions.create","kind":"SPAN_KIND_CLIENT","start_time":"2026-04-23T06:00:00.200000000Z","end_time":"2026-04-23T06:00:01.800000000Z","status":{"code":"STATUS_CODE_ERROR","message":"tool failure"},"resource":{"attributes":{"service.name":"agent-b","service.version":"0.2.0","deployment.environment":"prod"}},"scope":{"name":"@test/scope","version":"0.0.1"},"attributes":{"openinference.span.kind":"LLM","llm.provider":"openai","llm.model_name":"gpt-5.4","llm.token_count.prompt":200,"llm.token_count.completion":40,"inference.export.schema_version":1,"inference.project_id":"prj_test","inference.observation_kind":"LLM","inference.llm.provider":"openai","inference.llm.model_name":"gpt-5.4","inference.llm.input_tokens":200,"inference.llm.output_tokens":40,"inference.llm.cost.total":"0.0040","inference.agent_name":"agent-b"}}
{"trace_id":"t-cccc","span_id":"s-cccc-1","parent_span_id":"","trace_state":"","name":"root","kind":"SPAN_KIND_INTERNAL","start_time":"2026-04-23T07:00:00.000000000Z","end_time":"2026-04-23T07:00:00.500000000Z","status":{"code":"STATUS_CODE_OK","message":""},"resource":{"attributes":{"service.name":"agent-a","service.version":"0.1.0","deployment.environment":"prod"}},"scope":{"name":"@test/scope","version":"0.0.1"},"attributes":{"openinference.span.kind":"AGENT","inference.export.schema_version":1,"inference.project_id":"prj_test","inference.observation_kind":"AGENT","inference.agent_name":"agent-a"}}
{"trace_id":"t-cccc","span_id":"s-cccc-2","parent_span_id":"s-cccc-1","trace_state":"","name":"anthropic.messages.create","kind":"SPAN_KIND_CLIENT","start_time":"2026-04-23T07:00:00.100000000Z","end_time":"2026-04-23T07:00:00.400000000Z","status":{"code":"STATUS_CODE_OK","message":""},"resource":{"attributes":{"service.name":"agent-a","service.version":"0.1.0","deployment.environment":"prod"}},"scope":{"name":"@test/scope","version":"0.0.1"},"attributes":{"openinference.span.kind":"LLM","llm.provider":"anthropic","llm.model_name":"claude-haiku-4-5","llm.token_count.prompt":30,"llm.token_count.completion":10,"inference.export.schema_version":1,"inference.project_id":"prj_test","inference.observation_kind":"LLM","inference.llm.provider":"anthropic","inference.llm.model_name":"claude-haiku-4-5","inference.llm.input_tokens":30,"inference.llm.output_tokens":10,"inference.llm.cost.total":"0.0002","inference.agent_name":"agent-a"}}
```

- [ ] **Step 2: Commit**

```bash
git -C /home/declan/dev/HALO add engine/tests/fixtures/tiny_traces.jsonl
git -C /home/declan/dev/HALO commit -m "test(engine): add tiny traces fixture"
```

---

## Phase 1 — Core typed models

All tasks in this phase create `engine/<path>.py`, a matching `engine/tests/unit/<path>_test.py`, and commit. Tests validate construction, required fields, and round-trip through `model_dump_json` / `model_validate_json`.

### Task 1.1: JSON value aliases

**Files:**
- Create: `engine/engine/models/__init__.py` (empty)
- Create: `engine/engine/models/json_value.py`
- Create: `engine/tests/unit/__init__.py` (empty)
- Create: `engine/tests/unit/models/__init__.py` (empty)
- Create: `engine/tests/unit/models/test_json_value.py`

- [ ] **Step 1: Write the failing test**

`engine/tests/unit/models/test_json_value.py`:

```python
from __future__ import annotations

from engine.models.json_value import JsonMapping, JsonValue


def test_json_value_accepts_primitive_and_nested_shapes() -> None:
    value: JsonValue = {"a": [1, 2.0, True, None, "x", {"nested": [1]}]}
    assert isinstance(value, dict)


def test_json_mapping_alias_is_dict() -> None:
    mapping: JsonMapping = {"k": "v"}
    assert mapping["k"] == "v"
```

- [ ] **Step 2: Run — expect ImportError**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/models/test_json_value.py -v`
Expected: `ModuleNotFoundError: No module named 'engine.models.json_value'`.

- [ ] **Step 3: Implement `engine/engine/models/json_value.py`**

```python
from __future__ import annotations

from typing import TypeAlias

JsonValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)

JsonMapping: TypeAlias = dict[str, JsonValue]
```

- [ ] **Step 4: Run — expect pass**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/models/test_json_value.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO add engine/engine/models engine/tests/unit
git -C /home/declan/dev/HALO commit -m "feat(engine): add JsonValue/JsonMapping aliases"
```

---

### Task 1.2: `AgentMessage` + tool call models

**Files:**
- Create: `engine/engine/models/messages.py`
- Create: `engine/tests/unit/models/test_messages.py`

- [ ] **Step 1: Write failing tests**

`engine/tests/unit/models/test_messages.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from engine.models.messages import AgentMessage, AgentToolCall, AgentToolFunction


def test_user_message_minimum() -> None:
    msg = AgentMessage(role="user", content="hi")
    assert msg.role == "user"
    assert msg.content == "hi"
    assert msg.tool_calls is None


def test_assistant_tool_call_message() -> None:
    msg = AgentMessage(
        role="assistant",
        content=None,
        tool_calls=[
            AgentToolCall(
                id="call_1",
                function=AgentToolFunction(name="x", arguments="{}"),
            )
        ],
    )
    assert msg.tool_calls is not None
    assert msg.tool_calls[0].function.name == "x"


def test_tool_role_requires_tool_call_id_when_used() -> None:
    msg = AgentMessage(
        role="tool",
        content="{}",
        tool_call_id="call_1",
        name="x",
    )
    assert msg.tool_call_id == "call_1"


def test_invalid_role_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentMessage(role="bogus", content="x")  # type: ignore[arg-type]


def test_roundtrip_json() -> None:
    msg = AgentMessage(role="user", content="hi")
    blob = msg.model_dump_json()
    restored = AgentMessage.model_validate_json(blob)
    assert restored == msg
```

- [ ] **Step 2: Run — expect ImportError**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/models/test_messages.py -v`
Expected: ImportError on `engine.models.messages`.

- [ ] **Step 3: Implement `engine/engine/models/messages.py`**

```python
from __future__ import annotations

from typing import Literal, TypeAlias

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import BaseModel, ConfigDict

MessageContent: TypeAlias = str | list[ChatCompletionContentPartParam] | None


class AgentToolFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: str


class AgentToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["function"] = "function"
    function: AgentToolFunction


class AgentMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: MessageContent = None
    tool_calls: list[AgentToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
```

- [ ] **Step 4: Run — expect pass**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/models/test_messages.py -v`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO add engine/engine/models/messages.py engine/tests/unit/models/test_messages.py
git -C /home/declan/dev/HALO commit -m "feat(engine): add AgentMessage + tool call models"
```

---

### Task 1.3: Engine output items (`AgentOutputItem`, `AgentTextDelta`)

**Files:**
- Create: `engine/engine/models/engine_output.py`
- Create: `engine/tests/unit/models/test_engine_output.py`

- [ ] **Step 1: Write failing tests**

`engine/tests/unit/models/test_engine_output.py`:

```python
from __future__ import annotations

from engine.models.engine_output import AgentOutputItem, AgentTextDelta, EngineStreamEvent
from engine.models.messages import AgentMessage


def test_output_item_defaults() -> None:
    item = AgentOutputItem(
        sequence=0,
        agent_id="root",
        parent_agent_id=None,
        parent_tool_call_id=None,
        agent_name="root",
        depth=0,
        item=AgentMessage(role="assistant", content="hi"),
    )
    assert item.final is False


def test_delta_requires_text() -> None:
    delta = AgentTextDelta(
        sequence=1,
        agent_id="root",
        parent_agent_id=None,
        parent_tool_call_id=None,
        depth=0,
        item_id="msg_1",
        text_delta="par",
    )
    assert delta.text_delta == "par"


def test_stream_event_union_accepts_both() -> None:
    events: list[EngineStreamEvent] = [
        AgentOutputItem(
            sequence=0,
            agent_id="root",
            parent_agent_id=None,
            parent_tool_call_id=None,
            agent_name="root",
            depth=0,
            item=AgentMessage(role="assistant", content="hi"),
        ),
        AgentTextDelta(
            sequence=1,
            agent_id="root",
            parent_agent_id=None,
            parent_tool_call_id=None,
            depth=0,
            item_id="msg_1",
            text_delta="x",
        ),
    ]
    assert len(events) == 2
```

- [ ] **Step 2: Run — expect fail**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/models/test_engine_output.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`engine/engine/models/engine_output.py`:

```python
from __future__ import annotations

from typing import TypeAlias

from pydantic import BaseModel, ConfigDict

from engine.models.messages import AgentMessage


class AgentOutputItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int
    agent_id: str
    parent_agent_id: str | None
    parent_tool_call_id: str | None
    agent_name: str
    depth: int
    item: AgentMessage
    final: bool = False


class AgentTextDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int
    agent_id: str
    parent_agent_id: str | None
    parent_tool_call_id: str | None
    depth: int
    item_id: str
    text_delta: str


EngineStreamEvent: TypeAlias = AgentOutputItem | AgentTextDelta
```

- [ ] **Step 4: Run — expect pass**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/models/test_engine_output.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO add engine/engine/models/engine_output.py engine/tests/unit/models/test_engine_output.py
git -C /home/declan/dev/HALO commit -m "feat(engine): add AgentOutputItem and AgentTextDelta"
```

---

### Task 1.4: `ModelConfig` + `AvailableModelName`

**Files:**
- Create: `engine/engine/model_config.py`
- Create: `engine/tests/unit/test_model_config.py`

- [ ] **Step 1: Write failing tests**

`engine/tests/unit/test_model_config.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from engine.model_config import AvailableModelName, ModelConfig


def test_defaults() -> None:
    cfg = ModelConfig(name="claude-opus-4-7")
    assert cfg.temperature is None
    assert cfg.maximum_output_tokens is None
    assert cfg.parallel_tool_calls is True


def test_model_name_literal_enforced() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(name="not-a-real-model")  # type: ignore[arg-type]


def test_all_names_listed() -> None:
    expected = {
        "claude-opus-4-7",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "gpt-5.4",
        "gpt-5.4-mini",
    }
    actual = set(AvailableModelName.__args__)  # type: ignore[attr-defined]
    assert actual == expected
```

- [ ] **Step 2: Run — expect fail**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/test_model_config.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`engine/engine/model_config.py`:

```python
from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

AvailableModelName: TypeAlias = Literal[
    "claude-opus-4-7",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "gpt-5.4",
    "gpt-5.4-mini",
]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: AvailableModelName
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    maximum_output_tokens: int | None = Field(default=None, gt=0)
    parallel_tool_calls: bool = True
```

- [ ] **Step 4: Run — expect pass**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/test_model_config.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO add engine/engine/model_config.py engine/tests/unit/test_model_config.py
git -C /home/declan/dev/HALO commit -m "feat(engine): add ModelConfig and AvailableModelName"
```

---

### Task 1.5: `AgentConfig`

**Files:**
- Create: `engine/engine/agents/__init__.py` (empty)
- Create: `engine/engine/agents/agent_config.py`
- Create: `engine/tests/unit/agents/__init__.py` (empty)
- Create: `engine/tests/unit/agents/test_agent_config.py`

- [ ] **Step 1: Write failing tests**

`engine/tests/unit/agents/test_agent_config.py`:

```python
from __future__ import annotations

from engine.agents.agent_config import AgentConfig
from engine.model_config import ModelConfig


def test_agent_config_constructs() -> None:
    cfg = AgentConfig(
        name="root",
        instructions="You are root.",
        model=ModelConfig(name="claude-opus-4-7"),
        maximum_turns=20,
    )
    assert cfg.name == "root"
    assert cfg.maximum_turns == 20
    assert cfg.model.name == "claude-opus-4-7"
```

- [ ] **Step 2: Run — expect fail**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/agents/test_agent_config.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`engine/engine/agents/agent_config.py`:

```python
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from engine.model_config import ModelConfig


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    instructions: str
    model: ModelConfig
    maximum_turns: int = Field(gt=0)
```

- [ ] **Step 4: Run — expect pass**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/agents/test_agent_config.py -v`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO add engine/engine/agents engine/tests/unit/agents
git -C /home/declan/dev/HALO commit -m "feat(engine): add AgentConfig"
```

---

### Task 1.6: `TraceIndexConfig`

**Files:**
- Create: `engine/engine/traces/__init__.py` (empty)
- Create: `engine/engine/traces/models/__init__.py` (empty)
- Create: `engine/engine/traces/models/trace_index_config.py`
- Create: `engine/tests/unit/traces/__init__.py` (empty)
- Create: `engine/tests/unit/traces/models/__init__.py` (empty)
- Create: `engine/tests/unit/traces/models/test_trace_index_config.py`

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

from pathlib import Path

from engine.traces.models.trace_index_config import TraceIndexConfig


def test_defaults() -> None:
    cfg = TraceIndexConfig()
    assert cfg.index_path is None
    assert cfg.schema_version == 1


def test_explicit_index_path(tmp_path: Path) -> None:
    cfg = TraceIndexConfig(index_path=tmp_path / "idx.jsonl", schema_version=1)
    assert cfg.index_path == tmp_path / "idx.jsonl"
```

- [ ] **Step 2: Run — expect fail**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/traces/models/test_trace_index_config.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`engine/engine/traces/models/trace_index_config.py`:

```python
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class TraceIndexConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_path: Path | None = None
    schema_version: int = 1
```

- [ ] **Step 4: Run — expect pass**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/traces/models/test_trace_index_config.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO add engine/engine/traces engine/tests/unit/traces
git -C /home/declan/dev/HALO commit -m "feat(engine): add TraceIndexConfig"
```

---

### Task 1.7: `SandboxConfig` / `SandboxPolicy` / `CodeExecutionResult` / `RunCodeArguments`

**Files:**
- Create: `engine/engine/sandbox/__init__.py` (empty)
- Create: `engine/engine/sandbox/sandbox_config.py`
- Create: `engine/tests/unit/sandbox/__init__.py` (empty)
- Create: `engine/tests/unit/sandbox/test_sandbox_config.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

from pathlib import Path

from engine.sandbox.sandbox_config import (
    CodeExecutionResult,
    RunCodeArguments,
    SandboxConfig,
    SandboxPolicy,
)


def test_sandbox_config_defaults() -> None:
    cfg = SandboxConfig()
    assert cfg.timeout_seconds == 10.0
    assert cfg.maximum_stdout_bytes == 64_000
    assert cfg.maximum_stderr_bytes == 64_000
    assert cfg.python_executable is None


def test_sandbox_policy(tmp_path: Path) -> None:
    pol = SandboxPolicy(
        readonly_paths=[tmp_path / "ro"],
        writable_paths=[tmp_path / "w"],
        timeout_seconds=5.0,
    )
    assert pol.network_enabled is False


def test_result_shape() -> None:
    r = CodeExecutionResult(exit_code=0, stdout="ok", stderr="", timed_out=False)
    assert r.exit_code == 0


def test_run_code_arguments() -> None:
    args = RunCodeArguments(code="print(1)")
    assert args.code == "print(1)"
```

- [ ] **Step 2: Run — expect fail**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/sandbox/test_sandbox_config.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`engine/engine/sandbox/sandbox_config.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SandboxConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float = Field(default=10.0, gt=0)
    maximum_stdout_bytes: int = Field(default=64_000, gt=0)
    maximum_stderr_bytes: int = Field(default=64_000, gt=0)
    python_executable: Path | None = None


class SandboxPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    readonly_paths: list[Path]
    writable_paths: list[Path]
    network_enabled: Literal[False] = False
    timeout_seconds: float = Field(gt=0)


class CodeExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


class RunCodeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
```

- [ ] **Step 4: Run — expect pass**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/sandbox/test_sandbox_config.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO add engine/engine/sandbox engine/tests/unit/sandbox
git -C /home/declan/dev/HALO commit -m "feat(engine): add sandbox config models"
```

---

### Task 1.8: `EngineConfig`

**Files:**
- Create: `engine/engine/engine_config.py`
- Create: `engine/tests/unit/test_engine_config.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

from engine.agents.agent_config import AgentConfig
from engine.engine_config import EngineConfig
from engine.model_config import ModelConfig


def _agent(name: str) -> AgentConfig:
    return AgentConfig(
        name=name,
        instructions="",
        model=ModelConfig(name="claude-sonnet-4-5"),
        maximum_turns=10,
    )


def test_engine_config_defaults() -> None:
    cfg = EngineConfig(
        root_agent=_agent("root"),
        subagent=_agent("sub"),
        synthesis_model=ModelConfig(name="claude-haiku-4-5"),
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
    )
    assert cfg.text_message_compaction_keep_last_messages == 12
    assert cfg.tool_call_compaction_keep_last_messages == 6
    assert cfg.maximum_depth == 2
    assert cfg.maximum_parallel_subagents == 4
```

- [ ] **Step 2: Run — expect fail**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/test_engine_config.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`engine/engine/engine_config.py`:

```python
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from engine.agents.agent_config import AgentConfig
from engine.model_config import ModelConfig
from engine.sandbox.sandbox_config import SandboxConfig
from engine.traces.models.trace_index_config import TraceIndexConfig


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_agent: AgentConfig
    subagent: AgentConfig
    synthesis_model: ModelConfig
    compaction_model: ModelConfig
    trace_index: TraceIndexConfig = Field(default_factory=TraceIndexConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    text_message_compaction_keep_last_messages: int = Field(default=12, ge=0)
    tool_call_compaction_keep_last_messages: int = Field(default=6, ge=0)
    maximum_depth: int = Field(default=2, ge=0)
    maximum_parallel_subagents: int = Field(default=4, gt=0)
```

- [ ] **Step 4: Run — expect pass**

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO add engine/engine/engine_config.py engine/tests/unit/test_engine_config.py
git -C /home/declan/dev/HALO commit -m "feat(engine): add EngineConfig"
```

---

### Task 1.9: Engine-wide typed errors

**Files:**
- Create: `engine/engine/errors.py`
- Create: `engine/tests/unit/test_errors.py`

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

import pytest

from engine.errors import (
    EngineAgentExhaustedError,
    EngineError,
    EngineMaxDepthExceededError,
    EngineSandboxDeniedError,
    EngineToolError,
)


def test_hierarchy() -> None:
    for exc in (
        EngineAgentExhaustedError,
        EngineMaxDepthExceededError,
        EngineSandboxDeniedError,
        EngineToolError,
    ):
        assert issubclass(exc, EngineError)


def test_raise_and_message() -> None:
    with pytest.raises(EngineAgentExhaustedError) as ei:
        raise EngineAgentExhaustedError("10 consecutive llm failures")
    assert "10" in str(ei.value)
```

- [ ] **Step 2: Run — expect fail**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/test_errors.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`engine/engine/errors.py`:

```python
from __future__ import annotations


class EngineError(Exception):
    """Base engine error."""


class EngineAgentExhaustedError(EngineError):
    """Raised when an agent hits the consecutive-LLM-failure circuit breaker."""


class EngineMaxDepthExceededError(EngineError):
    """Raised when subagent spawn attempted beyond maximum_depth."""


class EngineSandboxDeniedError(EngineError):
    """Raised when sandbox execution was blocked by policy."""


class EngineToolError(EngineError):
    """Raised from a tool adapter when returning a typed error to the caller."""
```

- [ ] **Step 4: Run — expect pass**

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO add engine/engine/errors.py engine/tests/unit/test_errors.py
git -C /home/declan/dev/HALO commit -m "feat(engine): add typed engine errors"
```

---

### Task 1.10: Canonical span models

**Files:**
- Create: `engine/engine/traces/models/canonical_span.py`
- Create: `engine/tests/unit/traces/models/test_canonical_span.py`

- [ ] **Step 1: Write failing tests**

`engine/tests/unit/traces/models/test_canonical_span.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from engine.traces.models.canonical_span import SpanRecord


def test_parse_tiny_fixture_first_line(fixtures_dir: Path) -> None:
    raw = (fixtures_dir / "tiny_traces.jsonl").read_text().splitlines()[0]
    span = SpanRecord.model_validate_json(raw)
    assert span.trace_id == "t-aaaa"
    assert span.span_id == "s-aaaa-1"
    assert span.parent_span_id == ""
    assert span.resource.attributes["service.name"] == "agent-a"
    assert span.attributes["inference.export.schema_version"] == 1


def test_status_defaults_preserved() -> None:
    raw = {
        "trace_id": "t",
        "span_id": "s",
        "parent_span_id": "",
        "trace_state": "",
        "name": "x",
        "kind": "SPAN_KIND_INTERNAL",
        "start_time": "2026-04-23T00:00:00Z",
        "end_time": "2026-04-23T00:00:01Z",
        "status": {"code": "STATUS_CODE_OK", "message": ""},
        "resource": {"attributes": {}},
        "scope": {"name": "n", "version": "v"},
        "attributes": {},
    }
    span = SpanRecord.model_validate(raw)
    assert span.status.code == "STATUS_CODE_OK"
    assert json.loads(span.model_dump_json())["trace_id"] == "t"
```

- [ ] **Step 2: Run — expect fail**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/traces/models/test_canonical_span.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`engine/engine/traces/models/canonical_span.py`:

```python
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from engine.models.json_value import JsonMapping


class SpanStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str = ""


class SpanResource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attributes: JsonMapping


class SpanScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = ""


class SpanRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    trace_id: str
    span_id: str
    parent_span_id: str = ""
    trace_state: str = ""
    name: str
    kind: str
    start_time: str
    end_time: str
    status: SpanStatus
    resource: SpanResource
    scope: SpanScope
    attributes: JsonMapping
```

The `extra="allow"` on `SpanRecord` lets us tolerate upstream additions to the OTel envelope without silently dropping them.

- [ ] **Step 4: Run — expect pass**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/traces/models/test_canonical_span.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO add engine/engine/traces/models/canonical_span.py engine/tests/unit/traces/models/test_canonical_span.py
git -C /home/declan/dev/HALO commit -m "feat(engine): add canonical span models"
```

---

### Task 1.11: Trace index row + meta models

**Files:**
- Create: `engine/engine/traces/models/trace_index_models.py`
- Create: `engine/tests/unit/traces/models/test_trace_index_models.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

from engine.traces.models.trace_index_models import TraceIndexMeta, TraceIndexRow


def test_row_roundtrip() -> None:
    row = TraceIndexRow(
        trace_id="t1",
        byte_offsets=[0, 512],
        byte_lengths=[512, 256],
        span_count=2,
        start_time="2026-04-23T00:00:00Z",
        end_time="2026-04-23T00:00:01Z",
        has_errors=False,
        service_names=["svc"],
        model_names=["claude-sonnet-4-5"],
        total_input_tokens=100,
        total_output_tokens=50,
        project_id="prj_1",
        agent_names=["agent-a"],
    )
    blob = row.model_dump_json()
    restored = TraceIndexRow.model_validate_json(blob)
    assert restored == row


def test_meta_defaults() -> None:
    meta = TraceIndexMeta(schema_version=1, trace_count=3)
    assert meta.schema_version == 1
    assert meta.trace_count == 3
```

- [ ] **Step 2: Run — expect fail**

Expected: ImportError.

- [ ] **Step 3: Implement**

`engine/engine/traces/models/trace_index_models.py`:

```python
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TraceIndexRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    byte_offsets: list[int]
    byte_lengths: list[int]
    span_count: int = Field(ge=0)
    start_time: str
    end_time: str
    has_errors: bool
    service_names: list[str]
    model_names: list[str]
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    project_id: str | None = None
    agent_names: list[str]


class TraceIndexMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    trace_count: int = Field(ge=0)
```

- [ ] **Step 4: Run — expect pass**

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO add engine/engine/traces/models/trace_index_models.py engine/tests/unit/traces/models/test_trace_index_models.py
git -C /home/declan/dev/HALO commit -m "feat(engine): add trace index row/meta models"
```

---

### Task 1.12: Trace query models

**Files:**
- Create: `engine/engine/traces/models/trace_query_models.py`
- Create: `engine/tests/unit/traces/models/test_trace_query_models.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

from engine.traces.models.trace_query_models import (
    CountTracesArguments,
    CountTracesResult,
    DatasetOverview,
    DatasetOverviewArguments,
    DatasetOverviewResult,
    QueryTracesArguments,
    QueryTracesResult,
    SearchTraceArguments,
    SearchTraceResult,
    TraceCountResult,
    TraceFilters,
    TraceQueryResult,
    TraceSearchResult,
    TraceSummary,
    TraceView,
    ViewTraceArguments,
    ViewTraceResult,
)


def test_filters_all_optional() -> None:
    f = TraceFilters()
    assert f.has_errors is None
    assert f.model_names is None
    assert f.service_names is None


def test_query_arguments_defaults() -> None:
    args = QueryTracesArguments(filters=TraceFilters())
    assert args.limit == 50
    assert args.offset == 0


def test_trace_summary_roundtrip() -> None:
    s = TraceSummary(
        trace_id="t",
        span_count=2,
        start_time="a",
        end_time="b",
        has_errors=False,
        service_names=["svc"],
        model_names=["m"],
        total_input_tokens=1,
        total_output_tokens=2,
        agent_names=["a"],
    )
    assert TraceSummary.model_validate_json(s.model_dump_json()) == s


def test_count_result() -> None:
    r = TraceCountResult(total=7)
    assert r.total == 7


def test_search_result_holds_matches() -> None:
    r = TraceSearchResult(trace_id="t", match_count=2, matches=["hit1", "hit2"])
    assert r.match_count == 2


def test_view_has_span_list() -> None:
    v = TraceView(trace_id="t", spans=[])
    assert v.trace_id == "t"


def test_dataset_overview() -> None:
    ov = DatasetOverview(
        total_traces=3,
        total_spans=6,
        earliest_start_time="a",
        latest_end_time="b",
        service_names=["svc"],
        model_names=["m"],
        agent_names=["a"],
        error_trace_count=1,
        total_input_tokens=330,
        total_output_tokens=100,
    )
    assert ov.total_traces == 3


def test_result_wrappers_tool_boundary() -> None:
    assert QueryTracesResult(result=TraceQueryResult(traces=[], total=0)).result.total == 0
    assert ViewTraceResult(result=TraceView(trace_id="t", spans=[])).result.trace_id == "t"
    assert CountTracesResult(result=TraceCountResult(total=0)).result.total == 0
    assert SearchTraceResult(result=TraceSearchResult(trace_id="t", match_count=0, matches=[])).result.match_count == 0
    ov = DatasetOverview(
        total_traces=0, total_spans=0, earliest_start_time="", latest_end_time="",
        service_names=[], model_names=[], agent_names=[], error_trace_count=0,
        total_input_tokens=0, total_output_tokens=0,
    )
    assert DatasetOverviewResult(result=ov).result.total_traces == 0
    assert DatasetOverviewArguments(filters=TraceFilters()).filters.has_errors is None
    assert SearchTraceArguments(trace_id="t", pattern="x").pattern == "x"
    assert ViewTraceArguments(trace_id="t").trace_id == "t"
    assert CountTracesArguments(filters=TraceFilters()).filters.has_errors is None
```

- [ ] **Step 2: Run — expect fail**

Expected: ImportError.

- [ ] **Step 3: Implement**

`engine/engine/traces/models/trace_query_models.py`:

```python
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from engine.traces.models.canonical_span import SpanRecord


class TraceFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    has_errors: bool | None = None
    model_names: list[str] | None = None
    service_names: list[str] | None = None
    agent_names: list[str] | None = None
    project_id: str | None = None
    start_time_gte: str | None = None
    end_time_lte: str | None = None


class TraceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    span_count: int = Field(ge=0)
    start_time: str
    end_time: str
    has_errors: bool
    service_names: list[str]
    model_names: list[str]
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    agent_names: list[str]


class TraceQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    traces: list[TraceSummary]
    total: int = Field(ge=0)


class TraceCountResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=0)


class TraceSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    match_count: int = Field(ge=0)
    matches: list[str]


class TraceView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    spans: list[SpanRecord]


class DatasetOverview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_traces: int
    total_spans: int
    earliest_start_time: str
    latest_end_time: str
    service_names: list[str]
    model_names: list[str]
    agent_names: list[str]
    error_trace_count: int
    total_input_tokens: int
    total_output_tokens: int


class QueryTracesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filters: TraceFilters
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class CountTracesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filters: TraceFilters


class ViewTraceArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str


class SearchTraceArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    pattern: str


class DatasetOverviewArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filters: TraceFilters


class QueryTracesResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: TraceQueryResult


class CountTracesResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: TraceCountResult


class ViewTraceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: TraceView


class SearchTraceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: TraceSearchResult


class DatasetOverviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: DatasetOverview
```

- [ ] **Step 4: Run — expect pass**

Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO add engine/engine/traces/models/trace_query_models.py engine/tests/unit/traces/models/test_trace_query_models.py
git -C /home/declan/dev/HALO commit -m "feat(engine): add trace query/argument/result models"
```

---

## Phase 2 — TraceIndexBuilder

### Task 2.1: `ensure_index_exists` returns the computed index path without rebuilding if file present

**Files:**
- Create: `engine/engine/traces/trace_index_builder.py`
- Create: `engine/tests/unit/traces/test_trace_index_builder.py`

- [ ] **Step 1: Write failing test (presence-only check)**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder


@pytest.mark.asyncio
async def test_ensure_index_exists_default_path_returned(tmp_path: Path) -> None:
    trace_path = tmp_path / "t.jsonl"
    trace_path.write_text("")
    # Pre-create the index so builder returns without rebuilding
    default_index = Path(str(trace_path) + ".engine-index.jsonl")
    default_meta = Path(str(trace_path) + ".engine-index.meta.json")
    default_index.write_text("")
    default_meta.write_text('{"schema_version":1,"trace_count":0}')

    result_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=TraceIndexConfig(),
    )
    assert result_path == default_index


@pytest.mark.asyncio
async def test_ensure_index_exists_explicit_override(tmp_path: Path) -> None:
    trace_path = tmp_path / "t.jsonl"
    trace_path.write_text("")
    custom_index = tmp_path / "custom.idx.jsonl"
    custom_meta = Path(str(custom_index) + ".meta.json")
    custom_index.write_text("")
    custom_meta.write_text('{"schema_version":1,"trace_count":0}')

    result_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=TraceIndexConfig(index_path=custom_index),
    )
    assert result_path == custom_index
```

- [ ] **Step 2: Run — expect fail**

Expected: ImportError.

- [ ] **Step 3: Implement the happy path (stub build — build method filled in Task 2.2)**

`engine/engine/traces/trace_index_builder.py`:

```python
from __future__ import annotations

from pathlib import Path

from engine.traces.models.trace_index_config import TraceIndexConfig


class TraceIndexBuilder:
    @classmethod
    async def ensure_index_exists(
        cls,
        trace_path: Path,
        config: TraceIndexConfig,
    ) -> Path:
        index_path = config.index_path or Path(str(trace_path) + ".engine-index.jsonl")
        meta_path = cls._meta_path_for(index_path)

        if index_path.exists() and meta_path.exists():
            return index_path

        await cls.build_index(
            trace_path=trace_path,
            index_path=index_path,
            meta_path=meta_path,
            schema_version=config.schema_version,
        )
        return index_path

    @staticmethod
    def _meta_path_for(index_path: Path) -> Path:
        # Default index paths end in ".engine-index.jsonl"; swap suffix.
        name = index_path.name
        if name.endswith(".engine-index.jsonl"):
            return index_path.with_name(name[:-len(".jsonl")] + ".meta.json")
        return index_path.with_name(name + ".meta.json")

    @classmethod
    async def build_index(
        cls,
        trace_path: Path,
        index_path: Path,
        meta_path: Path,
        schema_version: int,
    ) -> None:
        # Real implementation lands in Task 2.2.
        raise NotImplementedError
```

- [ ] **Step 4: Run — expect pass**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/traces/test_trace_index_builder.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO add engine/engine/traces/trace_index_builder.py engine/tests/unit/traces/test_trace_index_builder.py
git -C /home/declan/dev/HALO commit -m "feat(engine): TraceIndexBuilder.ensure_index_exists presence check"
```

---

### Task 2.2: `build_index` scans JSONL once and writes index + meta atomically

**Files:**
- Modify: `engine/engine/traces/trace_index_builder.py`
- Modify: `engine/tests/unit/traces/test_trace_index_builder.py`

- [ ] **Step 1: Append failing build test**

Add to the test file:

```python
import json

from engine.traces.models.trace_index_models import TraceIndexMeta, TraceIndexRow


@pytest.mark.asyncio
async def test_build_index_from_tiny_fixture(tmp_path: Path, fixtures_dir: Path) -> None:
    src = fixtures_dir / "tiny_traces.jsonl"
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes(src.read_bytes())

    result_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=TraceIndexConfig(),
    )
    assert result_path.exists()
    meta_path = TraceIndexBuilder._meta_path_for(result_path)
    assert meta_path.exists()

    meta = TraceIndexMeta.model_validate_json(meta_path.read_text())
    assert meta.schema_version == 1
    assert meta.trace_count == 3

    rows = [TraceIndexRow.model_validate_json(line) for line in result_path.read_text().splitlines()]
    rows_by_id = {r.trace_id: r for r in rows}
    assert set(rows_by_id) == {"t-aaaa", "t-bbbb", "t-cccc"}

    bb = rows_by_id["t-bbbb"]
    assert bb.has_errors is True
    assert "gpt-5.4" in bb.model_names
    assert bb.total_input_tokens == 200
    assert bb.total_output_tokens == 40

    # Byte offset reads back the original span bytes
    with trace_path.open("rb") as fh:
        fh.seek(bb.byte_offsets[0])
        blob = fh.read(bb.byte_lengths[0])
    span = json.loads(blob)
    assert span["span_id"] == "s-bbbb-1"
```

- [ ] **Step 2: Run — expect NotImplementedError**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/traces/test_trace_index_builder.py::test_build_index_from_tiny_fixture -v`
Expected: `NotImplementedError` raised from `build_index`.

- [ ] **Step 3: Replace `build_index` body**

Replace the `build_index` classmethod in `engine/engine/traces/trace_index_builder.py` with:

```python
    @classmethod
    async def build_index(
        cls,
        trace_path: Path,
        index_path: Path,
        meta_path: Path,
        schema_version: int,
    ) -> None:
        if schema_version != 1:
            raise ValueError(f"unsupported trace index schema_version={schema_version}")

        rows_by_trace: dict[str, _RowAccumulator] = {}

        with trace_path.open("rb") as fh:
            offset = 0
            for raw_line in fh:
                byte_length = len(raw_line)
                stripped = raw_line.rstrip(b"\n")
                # Account for trailing newline in byte length; JSON content excludes it.
                if stripped:
                    span = SpanRecord.model_validate_json(stripped)
                    acc = rows_by_trace.setdefault(span.trace_id, _RowAccumulator(trace_id=span.trace_id))
                    acc.absorb(span=span, byte_offset=offset, byte_length=len(stripped))
                offset += byte_length

        rows = [acc.finalize() for acc in rows_by_trace.values()]

        tmp_index = index_path.with_suffix(index_path.suffix + ".tmp")
        tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")

        tmp_index.write_text(
            "\n".join(row.model_dump_json() for row in rows) + ("\n" if rows else "")
        )
        tmp_meta.write_text(
            TraceIndexMeta(schema_version=schema_version, trace_count=len(rows)).model_dump_json()
        )

        tmp_index.replace(index_path)
        tmp_meta.replace(meta_path)
```

And add to the top of the file:

```python
from dataclasses import dataclass, field

from engine.traces.models.canonical_span import SpanRecord
from engine.traces.models.trace_index_models import TraceIndexMeta, TraceIndexRow


@dataclass
class _RowAccumulator:
    trace_id: str
    byte_offsets: list[int] = field(default_factory=list)
    byte_lengths: list[int] = field(default_factory=list)
    span_count: int = 0
    start_time: str = ""
    end_time: str = ""
    has_errors: bool = False
    service_names: set[str] = field(default_factory=set)
    model_names: set[str] = field(default_factory=set)
    agent_names: set[str] = field(default_factory=set)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    project_id: str | None = None

    def absorb(self, *, span: SpanRecord, byte_offset: int, byte_length: int) -> None:
        self.byte_offsets.append(byte_offset)
        self.byte_lengths.append(byte_length)
        self.span_count += 1

        if not self.start_time or span.start_time < self.start_time:
            self.start_time = span.start_time
        if not self.end_time or span.end_time > self.end_time:
            self.end_time = span.end_time

        if span.status.code == "STATUS_CODE_ERROR":
            self.has_errors = True

        svc = span.resource.attributes.get("service.name")
        if isinstance(svc, str):
            self.service_names.add(svc)

        model = span.attributes.get("inference.llm.model_name") or span.attributes.get("llm.model_name")
        if isinstance(model, str) and model:
            self.model_names.add(model)

        agent = span.attributes.get("inference.agent_name")
        if isinstance(agent, str) and agent:
            self.agent_names.add(agent)

        input_tokens = span.attributes.get("inference.llm.input_tokens")
        if isinstance(input_tokens, int):
            self.total_input_tokens += input_tokens
        output_tokens = span.attributes.get("inference.llm.output_tokens")
        if isinstance(output_tokens, int):
            self.total_output_tokens += output_tokens

        proj = span.attributes.get("inference.project_id")
        if isinstance(proj, str) and self.project_id is None:
            self.project_id = proj

    def finalize(self) -> TraceIndexRow:
        return TraceIndexRow(
            trace_id=self.trace_id,
            byte_offsets=self.byte_offsets,
            byte_lengths=self.byte_lengths,
            span_count=self.span_count,
            start_time=self.start_time,
            end_time=self.end_time,
            has_errors=self.has_errors,
            service_names=sorted(self.service_names),
            model_names=sorted(self.model_names),
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            project_id=self.project_id,
            agent_names=sorted(self.agent_names),
        )
```

- [ ] **Step 4: Run — expect pass**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/traces/test_trace_index_builder.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO add engine/engine/traces/trace_index_builder.py engine/tests/unit/traces/test_trace_index_builder.py
git -C /home/declan/dev/HALO commit -m "feat(engine): implement TraceIndexBuilder.build_index"
```

---

### Task 2.3: Fail fast on unsupported schema version in existing meta

**Files:**
- Modify: `engine/engine/traces/trace_index_builder.py`
- Modify: `engine/tests/unit/traces/test_trace_index_builder.py`

- [ ] **Step 1: Append failing test**

```python
@pytest.mark.asyncio
async def test_ensure_index_rejects_unsupported_existing_schema(tmp_path: Path) -> None:
    trace_path = tmp_path / "t.jsonl"
    trace_path.write_text("")
    index_path = Path(str(trace_path) + ".engine-index.jsonl")
    meta_path = TraceIndexBuilder._meta_path_for(index_path)
    index_path.write_text("")
    meta_path.write_text('{"schema_version":999,"trace_count":0}')

    with pytest.raises(ValueError, match="schema_version"):
        await TraceIndexBuilder.ensure_index_exists(
            trace_path=trace_path,
            config=TraceIndexConfig(schema_version=1),
        )
```

- [ ] **Step 2: Run — expect fail**

Expected: test fails because the existing meta is loaded as-is without a version check.

- [ ] **Step 3: Patch `ensure_index_exists` to validate schema version of existing meta**

Replace the early-return branch:

```python
        if index_path.exists() and meta_path.exists():
            existing = TraceIndexMeta.model_validate_json(meta_path.read_text())
            if existing.schema_version != config.schema_version:
                raise ValueError(
                    f"existing index schema_version={existing.schema_version} "
                    f"does not match requested {config.schema_version}"
                )
            return index_path
```

- [ ] **Step 4: Run — expect pass**

Expected: `4 passed` total in the builder test file.

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): reject existing index with mismatched schema_version"
```

---

## Phase 3 — TraceStore

`TraceStore` reads the index once into memory and opens the trace file lazily for byte-range reads. It must not import anything outside stdlib + pydantic + `engine.traces.models.*` so it stays sandbox-safe.

### Task 3.1: `TraceStore.load` reads index file

**Files:**
- Create: `engine/engine/traces/trace_store.py`
- Create: `engine/tests/unit/traces/test_trace_store.py`

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore


@pytest.fixture
async def built_store(tmp_path: Path, fixtures_dir: Path) -> TraceStore:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    return TraceStore.load(trace_path=trace_path, index_path=index_path)


@pytest.mark.asyncio
async def test_load_sets_trace_count(built_store: TraceStore) -> None:
    assert built_store.trace_count == 3
```

- [ ] **Step 2: Run — expect ImportError**

- [ ] **Step 3: Implement — load only**

`engine/engine/traces/trace_store.py`:

```python
from __future__ import annotations

from pathlib import Path

from engine.traces.models.trace_index_models import TraceIndexRow


class TraceStore:
    def __init__(self, trace_path: Path, index_path: Path, rows: list[TraceIndexRow]) -> None:
        self._trace_path = trace_path
        self._index_path = index_path
        self._rows = rows
        self._rows_by_id: dict[str, TraceIndexRow] = {r.trace_id: r for r in rows}

    @classmethod
    def load(cls, trace_path: Path, index_path: Path) -> "TraceStore":
        raw = index_path.read_text().splitlines()
        rows = [TraceIndexRow.model_validate_json(line) for line in raw if line]
        return cls(trace_path=trace_path, index_path=index_path, rows=rows)

    @property
    def trace_count(self) -> int:
        return len(self._rows)
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO add engine/engine/traces/trace_store.py engine/tests/unit/traces/test_trace_store.py
git -C /home/declan/dev/HALO commit -m "feat(engine): TraceStore.load"
```

---

### Task 3.2: `view_trace` reads spans by byte offset

**Files:**
- Modify: `engine/engine/traces/trace_store.py`
- Modify: `engine/tests/unit/traces/test_trace_store.py`

- [ ] **Step 1: Append failing test**

```python
@pytest.mark.asyncio
async def test_view_trace_returns_span_records(built_store: TraceStore) -> None:
    view = built_store.view_trace("t-bbbb")
    assert view.trace_id == "t-bbbb"
    assert len(view.spans) == 2
    assert view.spans[0].span_id == "s-bbbb-1"
    assert view.spans[1].attributes["llm.model_name"] == "gpt-5.4"


@pytest.mark.asyncio
async def test_view_trace_unknown_raises(built_store: TraceStore) -> None:
    with pytest.raises(KeyError):
        built_store.view_trace("unknown")
```

- [ ] **Step 2: Run — expect fail (AttributeError)**

- [ ] **Step 3: Implement**

Append to `TraceStore`:

```python
    def view_trace(self, trace_id: str) -> "TraceView":
        from engine.traces.models.canonical_span import SpanRecord
        from engine.traces.models.trace_query_models import TraceView

        if trace_id not in self._rows_by_id:
            raise KeyError(trace_id)
        row = self._rows_by_id[trace_id]

        with self._trace_path.open("rb") as fh:
            spans: list[SpanRecord] = []
            for offset, length in zip(row.byte_offsets, row.byte_lengths, strict=True):
                fh.seek(offset)
                blob = fh.read(length)
                spans.append(SpanRecord.model_validate_json(blob))
        return TraceView(trace_id=trace_id, spans=spans)
```

The `from engine.traces...` imports are kept local to `view_trace` so `TraceStore.__init__` has minimal import surface — helpful for sandbox diagnostics.

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): TraceStore.view_trace via byte-range reads"
```

---

### Task 3.3: `query_traces` filters + paginates from index rows

**Files:**
- Modify: `engine/engine/traces/trace_store.py`
- Modify: `engine/tests/unit/traces/test_trace_store.py`

- [ ] **Step 1: Append failing tests**

```python
from engine.traces.models.trace_query_models import TraceFilters


@pytest.mark.asyncio
async def test_query_filter_has_errors(built_store: TraceStore) -> None:
    result = built_store.query_traces(
        filters=TraceFilters(has_errors=True),
        limit=10,
        offset=0,
    )
    assert result.total == 1
    assert len(result.traces) == 1
    assert result.traces[0].trace_id == "t-bbbb"


@pytest.mark.asyncio
async def test_query_filter_model_intersection(built_store: TraceStore) -> None:
    result = built_store.query_traces(
        filters=TraceFilters(model_names=["claude-haiku-4-5"]),
        limit=10,
        offset=0,
    )
    assert {t.trace_id for t in result.traces} == {"t-cccc"}
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

Append:

```python
    def query_traces(
        self,
        filters: "TraceFilters",
        limit: int = 50,
        offset: int = 0,
    ) -> "TraceQueryResult":
        from engine.traces.models.trace_query_models import TraceQueryResult, TraceSummary

        filtered = [row for row in self._rows if _matches_filters(row, filters)]
        summaries = [
            TraceSummary(
                trace_id=row.trace_id,
                span_count=row.span_count,
                start_time=row.start_time,
                end_time=row.end_time,
                has_errors=row.has_errors,
                service_names=row.service_names,
                model_names=row.model_names,
                total_input_tokens=row.total_input_tokens,
                total_output_tokens=row.total_output_tokens,
                agent_names=row.agent_names,
            )
            for row in filtered[offset : offset + limit]
        ]
        return TraceQueryResult(traces=summaries, total=len(filtered))
```

And add at module level:

```python
def _matches_filters(row: TraceIndexRow, filters: "TraceFilters") -> bool:
    if filters.has_errors is not None and row.has_errors != filters.has_errors:
        return False
    if filters.model_names is not None and not any(m in row.model_names for m in filters.model_names):
        return False
    if filters.service_names is not None and not any(s in row.service_names for s in filters.service_names):
        return False
    if filters.agent_names is not None and not any(a in row.agent_names for a in filters.agent_names):
        return False
    if filters.project_id is not None and row.project_id != filters.project_id:
        return False
    if filters.start_time_gte is not None and row.start_time < filters.start_time_gte:
        return False
    if filters.end_time_lte is not None and row.end_time > filters.end_time_lte:
        return False
    return True
```

Add the TYPE_CHECKING import at the top if it helps the editor; otherwise keep `"TraceFilters"` as a forward reference and import inside the method.

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): TraceStore.query_traces with filters"
```

---

### Task 3.4: `count_traces`

**Files:**
- Modify: `engine/engine/traces/trace_store.py`
- Modify: `engine/tests/unit/traces/test_trace_store.py`

- [ ] **Step 1: Test**

```python
@pytest.mark.asyncio
async def test_count_traces_with_and_without_filter(built_store: TraceStore) -> None:
    assert built_store.count_traces(TraceFilters()).total == 3
    assert built_store.count_traces(TraceFilters(has_errors=True)).total == 1
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

```python
    def count_traces(self, filters: "TraceFilters") -> "TraceCountResult":
        from engine.traces.models.trace_query_models import TraceCountResult

        total = sum(1 for row in self._rows if _matches_filters(row, filters))
        return TraceCountResult(total=total)
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): TraceStore.count_traces"
```

---

### Task 3.5: `get_overview`

**Files:**
- Modify: `engine/engine/traces/trace_store.py`
- Modify: `engine/tests/unit/traces/test_trace_store.py`

- [ ] **Step 1: Test**

```python
@pytest.mark.asyncio
async def test_overview_full(built_store: TraceStore) -> None:
    ov = built_store.get_overview(TraceFilters())
    assert ov.total_traces == 3
    assert ov.total_spans == 6
    assert "agent-a" in ov.agent_names
    assert "gpt-5.4" in ov.model_names
    assert ov.error_trace_count == 1
    assert ov.total_input_tokens == 100 + 200 + 30
    assert ov.total_output_tokens == 50 + 40 + 10
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

```python
    def get_overview(self, filters: "TraceFilters") -> "DatasetOverview":
        from engine.traces.models.trace_query_models import DatasetOverview

        rows = [r for r in self._rows if _matches_filters(r, filters)]
        if not rows:
            return DatasetOverview(
                total_traces=0, total_spans=0, earliest_start_time="",
                latest_end_time="", service_names=[], model_names=[], agent_names=[],
                error_trace_count=0, total_input_tokens=0, total_output_tokens=0,
            )

        services: set[str] = set()
        models: set[str] = set()
        agents: set[str] = set()
        for r in rows:
            services.update(r.service_names)
            models.update(r.model_names)
            agents.update(r.agent_names)

        return DatasetOverview(
            total_traces=len(rows),
            total_spans=sum(r.span_count for r in rows),
            earliest_start_time=min(r.start_time for r in rows),
            latest_end_time=max(r.end_time for r in rows),
            service_names=sorted(services),
            model_names=sorted(models),
            agent_names=sorted(agents),
            error_trace_count=sum(1 for r in rows if r.has_errors),
            total_input_tokens=sum(r.total_input_tokens for r in rows),
            total_output_tokens=sum(r.total_output_tokens for r in rows),
        )
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): TraceStore.get_overview"
```

---

### Task 3.6: `search_trace` substring search across span JSON

**Files:**
- Modify: `engine/engine/traces/trace_store.py`
- Modify: `engine/tests/unit/traces/test_trace_store.py`

- [ ] **Step 1: Test**

```python
@pytest.mark.asyncio
async def test_search_returns_matches(built_store: TraceStore) -> None:
    result = built_store.search_trace("t-bbbb", "tool failure")
    assert result.match_count >= 1
    assert any("tool failure" in m for m in result.matches)


@pytest.mark.asyncio
async def test_search_no_match(built_store: TraceStore) -> None:
    result = built_store.search_trace("t-aaaa", "nonexistent-needle")
    assert result.match_count == 0
    assert result.matches == []
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

```python
    def search_trace(self, trace_id: str, pattern: str) -> "TraceSearchResult":
        from engine.traces.models.trace_query_models import TraceSearchResult

        if trace_id not in self._rows_by_id:
            raise KeyError(trace_id)
        row = self._rows_by_id[trace_id]

        matches: list[str] = []
        with self._trace_path.open("rb") as fh:
            for offset, length in zip(row.byte_offsets, row.byte_lengths, strict=True):
                fh.seek(offset)
                blob = fh.read(length).decode("utf-8", errors="replace")
                if pattern in blob:
                    matches.append(blob)
        return TraceSearchResult(trace_id=trace_id, match_count=len(matches), matches=matches)
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): TraceStore.search_trace substring search"
```

---

### Task 3.7: `render_trace` returns prompt-friendly text with a byte budget

**Files:**
- Modify: `engine/engine/traces/trace_store.py`
- Modify: `engine/tests/unit/traces/test_trace_store.py`

- [ ] **Step 1: Test**

```python
@pytest.mark.asyncio
async def test_render_trace_under_budget(built_store: TraceStore) -> None:
    rendered = built_store.render_trace("t-aaaa", budget=4000)
    assert "t-aaaa" in rendered
    assert "s-aaaa-1" in rendered
    assert "s-aaaa-2" in rendered


@pytest.mark.asyncio
async def test_render_trace_truncates_when_over_budget(built_store: TraceStore) -> None:
    rendered = built_store.render_trace("t-aaaa", budget=200)
    assert rendered.endswith("... [truncated]")
    assert len(rendered) <= 200 + len("... [truncated]")
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

```python
    def render_trace(self, trace_id: str, budget: int) -> str:
        view = self.view_trace(trace_id)
        lines: list[str] = [f"trace_id: {trace_id}", f"spans: {len(view.spans)}"]
        for s in view.spans:
            lines.append(
                f"- span_id={s.span_id} parent={s.parent_span_id or '∅'} "
                f"name={s.name} kind={s.kind} status={s.status.code}"
            )
            lines.append(f"  start={s.start_time} end={s.end_time}")
            model = s.attributes.get("inference.llm.model_name") or s.attributes.get("llm.model_name")
            if model:
                lines.append(f"  model={model}")
            in_tok = s.attributes.get("inference.llm.input_tokens")
            out_tok = s.attributes.get("inference.llm.output_tokens")
            if in_tok is not None or out_tok is not None:
                lines.append(f"  tokens: input={in_tok} output={out_tok}")

        rendered = "\n".join(lines)
        if len(rendered) > budget:
            return rendered[:budget] + "... [truncated]"
        return rendered
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): TraceStore.render_trace with budget"
```

---

## Phase 4 — Tool protocol and trace tools

### Task 4.1: `ToolContext` + `EngineTool` protocol + SDK adapter

**Context:** Tool functions operate against `TraceStore`, `AgentContext`, and `SandboxRunner`. We carry these in a `ToolContext` passed through the OpenAI Agents SDK's `RunContextWrapper[EngineRunContext]`. The adapter converts `EngineTool` instances into SDK `FunctionTool` objects.

**Files:**
- Create: `engine/engine/tools/__init__.py` (empty)
- Create: `engine/engine/tools/tool_protocol.py`
- Create: `engine/tests/unit/tools/__init__.py` (empty)
- Create: `engine/tests/unit/tools/test_tool_protocol.py`

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

from pydantic import BaseModel

from engine.tools.tool_protocol import EngineTool, ToolContext


class _EchoArgs(BaseModel):
    value: str


class _EchoResult(BaseModel):
    value: str


class _EchoTool:
    name = "echo"
    description = "Echo."
    arguments_model = _EchoArgs
    result_model = _EchoResult

    async def run(self, tool_context: ToolContext, arguments: _EchoArgs) -> _EchoResult:
        return _EchoResult(value=arguments.value)


async def test_engine_tool_runtime_conforms() -> None:
    tool: EngineTool = _EchoTool()
    ctx = ToolContext.model_construct()  # placeholder ctx
    result = await tool.run(ctx, _EchoArgs(value="x"))
    assert isinstance(result, _EchoResult)
    assert result.value == "x"
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

`engine/engine/tools/tool_protocol.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from engine.agents.agent_context import AgentContext
    from engine.agents.agent_execution import AgentExecution
    from engine.agents.engine_output_bus import EngineOutputBus
    from engine.agents.engine_run_state import EngineRunState
    from engine.sandbox.sandbox_runner import SandboxRunner
    from engine.traces.trace_store import TraceStore


class ToolContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    run_state: "EngineRunState | None" = None
    trace_store: "TraceStore | None" = None
    agent_context: "AgentContext | None" = None
    agent_execution: "AgentExecution | None" = None
    output_bus: "EngineOutputBus | None" = None
    sandbox_runner: "SandboxRunner | None" = None

    def require_trace_store(self) -> "TraceStore":
        if self.trace_store is None:
            raise RuntimeError("ToolContext.trace_store required")
        return self.trace_store

    def require_agent_context(self) -> "AgentContext":
        if self.agent_context is None:
            raise RuntimeError("ToolContext.agent_context required")
        return self.agent_context


@runtime_checkable
class EngineTool(Protocol):
    name: str
    description: str
    arguments_model: type[BaseModel]
    result_model: type[BaseModel]

    async def run(self, tool_context: ToolContext, arguments: Any) -> BaseModel: ...
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO add engine/engine/tools engine/tests/unit/tools
git -C /home/declan/dev/HALO commit -m "feat(engine): EngineTool protocol + ToolContext"
```

---

### Task 4.2: SDK function-tool adapter

**Files:**
- Modify: `engine/engine/tools/tool_protocol.py`
- Create: `engine/tests/unit/tools/test_sdk_adapter.py`

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

from pydantic import BaseModel

from engine.tools.tool_protocol import ToolContext, to_sdk_function_tool


class _Args(BaseModel):
    value: str


class _Result(BaseModel):
    echoed: str


class _Echo:
    name = "echo"
    description = "Echo a value."
    arguments_model = _Args
    result_model = _Result

    async def run(self, tool_context: ToolContext, arguments: _Args) -> _Result:
        return _Result(echoed=arguments.value)


def test_adapter_produces_sdk_function_tool() -> None:
    from agents import FunctionTool

    sdk_tool = to_sdk_function_tool(_Echo(), context_factory=ToolContext.model_construct)
    assert isinstance(sdk_tool, FunctionTool)
    assert sdk_tool.name == "echo"
    assert "Echo" in (sdk_tool.description or "")
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement adapter**

Append to `engine/engine/tools/tool_protocol.py`:

```python
import json
from collections.abc import Callable

from agents import FunctionTool, RunContextWrapper


def to_sdk_function_tool(
    tool: EngineTool,
    *,
    context_factory: Callable[[RunContextWrapper[Any]], ToolContext],
) -> FunctionTool:
    arguments_model = tool.arguments_model

    async def _invoke(ctx: RunContextWrapper[Any], raw_arguments: str) -> str:
        parsed = arguments_model.model_validate_json(raw_arguments or "{}")
        tool_context = context_factory(ctx)
        result = await tool.run(tool_context, parsed)
        return result.model_dump_json()

    schema = arguments_model.model_json_schema()
    return FunctionTool(
        name=tool.name,
        description=tool.description,
        params_json_schema=schema,
        on_invoke_tool=_invoke,
    )
```

Note: the `context_factory` takes the SDK's `RunContextWrapper` so the caller can pull `EngineRunState` off `ctx.context` and build a `ToolContext` with the right slices for that run.

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): SDK FunctionTool adapter for EngineTool"
```

---

### Task 4.3: Trace tools — five tool classes backed by `TraceStore`

**Files:**
- Create: `engine/engine/tools/trace_tools.py`
- Create: `engine/tests/unit/tools/test_trace_tools.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from engine.tools.tool_protocol import ToolContext
from engine.tools.trace_tools import (
    CountTracesTool,
    GetDatasetOverviewTool,
    QueryTracesTool,
    SearchTraceTool,
    ViewTraceTool,
)
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.models.trace_query_models import (
    CountTracesArguments,
    DatasetOverviewArguments,
    QueryTracesArguments,
    SearchTraceArguments,
    TraceFilters,
    ViewTraceArguments,
)
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore


@pytest.fixture
async def ctx(tmp_path: Path, fixtures_dir: Path) -> ToolContext:
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    store = TraceStore.load(trace_path=trace_path, index_path=index_path)
    return ToolContext(trace_store=store)


@pytest.mark.asyncio
async def test_query_traces_tool(ctx: ToolContext) -> None:
    tool = QueryTracesTool()
    result = await tool.run(ctx, QueryTracesArguments(filters=TraceFilters()))
    assert result.result.total == 3


@pytest.mark.asyncio
async def test_count_traces_tool(ctx: ToolContext) -> None:
    tool = CountTracesTool()
    result = await tool.run(ctx, CountTracesArguments(filters=TraceFilters(has_errors=True)))
    assert result.result.total == 1


@pytest.mark.asyncio
async def test_view_trace_tool(ctx: ToolContext) -> None:
    tool = ViewTraceTool()
    result = await tool.run(ctx, ViewTraceArguments(trace_id="t-aaaa"))
    assert len(result.result.spans) == 2


@pytest.mark.asyncio
async def test_search_trace_tool(ctx: ToolContext) -> None:
    tool = SearchTraceTool()
    result = await tool.run(ctx, SearchTraceArguments(trace_id="t-bbbb", pattern="tool failure"))
    assert result.result.match_count >= 1


@pytest.mark.asyncio
async def test_overview_tool(ctx: ToolContext) -> None:
    tool = GetDatasetOverviewTool()
    result = await tool.run(ctx, DatasetOverviewArguments(filters=TraceFilters()))
    assert result.result.total_traces == 3
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

`engine/engine/tools/trace_tools.py`:

```python
from __future__ import annotations

from engine.tools.tool_protocol import ToolContext
from engine.traces.models.trace_query_models import (
    CountTracesArguments,
    CountTracesResult,
    DatasetOverviewArguments,
    DatasetOverviewResult,
    QueryTracesArguments,
    QueryTracesResult,
    SearchTraceArguments,
    SearchTraceResult,
    ViewTraceArguments,
    ViewTraceResult,
)


class GetDatasetOverviewTool:
    name = "get_dataset_overview"
    description = "Return high-level stats about the trace dataset: counts, services, models, totals."
    arguments_model = DatasetOverviewArguments
    result_model = DatasetOverviewResult

    async def run(self, tool_context: ToolContext, arguments: DatasetOverviewArguments) -> DatasetOverviewResult:
        store = tool_context.require_trace_store()
        return DatasetOverviewResult(result=store.get_overview(arguments.filters))


class QueryTracesTool:
    name = "query_traces"
    description = "List trace summaries matching filters with pagination."
    arguments_model = QueryTracesArguments
    result_model = QueryTracesResult

    async def run(self, tool_context: ToolContext, arguments: QueryTracesArguments) -> QueryTracesResult:
        store = tool_context.require_trace_store()
        return QueryTracesResult(
            result=store.query_traces(
                filters=arguments.filters,
                limit=arguments.limit,
                offset=arguments.offset,
            )
        )


class CountTracesTool:
    name = "count_traces"
    description = "Count traces matching filters."
    arguments_model = CountTracesArguments
    result_model = CountTracesResult

    async def run(self, tool_context: ToolContext, arguments: CountTracesArguments) -> CountTracesResult:
        store = tool_context.require_trace_store()
        return CountTracesResult(result=store.count_traces(arguments.filters))


class ViewTraceTool:
    name = "view_trace"
    description = "Return all spans of a trace by id."
    arguments_model = ViewTraceArguments
    result_model = ViewTraceResult

    async def run(self, tool_context: ToolContext, arguments: ViewTraceArguments) -> ViewTraceResult:
        store = tool_context.require_trace_store()
        return ViewTraceResult(result=store.view_trace(arguments.trace_id))


class SearchTraceTool:
    name = "search_trace"
    description = "Substring search inside the spans of one trace."
    arguments_model = SearchTraceArguments
    result_model = SearchTraceResult

    async def run(self, tool_context: ToolContext, arguments: SearchTraceArguments) -> SearchTraceResult:
        store = tool_context.require_trace_store()
        return SearchTraceResult(result=store.search_trace(arguments.trace_id, arguments.pattern))
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): trace tools backed by TraceStore"
```

---

## Phase 5 — AgentContext and context tools

### Task 5.1: `AgentContextItem`

**Files:**
- Create: `engine/engine/agents/agent_context_items.py`
- Create: `engine/tests/unit/agents/test_agent_context_items.py`

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

from engine.agents.agent_context_items import AgentContextItem
from engine.models.messages import AgentToolCall, AgentToolFunction


def test_user_item() -> None:
    item = AgentContextItem(item_id="msg_1", role="user", content="hi")
    assert item.is_compacted is False
    assert item.agent_id is None


def test_assistant_tool_call_item_with_lineage() -> None:
    item = AgentContextItem(
        item_id="msg_2",
        role="assistant",
        content=None,
        tool_calls=[
            AgentToolCall(id="c1", function=AgentToolFunction(name="x", arguments="{}"))
        ],
        agent_id="root",
        parent_agent_id=None,
    )
    assert item.tool_calls is not None


def test_compacted_item() -> None:
    item = AgentContextItem(
        item_id="msg_3", role="user", content="hi",
        is_compacted=True, compaction_summary="User said hi.",
    )
    assert item.is_compacted is True
    assert item.compaction_summary == "User said hi."
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

`engine/engine/agents/agent_context_items.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from engine.models.messages import AgentToolCall, MessageContent


class AgentContextItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    role: Literal["system", "user", "assistant", "tool"]
    content: MessageContent = None
    tool_calls: list[AgentToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    is_compacted: bool = False
    compaction_summary: str | None = None
    agent_id: str | None = None
    parent_agent_id: str | None = None
    parent_tool_call_id: str | None = None
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): AgentContextItem"
```

---

### Task 5.2: `AgentContext.append` / `get_item` / `to_messages_array` (uncompacted)

**Files:**
- Create: `engine/engine/agents/agent_context.py`
- Create: `engine/tests/unit/agents/test_agent_context.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

import pytest

from engine.agents.agent_context import AgentContext
from engine.agents.agent_context_items import AgentContextItem
from engine.model_config import ModelConfig
from engine.models.messages import AgentToolCall, AgentToolFunction


def _ctx() -> AgentContext:
    return AgentContext(
        items=[],
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        text_message_compaction_keep_last_messages=2,
        tool_call_compaction_keep_last_messages=2,
    )


def test_append_and_get_item() -> None:
    ctx = _ctx()
    ctx.append(AgentContextItem(item_id="1", role="user", content="hi"))
    assert ctx.get_item("1").content == "hi"


def test_get_item_missing_raises() -> None:
    ctx = _ctx()
    with pytest.raises(KeyError):
        ctx.get_item("nope")


def test_to_messages_array_uncompacted_user() -> None:
    ctx = _ctx()
    ctx.append(AgentContextItem(item_id="1", role="user", content="hi"))
    msgs = ctx.to_messages_array()
    assert len(msgs) == 1
    assert msgs[0].role == "user"
    assert msgs[0].content == "hi"


def test_to_messages_array_assistant_tool_call_item() -> None:
    ctx = _ctx()
    ctx.append(AgentContextItem(
        item_id="2",
        role="assistant",
        content=None,
        tool_calls=[AgentToolCall(id="c1", function=AgentToolFunction(name="x", arguments="{}"))],
    ))
    ctx.append(AgentContextItem(
        item_id="3",
        role="tool",
        content="ok",
        tool_call_id="c1",
        name="x",
    ))
    msgs = ctx.to_messages_array()
    assert msgs[0].role == "assistant" and msgs[0].tool_calls is not None
    assert msgs[1].role == "tool" and msgs[1].tool_call_id == "c1"
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement (compaction branch stubbed to raise so tests above still pass)**

`engine/engine/agents/agent_context.py`:

```python
from __future__ import annotations

from engine.agents.agent_context_items import AgentContextItem
from engine.model_config import ModelConfig
from engine.models.messages import AgentMessage


class AgentContext:
    def __init__(
        self,
        items: list[AgentContextItem],
        compaction_model: ModelConfig,
        text_message_compaction_keep_last_messages: int,
        tool_call_compaction_keep_last_messages: int,
    ) -> None:
        self.items = list(items)
        self.compaction_model = compaction_model
        self.text_message_compaction_keep_last_messages = text_message_compaction_keep_last_messages
        self.tool_call_compaction_keep_last_messages = tool_call_compaction_keep_last_messages
        self._index: dict[str, AgentContextItem] = {item.item_id: item for item in items}

    def append(self, item: AgentContextItem) -> None:
        self.items.append(item)
        self._index[item.item_id] = item

    def get_item(self, item_id: str) -> AgentContextItem:
        return self._index[item_id]

    def to_messages_array(self) -> list[AgentMessage]:
        return [_render_item(item) for item in self.items]


def _render_item(item: AgentContextItem) -> AgentMessage:
    if not item.is_compacted:
        return AgentMessage(
            role=item.role,
            content=item.content,
            tool_calls=item.tool_calls,
            tool_call_id=item.tool_call_id,
            name=item.name,
        )

    # Compacted branch — implemented in Task 5.4.
    summary = item.compaction_summary or ""
    if item.role == "user":
        return AgentMessage(role="user", content=f"Compacted message (id: {item.item_id}): {summary}")
    if item.role == "assistant":
        return AgentMessage(
            role="assistant",
            content=f"Compacted tool calls (id: {item.item_id}): {summary}",
        )
    if item.role == "tool":
        tool_name = item.name or "tool"
        return AgentMessage(
            role="assistant",
            content=f"Compacted tool result (id: {item.item_id}, tool: {tool_name}): {summary}",
        )
    # System messages are never compacted; defensively return original content.
    return AgentMessage(role=item.role, content=item.content)
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): AgentContext append/get_item/to_messages_array"
```

---

### Task 5.3: `compact_old_items` — per-item eligibility after each turn

**Files:**
- Modify: `engine/engine/agents/agent_context.py`
- Create: `engine/engine/agents/prompt_templates.py` (partial — add `COMPACTION_PROMPT` template)
- Modify: `engine/tests/unit/agents/test_agent_context.py`

- [ ] **Step 1: Write failing test with a stub compactor**

Append to the context test file:

```python
from engine.agents.agent_context import AgentContext
from engine.agents.agent_context_items import AgentContextItem


class _StubCompactor:
    def __init__(self) -> None:
        self.calls: list[AgentContextItem] = []

    async def compact(self, item: AgentContextItem) -> str:
        self.calls.append(item)
        return f"SUMMARY({item.item_id})"


@pytest.mark.asyncio
async def test_compact_old_items_only_touches_eligible_text() -> None:
    ctx = AgentContext(
        items=[],
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        text_message_compaction_keep_last_messages=2,
        tool_call_compaction_keep_last_messages=2,
    )
    # 4 text messages: oldest 2 should compact, newest 2 stay
    for i in range(4):
        ctx.append(AgentContextItem(item_id=f"t{i}", role="user", content=f"msg {i}"))

    stub = _StubCompactor()
    await ctx.compact_old_items(compactor=stub.compact)

    ids_compacted = {call.item_id for call in stub.calls}
    assert ids_compacted == {"t0", "t1"}
    assert ctx.get_item("t0").is_compacted is True
    assert ctx.get_item("t0").compaction_summary == "SUMMARY(t0)"
    assert ctx.get_item("t3").is_compacted is False


@pytest.mark.asyncio
async def test_compact_old_items_separate_thresholds_for_tools() -> None:
    ctx = AgentContext(
        items=[],
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        text_message_compaction_keep_last_messages=10,
        tool_call_compaction_keep_last_messages=1,
    )
    # 3 tool-call pairs: an assistant tool_call message + a tool result
    for i in range(3):
        ctx.append(AgentContextItem(
            item_id=f"a{i}",
            role="assistant",
            content=None,
            tool_calls=[AgentToolCall(id=f"c{i}", function=AgentToolFunction(name="x", arguments="{}"))],
        ))
        ctx.append(AgentContextItem(
            item_id=f"r{i}",
            role="tool",
            content="ok",
            tool_call_id=f"c{i}",
            name="x",
        ))

    stub = _StubCompactor()
    await ctx.compact_old_items(compactor=stub.compact)

    ids_compacted = {call.item_id for call in stub.calls}
    # Keep last 1 tool-call-related item. The 5 oldest are eligible (a0,r0,a1,r1,a2).
    assert ids_compacted == {"a0", "r0", "a1", "r1", "a2"}


@pytest.mark.asyncio
async def test_compact_old_items_skips_system_and_already_compacted() -> None:
    ctx = AgentContext(
        items=[],
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        text_message_compaction_keep_last_messages=0,
        tool_call_compaction_keep_last_messages=0,
    )
    ctx.append(AgentContextItem(item_id="s", role="system", content="sys"))
    ctx.append(AgentContextItem(item_id="u1", role="user", content="hi", is_compacted=True, compaction_summary="x"))
    ctx.append(AgentContextItem(item_id="u2", role="user", content="hello"))
    stub = _StubCompactor()
    await ctx.compact_old_items(compactor=stub.compact)
    compacted_ids = {c.item_id for c in stub.calls}
    assert compacted_ids == {"u2"}
    assert ctx.get_item("s").is_compacted is False
```

- [ ] **Step 2: Run — expect AttributeError**

- [ ] **Step 3: Implement**

Add to `AgentContext`:

```python
    async def compact_old_items(self, compactor: "Compactor") -> None:
        # Classify items into streams: text (user/assistant-text) vs tool-related (assistant-with-tool-calls + tool).
        text_positions: list[int] = []
        tool_positions: list[int] = []
        for idx, item in enumerate(self.items):
            if item.is_compacted or item.role == "system":
                continue
            if _is_tool_related(item):
                tool_positions.append(idx)
            else:
                text_positions.append(idx)

        eligible: list[int] = []
        if len(text_positions) > self.text_message_compaction_keep_last_messages:
            cutoff = len(text_positions) - self.text_message_compaction_keep_last_messages
            eligible.extend(text_positions[:cutoff])
        if len(tool_positions) > self.tool_call_compaction_keep_last_messages:
            cutoff = len(tool_positions) - self.tool_call_compaction_keep_last_messages
            eligible.extend(tool_positions[:cutoff])

        for idx in sorted(eligible):
            item = self.items[idx]
            summary = await compactor(item)
            self.items[idx] = item.model_copy(update={"is_compacted": True, "compaction_summary": summary})
            self._index[item.item_id] = self.items[idx]


def _is_tool_related(item: AgentContextItem) -> bool:
    if item.role == "tool":
        return True
    if item.role == "assistant" and item.tool_calls:
        return True
    return False
```

At the top of the file add:

```python
from collections.abc import Awaitable, Callable
from typing import TypeAlias

Compactor: TypeAlias = Callable[[AgentContextItem], Awaitable[str]]
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): AgentContext.compact_old_items per-item eligibility"
```

---

### Task 5.4: Prompt templates + live compactor using `openai` SDK

**Files:**
- Create: `engine/engine/agents/prompt_templates.py`
- Create: `engine/tests/unit/agents/test_prompt_templates.py`

- [ ] **Step 1: Write failing tests (templates only — live LLM tested at integration time)**

```python
from __future__ import annotations

from engine.agents.prompt_templates import (
    COMPACTION_SYSTEM_PROMPT,
    FINAL_SENTINEL,
    ROOT_SYSTEM_PROMPT_TEMPLATE,
    SUBAGENT_SYSTEM_PROMPT_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
    render_root_system_prompt,
    render_subagent_system_prompt,
)


def test_final_sentinel_constant() -> None:
    assert FINAL_SENTINEL == "<final/>"


def test_root_prompt_includes_sentinel_instruction() -> None:
    text = render_root_system_prompt(
        user_instructions="Investigate failing traces.",
        maximum_depth=2,
        maximum_parallel_subagents=4,
    )
    assert FINAL_SENTINEL in text
    assert "Investigate failing traces." in text


def test_subagent_prompt_reports_depth() -> None:
    text = render_subagent_system_prompt(
        user_instructions="You are a sub.",
        depth=1,
        maximum_depth=2,
        maximum_parallel_subagents=4,
    )
    assert "depth=1" in text
    assert "maximum_depth=2" in text


def test_compaction_and_synthesis_prompts_are_strings() -> None:
    assert isinstance(COMPACTION_SYSTEM_PROMPT, str) and COMPACTION_SYSTEM_PROMPT
    assert isinstance(SYNTHESIS_SYSTEM_PROMPT, str) and SYNTHESIS_SYSTEM_PROMPT
    assert "<final/>" in ROOT_SYSTEM_PROMPT_TEMPLATE
    assert "{user_instructions}" in SUBAGENT_SYSTEM_PROMPT_TEMPLATE
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

`engine/engine/agents/prompt_templates.py`:

```python
from __future__ import annotations

FINAL_SENTINEL = "<final/>"

ROOT_SYSTEM_PROMPT_TEMPLATE = """\
You are the root agent in the HALO engine. You explore OTel trace data
using the provided tools: dataset overview, query/count/view/search traces,
get_context_item, synthesize_traces, run_code, and call_subagent.

Depth rules:
- You are at depth=0.
- maximum_depth={maximum_depth}. Subagents you spawn are at depth=1.
- Spawn at most {maximum_parallel_subagents} subagents concurrently.

Output rules:
- When you are finished and have produced your final answer, end that
  assistant message with a single line containing only: <final/>
- Do not emit <final/> in intermediate messages.

User instructions:
{user_instructions}
"""

SUBAGENT_SYSTEM_PROMPT_TEMPLATE = """\
You are a HALO subagent at depth={depth} of {maximum_depth}. You answer a
question delegated to you by a parent agent. You have trace tools and, if
your depth permits, a call_subagent tool.

When finished, return a concise answer. Do not emit <final/> — that
sentinel is reserved for the root agent.

User instructions:
{user_instructions}
"""

COMPACTION_SYSTEM_PROMPT = """\
You summarize a single conversation item for storage. Preserve tool names,
argument shapes, and key result facts that future reasoning might need.
Return a short plain-text summary — no JSON wrapping, no surrounding prose.
"""

SYNTHESIS_SYSTEM_PROMPT = """\
You synthesize findings across a set of traces into a short plain-text
summary suitable as a tool result. Include concrete trace ids, error
patterns, model names, and token counts when available.
"""


def render_root_system_prompt(
    *,
    user_instructions: str,
    maximum_depth: int,
    maximum_parallel_subagents: int,
) -> str:
    return ROOT_SYSTEM_PROMPT_TEMPLATE.format(
        user_instructions=user_instructions,
        maximum_depth=maximum_depth,
        maximum_parallel_subagents=maximum_parallel_subagents,
    )


def render_subagent_system_prompt(
    *,
    user_instructions: str,
    depth: int,
    maximum_depth: int,
    maximum_parallel_subagents: int,
) -> str:
    return SUBAGENT_SYSTEM_PROMPT_TEMPLATE.format(
        user_instructions=user_instructions,
        depth=depth,
        maximum_depth=maximum_depth,
        maximum_parallel_subagents=maximum_parallel_subagents,
    )
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): prompt templates + final sentinel"
```

---

### Task 5.5: `get_context_item` tool

**Files:**
- Create: `engine/engine/tools/agent_context_tools.py`
- Create: `engine/tests/unit/tools/test_agent_context_tools.py`

- [ ] **Step 1: Test**

```python
from __future__ import annotations

import pytest

from engine.agents.agent_context import AgentContext
from engine.agents.agent_context_items import AgentContextItem
from engine.model_config import ModelConfig
from engine.tools.agent_context_tools import (
    GetContextItemArguments,
    GetContextItemTool,
)
from engine.tools.tool_protocol import ToolContext


@pytest.mark.asyncio
async def test_get_context_item_returns_full_stored_item() -> None:
    agent_context = AgentContext(
        items=[],
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        text_message_compaction_keep_last_messages=2,
        tool_call_compaction_keep_last_messages=2,
    )
    agent_context.append(AgentContextItem(
        item_id="m1", role="user", content="hi",
        is_compacted=True, compaction_summary="user said hi",
    ))
    ctx = ToolContext(agent_context=agent_context)

    tool = GetContextItemTool()
    result = await tool.run(ctx, GetContextItemArguments(item_id="m1"))
    assert result.item.item_id == "m1"
    # Original content survives alongside the summary
    assert result.item.content == "hi"
    assert result.item.compaction_summary == "user said hi"
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

`engine/engine/tools/agent_context_tools.py`:

```python
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from engine.agents.agent_context_items import AgentContextItem
from engine.tools.tool_protocol import ToolContext


class GetContextItemArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str


class GetContextItemResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: AgentContextItem


class GetContextItemTool:
    name = "get_context_item"
    description = "Return the full stored context item by id, including original content and compaction summary."
    arguments_model = GetContextItemArguments
    result_model = GetContextItemResult

    async def run(self, tool_context: ToolContext, arguments: GetContextItemArguments) -> GetContextItemResult:
        agent_context = tool_context.require_agent_context()
        item = agent_context.get_item(arguments.item_id)
        return GetContextItemResult(item=item)
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): get_context_item tool"
```

---

## Phase 6 — Bus, run state, execution summaries, and the event mapper

### Task 6.1: `EngineOutputBus`

**Files:**
- Create: `engine/engine/agents/engine_output_bus.py`
- Create: `engine/tests/unit/agents/test_engine_output_bus.py`

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

import pytest

from engine.agents.engine_output_bus import EngineOutputBus
from engine.models.engine_output import AgentOutputItem, AgentTextDelta
from engine.models.messages import AgentMessage


def _msg(agent: str = "root", text: str = "hi") -> AgentOutputItem:
    return AgentOutputItem(
        sequence=0,
        agent_id=agent,
        parent_agent_id=None,
        parent_tool_call_id=None,
        agent_name=agent,
        depth=0,
        item=AgentMessage(role="assistant", content=text),
    )


@pytest.mark.asyncio
async def test_bus_assigns_monotonic_sequences() -> None:
    bus = EngineOutputBus()
    a = await bus.emit(_msg(text="a"))
    b = await bus.emit(_msg(text="b"))
    assert a.sequence == 0
    assert b.sequence == 1


@pytest.mark.asyncio
async def test_bus_stream_emits_and_closes() -> None:
    bus = EngineOutputBus()
    await bus.emit(_msg(text="a"))
    await bus.close()
    collected = [item async for item in bus.stream()]
    assert len(collected) == 1


@pytest.mark.asyncio
async def test_bus_fail_propagates_after_drain() -> None:
    bus = EngineOutputBus()
    await bus.emit(_msg(text="a"))
    await bus.fail(RuntimeError("boom"))
    events: list = []
    with pytest.raises(RuntimeError, match="boom"):
        async for ev in bus.stream():
            events.append(ev)
    assert len(events) == 1


@pytest.mark.asyncio
async def test_bus_handles_deltas_and_items() -> None:
    bus = EngineOutputBus()
    await bus.emit(_msg(text="full"))
    await bus.emit(AgentTextDelta(
        sequence=0,
        agent_id="root",
        parent_agent_id=None,
        parent_tool_call_id=None,
        depth=0,
        item_id="x",
        text_delta="par",
    ))
    await bus.close()
    events = [ev async for ev in bus.stream()]
    assert [type(ev).__name__ for ev in events] == ["AgentOutputItem", "AgentTextDelta"]
    assert events[1].sequence == 1
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

`engine/engine/agents/engine_output_bus.py`:

```python
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from engine.models.engine_output import AgentOutputItem, AgentTextDelta, EngineStreamEvent


class _BusSignal:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error


class EngineOutputBus:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[EngineStreamEvent | _BusSignal] = asyncio.Queue()
        self._next_sequence = 0
        self._lock = asyncio.Lock()

    async def emit(self, item: EngineStreamEvent) -> EngineStreamEvent:
        async with self._lock:
            sequenced = item.model_copy(update={"sequence": self._next_sequence})
            self._next_sequence += 1
        await self._queue.put(sequenced)
        return sequenced

    async def close(self) -> None:
        await self._queue.put(_BusSignal())

    async def fail(self, error: BaseException) -> None:
        await self._queue.put(_BusSignal(error=error))

    async def stream(self) -> AsyncIterator[EngineStreamEvent]:
        while True:
            event = await self._queue.get()
            if isinstance(event, _BusSignal):
                if event.error is not None:
                    raise event.error
                return
            yield event
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO add engine/engine/agents/engine_output_bus.py engine/tests/unit/agents/test_engine_output_bus.py
git -C /home/declan/dev/HALO commit -m "feat(engine): EngineOutputBus with monotonic sequences + fail"
```

---

### Task 6.2: `AgentExecution`

Tracks per-agent-run state: id, depth, lineage, consecutive LLM failure counter, cumulative tool-call counter.

**Files:**
- Create: `engine/engine/agents/agent_execution.py`
- Create: `engine/tests/unit/agents/test_agent_execution.py`

- [ ] **Step 1: Test**

```python
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
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

`engine/engine/agents/agent_execution.py`:

```python
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
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): AgentExecution dataclass"
```

---

### Task 6.3: `EngineRunState`

**Files:**
- Create: `engine/engine/agents/engine_run_state.py`
- Create: `engine/tests/unit/agents/test_engine_run_state.py`

- [ ] **Step 1: Test**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from engine.agents.agent_config import AgentConfig
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.engine_run_state import EngineRunState
from engine.engine_config import EngineConfig
from engine.model_config import ModelConfig
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore


def _cfg() -> EngineConfig:
    ac = AgentConfig(
        name="root", instructions="",
        model=ModelConfig(name="claude-sonnet-4-5"), maximum_turns=10,
    )
    return EngineConfig(
        root_agent=ac, subagent=ac,
        synthesis_model=ModelConfig(name="claude-haiku-4-5"),
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
    )


@pytest.mark.asyncio
async def test_run_state_holds_registries(tmp_path: Path, fixtures_dir: Path) -> None:
    trace_path = tmp_path / "t.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = await TraceIndexBuilder.ensure_index_exists(trace_path=trace_path, config=TraceIndexConfig())
    store = TraceStore.load(trace_path=trace_path, index_path=index_path)

    state = EngineRunState(trace_store=store, output_bus=EngineOutputBus(), config=_cfg())

    exec_ = AgentExecution(
        agent_id="root", agent_name="root", depth=0,
        parent_agent_id=None, parent_tool_call_id=None,
    )
    state.register(exec_)
    assert state.get_by_agent_id("root") is exec_

    # Register a child keyed by tool_call_id too
    child = AgentExecution(
        agent_id="sub1", agent_name="sub", depth=1,
        parent_agent_id="root", parent_tool_call_id="call_xyz",
    )
    state.register(child)
    assert state.get_by_tool_call_id("call_xyz") is child
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

`engine/engine/agents/engine_run_state.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.engine_config import EngineConfig
from engine.traces.trace_store import TraceStore


@dataclass
class EngineRunState:
    trace_store: TraceStore
    output_bus: EngineOutputBus
    config: EngineConfig
    executions_by_agent_id: dict[str, AgentExecution] = field(default_factory=dict)
    executions_by_tool_call_id: dict[str, AgentExecution] = field(default_factory=dict)

    def register(self, execution: AgentExecution) -> None:
        self.executions_by_agent_id[execution.agent_id] = execution
        if execution.parent_tool_call_id is not None:
            self.executions_by_tool_call_id[execution.parent_tool_call_id] = execution

    def get_by_agent_id(self, agent_id: str) -> AgentExecution:
        return self.executions_by_agent_id[agent_id]

    def get_by_tool_call_id(self, tool_call_id: str) -> AgentExecution:
        return self.executions_by_tool_call_id[tool_call_id]
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): EngineRunState registries"
```

---

### Task 6.4: `OpenAiEventMapper` — convert SDK events to context items, output items, and deltas

**Context:** Keep this class narrow. Given the SDK's `RunItem` and `RawResponsesStreamEvent` types, produce three outputs:
1. An optional `AgentContextItem` to append to the agent's `AgentContext`.
2. An optional `AgentOutputItem` to emit on the bus.
3. An optional `AgentTextDelta` (only from `response.output_text.delta` raw events).

The mapper also detects the `<final/>` sentinel in a root assistant message and strips it while marking `final=True` on the output item.

**Files:**
- Create: `engine/engine/agents/openai_event_mapper.py`
- Create: `engine/tests/unit/agents/test_openai_event_mapper.py`

- [ ] **Step 1: Write failing tests using synthetic SDK event shapes**

```python
from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.agents.agent_execution import AgentExecution
from engine.agents.openai_event_mapper import OpenAiEventMapper
from engine.models.engine_output import AgentOutputItem


def _exec() -> AgentExecution:
    return AgentExecution(
        agent_id="root", agent_name="root", depth=0,
        parent_agent_id=None, parent_tool_call_id=None,
    )


def test_assistant_text_item_plain() -> None:
    mapper = OpenAiEventMapper()
    raw = SimpleNamespace(
        type="message_output_item",
        message=SimpleNamespace(
            id="msg_1",
            role="assistant",
            content=[SimpleNamespace(type="output_text", text="Done.")],
            tool_calls=None,
        ),
    )
    execution = _exec()
    mapped = mapper.to_mapped_event(raw, execution=execution, is_root=True)
    assert mapped.context_item is not None
    assert mapped.context_item.role == "assistant"
    assert mapped.output_item is not None
    assert isinstance(mapped.output_item, AgentOutputItem)
    assert mapped.output_item.final is False


def test_root_assistant_final_sentinel_strips_and_sets_final() -> None:
    mapper = OpenAiEventMapper()
    raw = SimpleNamespace(
        type="message_output_item",
        message=SimpleNamespace(
            id="msg_2",
            role="assistant",
            content=[SimpleNamespace(type="output_text", text="Final answer.\n<final/>")],
            tool_calls=None,
        ),
    )
    execution = _exec()
    mapped = mapper.to_mapped_event(raw, execution=execution, is_root=True)
    assert mapped.output_item is not None
    assert mapped.output_item.final is True
    assert mapped.output_item.item.content == "Final answer."
    # Context item also stores the stripped content
    assert mapped.context_item.content == "Final answer."


def test_subagent_assistant_final_sentinel_ignored() -> None:
    mapper = OpenAiEventMapper()
    raw = SimpleNamespace(
        type="message_output_item",
        message=SimpleNamespace(
            id="msg_3",
            role="assistant",
            content=[SimpleNamespace(type="output_text", text="sub done <final/>")],
            tool_calls=None,
        ),
    )
    execution = AgentExecution(
        agent_id="sub", agent_name="sub", depth=1,
        parent_agent_id="root", parent_tool_call_id="c1",
    )
    mapped = mapper.to_mapped_event(raw, execution=execution, is_root=False)
    assert mapped.output_item is not None
    assert mapped.output_item.final is False
    # Sentinel not stripped for non-root agents
    assert "sub done" in (mapped.output_item.item.content or "")


def test_tool_call_output_item() -> None:
    mapper = OpenAiEventMapper()
    raw = SimpleNamespace(
        type="tool_call_item",
        tool_call=SimpleNamespace(
            id="call_1",
            type="function",
            function=SimpleNamespace(name="query_traces", arguments="{}"),
        ),
        message_id="msg_4",
    )
    mapped = mapper.to_mapped_event(raw, execution=_exec(), is_root=True)
    assert mapped.context_item is not None
    assert mapped.context_item.role == "assistant"
    assert mapped.context_item.tool_calls is not None
    assert mapped.context_item.tool_calls[0].function.name == "query_traces"


def test_tool_output_item() -> None:
    mapper = OpenAiEventMapper()
    raw = SimpleNamespace(
        type="tool_call_output_item",
        tool_call_id="call_1",
        name="query_traces",
        output="ok",
        message_id="msg_5",
    )
    mapped = mapper.to_mapped_event(raw, execution=_exec(), is_root=True)
    assert mapped.context_item is not None
    assert mapped.context_item.role == "tool"
    assert mapped.context_item.tool_call_id == "call_1"


def test_raw_text_delta_produces_delta_only() -> None:
    mapper = OpenAiEventMapper()
    raw = SimpleNamespace(
        type="raw_response_event",
        data=SimpleNamespace(
            type="response.output_text.delta",
            delta="par",
            item_id="msg_1",
        ),
    )
    mapped = mapper.to_mapped_event(raw, execution=_exec(), is_root=True)
    assert mapped.context_item is None
    assert mapped.output_item is None
    assert mapped.delta is not None
    assert mapped.delta.text_delta == "par"
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

`engine/engine/agents/openai_event_mapper.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.agents.agent_context_items import AgentContextItem
from engine.agents.agent_execution import AgentExecution
from engine.agents.prompt_templates import FINAL_SENTINEL
from engine.models.engine_output import AgentOutputItem, AgentTextDelta
from engine.models.messages import AgentMessage, AgentToolCall, AgentToolFunction


@dataclass
class MappedEvent:
    context_item: AgentContextItem | None = None
    output_item: AgentOutputItem | None = None
    delta: AgentTextDelta | None = None


class OpenAiEventMapper:
    def to_mapped_event(
        self,
        raw_event: Any,
        *,
        execution: AgentExecution,
        is_root: bool,
    ) -> MappedEvent:
        kind = getattr(raw_event, "type", None)

        if kind == "raw_response_event":
            return self._map_raw_delta(raw_event, execution=execution)

        if kind == "message_output_item":
            return self._map_assistant_message(raw_event, execution=execution, is_root=is_root)

        if kind == "tool_call_item":
            return self._map_tool_call(raw_event, execution=execution)

        if kind == "tool_call_output_item":
            return self._map_tool_output(raw_event, execution=execution)

        return MappedEvent()

    def _map_raw_delta(self, raw: Any, *, execution: AgentExecution) -> MappedEvent:
        data = getattr(raw, "data", None)
        if data is None:
            return MappedEvent()
        if getattr(data, "type", None) != "response.output_text.delta":
            return MappedEvent()
        delta = AgentTextDelta(
            sequence=0,
            agent_id=execution.agent_id,
            parent_agent_id=execution.parent_agent_id,
            parent_tool_call_id=execution.parent_tool_call_id,
            depth=execution.depth,
            item_id=str(getattr(data, "item_id", "")),
            text_delta=str(getattr(data, "delta", "")),
        )
        return MappedEvent(delta=delta)

    def _map_assistant_message(
        self,
        raw: Any,
        *,
        execution: AgentExecution,
        is_root: bool,
    ) -> MappedEvent:
        msg = raw.message
        text_parts = [getattr(p, "text", "") for p in (getattr(msg, "content", None) or []) if getattr(p, "type", None) == "output_text"]
        text = "".join(text_parts)

        final = False
        if is_root and FINAL_SENTINEL in text:
            final = True
            text = text.replace(FINAL_SENTINEL, "").rstrip()

        item_id = str(getattr(msg, "id", ""))
        context_item = AgentContextItem(
            item_id=item_id,
            role="assistant",
            content=text or None,
            agent_id=execution.agent_id,
            parent_agent_id=execution.parent_agent_id,
            parent_tool_call_id=execution.parent_tool_call_id,
        )
        output_item = AgentOutputItem(
            sequence=0,
            agent_id=execution.agent_id,
            parent_agent_id=execution.parent_agent_id,
            parent_tool_call_id=execution.parent_tool_call_id,
            agent_name=execution.agent_name,
            depth=execution.depth,
            item=AgentMessage(role="assistant", content=text or None),
            final=final,
        )
        return MappedEvent(context_item=context_item, output_item=output_item)

    def _map_tool_call(self, raw: Any, *, execution: AgentExecution) -> MappedEvent:
        call = raw.tool_call
        tc = AgentToolCall(
            id=str(call.id),
            function=AgentToolFunction(
                name=str(call.function.name),
                arguments=str(call.function.arguments),
            ),
        )
        item_id = str(getattr(raw, "message_id", call.id))
        context_item = AgentContextItem(
            item_id=item_id,
            role="assistant",
            content=None,
            tool_calls=[tc],
            agent_id=execution.agent_id,
            parent_agent_id=execution.parent_agent_id,
            parent_tool_call_id=execution.parent_tool_call_id,
        )
        output_item = AgentOutputItem(
            sequence=0,
            agent_id=execution.agent_id,
            parent_agent_id=execution.parent_agent_id,
            parent_tool_call_id=execution.parent_tool_call_id,
            agent_name=execution.agent_name,
            depth=execution.depth,
            item=AgentMessage(role="assistant", content=None, tool_calls=[tc]),
        )
        return MappedEvent(context_item=context_item, output_item=output_item)

    def _map_tool_output(self, raw: Any, *, execution: AgentExecution) -> MappedEvent:
        item_id = str(getattr(raw, "message_id", raw.tool_call_id))
        content = str(raw.output)
        context_item = AgentContextItem(
            item_id=item_id,
            role="tool",
            content=content,
            tool_call_id=str(raw.tool_call_id),
            name=str(getattr(raw, "name", "")) or None,
            agent_id=execution.agent_id,
            parent_agent_id=execution.parent_agent_id,
            parent_tool_call_id=execution.parent_tool_call_id,
        )
        output_item = AgentOutputItem(
            sequence=0,
            agent_id=execution.agent_id,
            parent_agent_id=execution.parent_agent_id,
            parent_tool_call_id=execution.parent_tool_call_id,
            agent_name=execution.agent_name,
            depth=execution.depth,
            item=AgentMessage(
                role="tool",
                content=content,
                tool_call_id=str(raw.tool_call_id),
                name=str(getattr(raw, "name", "")) or None,
            ),
        )
        return MappedEvent(context_item=context_item, output_item=output_item)
```

> Implementation note for the executor: the exact SDK attribute names (`message_output_item`, `tool_call_item`, `tool_call_output_item`, `raw_response_event`) follow `openai-agents`' `RunItemStreamEvent` and `RawResponsesStreamEvent` shapes. If the installed SDK version exposes different field names (e.g. `raw_item` vs `tool_call`), update `_map_*` branches and the corresponding test `SimpleNamespace` shapes together. Do not swallow unknown event kinds — the default `MappedEvent()` return is the intended no-op path.

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): OpenAiEventMapper with final-sentinel stripping + deltas"
```

---

## Phase 7 — Root agent runner and public entrypoints

### Task 7.1: `OpenAiAgentRunner.run` — drive `Runner.run_streamed`, push mapped events, enforce turn + failure limits

**Context:** `OpenAiAgentRunner` takes an assembled SDK `Agent`, the `AgentContext` to seed it from, and the `AgentExecution` record. It loops over `Runner.run_streamed(...).stream_events()`, maps each event, appends context items, and emits output/delta events. After each turn it invokes `AgentContext.compact_old_items`. Ten consecutive LLM failures fail the stream with `EngineAgentExhaustedError`.

**Files:**
- Create: `engine/engine/agents/openai_agent_runner.py`
- Create: `engine/tests/unit/agents/test_openai_agent_runner.py`

- [ ] **Step 1: Write failing test using a fake `Runner.run_streamed` shim**

For this TDD step, we inject a fake `run_streamed` callable so the test doesn't need a real LLM or SDK execution. We'll test that:
- Output items are emitted to the bus with sequence numbers assigned by the bus.
- Context items accumulate on the agent's context.
- The compactor runs after each turn (we assert its call count).
- Circuit breaker trips at 10 consecutive failures.

```python
from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.agents.agent_context import AgentContext
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.openai_agent_runner import OpenAiAgentRunner
from engine.errors import EngineAgentExhaustedError
from engine.model_config import ModelConfig


def _assistant_event(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="message_output_item",
        message=SimpleNamespace(
            id="m1",
            role="assistant",
            content=[SimpleNamespace(type="output_text", text=text)],
            tool_calls=None,
        ),
    )


class _FakeStream:
    def __init__(self, events: list) -> None:
        self._events = events

    async def stream_events(self):
        for e in self._events:
            yield e


def _context() -> AgentContext:
    return AgentContext(
        items=[],
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        text_message_compaction_keep_last_messages=2,
        tool_call_compaction_keep_last_messages=2,
    )


@pytest.mark.asyncio
async def test_runner_emits_final_output_and_updates_context() -> None:
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root", agent_name="root", depth=0,
        parent_agent_id=None, parent_tool_call_id=None,
    )

    async def fake_run_streamed(*, agent, input, context):
        return _FakeStream([_assistant_event("answer\n<final/>")])

    compact_calls: list[int] = []

    async def fake_compactor(item):
        compact_calls.append(1)
        return "sum"

    runner = OpenAiAgentRunner(
        run_streamed=fake_run_streamed,
        compactor_factory=lambda _: fake_compactor,
    )

    await runner.run(
        sdk_agent=object(),
        agent_context=ctx,
        agent_execution=execution,
        output_bus=bus,
        is_root=True,
    )

    await bus.close()
    events = [e async for e in bus.stream()]
    assert any(getattr(e, "final", False) for e in events)
    assert any(item.role == "assistant" for item in ctx.items)


@pytest.mark.asyncio
async def test_runner_circuit_breaker() -> None:
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root", agent_name="root", depth=0,
        parent_agent_id=None, parent_tool_call_id=None,
    )

    async def always_fail(*, agent, input, context):
        raise RuntimeError("provider 500")

    async def noop_compactor(_):
        return ""

    runner = OpenAiAgentRunner(
        run_streamed=always_fail,
        compactor_factory=lambda _: noop_compactor,
    )

    with pytest.raises(EngineAgentExhaustedError):
        await runner.run(
            sdk_agent=object(),
            agent_context=ctx,
            agent_execution=execution,
            output_bus=bus,
            is_root=True,
        )
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

`engine/engine/agents/openai_agent_runner.py`:

```python
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from engine.agents.agent_context import AgentContext, Compactor
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.openai_event_mapper import OpenAiEventMapper
from engine.errors import EngineAgentExhaustedError

MAX_CONSECUTIVE_LLM_FAILURES = 10

RunStreamedCallable = Callable[..., Awaitable[Any]]
CompactorFactory = Callable[[AgentExecution], Compactor]


class OpenAiAgentRunner:
    def __init__(
        self,
        run_streamed: RunStreamedCallable,
        compactor_factory: CompactorFactory,
        event_mapper: OpenAiEventMapper | None = None,
    ) -> None:
        self._run_streamed = run_streamed
        self._compactor_factory = compactor_factory
        self._mapper = event_mapper or OpenAiEventMapper()

    async def run(
        self,
        *,
        sdk_agent: Any,
        agent_context: AgentContext,
        agent_execution: AgentExecution,
        output_bus: EngineOutputBus,
        is_root: bool,
        run_context: Any | None = None,
    ) -> None:
        messages = [m.model_dump() for m in agent_context.to_messages_array()]
        try:
            stream = await self._run_streamed(agent=sdk_agent, input=messages, context=run_context)
        except Exception as exc:
            agent_execution.record_llm_failure()
            logger.warning(
                "llm call failed for agent_id={} (failure {} of {})",
                agent_execution.agent_id,
                agent_execution.consecutive_llm_failures,
                MAX_CONSECUTIVE_LLM_FAILURES,
            )
            if agent_execution.consecutive_llm_failures >= MAX_CONSECUTIVE_LLM_FAILURES:
                raise EngineAgentExhaustedError(
                    f"agent {agent_execution.agent_id} exhausted "
                    f"after {MAX_CONSECUTIVE_LLM_FAILURES} consecutive failures"
                ) from exc
            return

        agent_execution.record_llm_success()

        async for raw_event in stream.stream_events():
            mapped = self._mapper.to_mapped_event(
                raw_event, execution=agent_execution, is_root=is_root
            )
            if mapped.context_item is not None:
                agent_context.append(mapped.context_item)
            if mapped.output_item is not None:
                emitted = await output_bus.emit(mapped.output_item)
                if agent_execution.output_start_sequence is None:
                    agent_execution.output_start_sequence = emitted.sequence
                agent_execution.output_end_sequence = emitted.sequence
            if mapped.delta is not None:
                await output_bus.emit(mapped.delta)

        agent_execution.turns_used += 1
        await agent_context.compact_old_items(self._compactor_factory(agent_execution))
```

Note: for retries on transient errors beyond the first, `Runner.run_streamed` is called again by the caller in the subagent-tool path or by an outer retry loop. The in-method logic only increments the failure counter and gives up at 10. The caller (main or subagent runner) decides whether to retry.

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): OpenAiAgentRunner driving streamed run"
```

---

### Task 7.2: Public entrypoints — `stream_engine_async`, `run_engine_async`, sync wrappers

**Context:** `main.py` is the public API. It:
1. Calls `TraceIndexBuilder.ensure_index_exists`, then `TraceStore.load`.
2. Builds `EngineRunState`, `EngineOutputBus`.
3. Seeds the root `AgentContext` from the input `list[AgentMessage]`. The root system prompt is prepended as a synthetic `system` item.
4. Calls a `build_root_agent(...)` helper (implemented in the next task) that returns an SDK `Agent` wired with all tools.
5. Starts `OpenAiAgentRunner.run(...)` as a background task.
6. Yields from `output_bus.stream()`.
7. On root task exception, calls `output_bus.fail(err)` and re-raises to the stream consumer (already handled by `EngineOutputBus.fail`).

**Files:**
- Create: `engine/engine/main.py`
- Create: `engine/tests/unit/test_main.py` (only tests public signature + import; end-to-end behavior tested in Phase 10)

- [ ] **Step 1: Write signature-level test**

```python
from __future__ import annotations

import inspect
import pytest

import engine.main as main


def test_public_entrypoints_exist_and_are_async() -> None:
    assert inspect.iscoroutinefunction(main.stream_engine_async)
    assert inspect.iscoroutinefunction(main.run_engine_async)
    assert callable(main.stream_engine)
    assert callable(main.run_engine)


def test_async_signatures_match() -> None:
    for fn in (main.stream_engine_async, main.run_engine_async):
        params = list(inspect.signature(fn).parameters)
        assert params[:3] == ["messages", "engine_config", "trace_path"]
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

`engine/engine/main.py`:

```python
from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from engine.agents.agent_context import AgentContext
from engine.agents.agent_context_items import AgentContextItem
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.engine_run_state import EngineRunState
from engine.agents.openai_agent_runner import OpenAiAgentRunner
from engine.agents.prompt_templates import render_root_system_prompt
from engine.engine_config import EngineConfig
from engine.models.engine_output import AgentOutputItem, EngineStreamEvent
from engine.models.messages import AgentMessage
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore


async def stream_engine_async(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
) -> AsyncIterator[EngineStreamEvent]:
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path,
        config=engine_config.trace_index,
    )
    trace_store = TraceStore.load(trace_path=trace_path, index_path=index_path)

    output_bus = EngineOutputBus()
    run_state = EngineRunState(
        trace_store=trace_store,
        output_bus=output_bus,
        config=engine_config,
    )

    root_execution = AgentExecution(
        agent_id=f"root-{uuid.uuid4().hex[:8]}",
        agent_name=engine_config.root_agent.name,
        depth=0,
        parent_agent_id=None,
        parent_tool_call_id=None,
    )
    run_state.register(root_execution)

    root_context = _build_root_context(messages=messages, engine_config=engine_config)

    # Late import breaks cycles (subagent factory depends on run_state).
    from engine.tools.subagent_tool_factory import build_root_sdk_agent

    sdk_agent = build_root_sdk_agent(
        engine_config=engine_config,
        run_state=run_state,
        agent_execution=root_execution,
    )

    async def _drive() -> None:
        from agents import Runner

        runner = OpenAiAgentRunner(
            run_streamed=Runner.run_streamed,
            compactor_factory=_build_compactor_factory(engine_config),
        )
        await runner.run(
            sdk_agent=sdk_agent,
            agent_context=root_context,
            agent_execution=root_execution,
            output_bus=output_bus,
            is_root=True,
            run_context=run_state,
        )
        await output_bus.close()

    task = asyncio.create_task(_drive())

    try:
        async for event in output_bus.stream():
            yield event
        # stream() only exits cleanly when close() was called; surface task errors otherwise.
        await task
    except BaseException:
        task.cancel()
        raise


async def run_engine_async(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
) -> list[AgentOutputItem]:
    out: list[AgentOutputItem] = []
    async for event in stream_engine_async(messages, engine_config, trace_path):
        if isinstance(event, AgentOutputItem):
            out.append(event)
    return out


def stream_engine(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
) -> list[EngineStreamEvent]:
    async def _collect() -> list[EngineStreamEvent]:
        out: list[EngineStreamEvent] = []
        async for ev in stream_engine_async(messages, engine_config, trace_path):
            out.append(ev)
        return out
    return asyncio.run(_collect())


def run_engine(
    messages: list[AgentMessage],
    engine_config: EngineConfig,
    trace_path: Path,
) -> list[AgentOutputItem]:
    return asyncio.run(run_engine_async(messages, engine_config, trace_path))


def _build_root_context(
    *,
    messages: list[AgentMessage],
    engine_config: EngineConfig,
) -> AgentContext:
    # Derive user instructions from the first user message's text if present.
    first_user = next((m for m in messages if m.role == "user" and isinstance(m.content, str)), None)
    user_instructions = first_user.content if first_user else engine_config.root_agent.instructions

    system_prompt = render_root_system_prompt(
        user_instructions=user_instructions,
        maximum_depth=engine_config.maximum_depth,
        maximum_parallel_subagents=engine_config.maximum_parallel_subagents,
    )
    items: list[AgentContextItem] = [
        AgentContextItem(item_id="sys-0", role="system", content=system_prompt)
    ]
    for i, msg in enumerate(messages):
        items.append(AgentContextItem(
            item_id=f"in-{i}",
            role=msg.role,
            content=msg.content,
            tool_calls=msg.tool_calls,
            tool_call_id=msg.tool_call_id,
            name=msg.name,
        ))

    return AgentContext(
        items=items,
        compaction_model=engine_config.compaction_model,
        text_message_compaction_keep_last_messages=engine_config.text_message_compaction_keep_last_messages,
        tool_call_compaction_keep_last_messages=engine_config.tool_call_compaction_keep_last_messages,
    )


def _build_compactor_factory(engine_config: EngineConfig):
    from engine.agents.agent_context import Compactor
    from engine.agents.agent_context_items import AgentContextItem
    from engine.agents.prompt_templates import COMPACTION_SYSTEM_PROMPT
    from openai import AsyncOpenAI

    client = AsyncOpenAI()

    def factory(_execution) -> Compactor:
        async def compact(item: AgentContextItem) -> str:
            user_text = _item_as_prompt(item)
            response = await client.chat.completions.create(
                model=engine_config.compaction_model.name,
                messages=[
                    {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                temperature=engine_config.compaction_model.temperature or 0.0,
            )
            return (response.choices[0].message.content or "").strip()
        return compact
    return factory


def _item_as_prompt(item: AgentContextItem) -> str:
    if item.role == "user":
        return f"USER MESSAGE:\n{item.content}"
    if item.role == "assistant":
        if item.tool_calls:
            calls = "\n".join(
                f"- {tc.function.name}({tc.function.arguments})"
                for tc in item.tool_calls
            )
            return f"ASSISTANT TOOL CALLS:\n{calls}"
        return f"ASSISTANT MESSAGE:\n{item.content}"
    if item.role == "tool":
        return f"TOOL RESULT (tool={item.name}, call={item.tool_call_id}):\n{item.content}"
    return str(item.content or "")
```

> The `openai-agents` `Runner` attribute name, the `Runner.run_streamed` signature, and the bound first argument are specified by that SDK. If the installed version names this differently (e.g. a class method vs bound method), update only the `from agents import Runner` line + the call site. Do not refactor `OpenAiAgentRunner` — its `run_streamed` callable is intentionally injectable.

- [ ] **Step 4: Run the signature tests — expect pass**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/test_main.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): public entrypoints — stream/run engine (sync + async)"
```

---

## Phase 8 — Subagents and synthesis

### Task 8.1: Synthesis tool

**Files:**
- Create: `engine/engine/tools/synthesis_tool.py`
- Create: `engine/tests/unit/tools/test_synthesis_tool.py`

- [ ] **Step 1: Test (mock the OpenAI client)**

```python
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engine.tools.synthesis_tool import SynthesisTool, SynthesizeTracesArguments
from engine.tools.tool_protocol import ToolContext


@pytest.mark.asyncio
async def test_synthesis_tool_calls_client_and_returns_summary(monkeypatch) -> None:
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(
            return_value=SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="summary")
            )])
        )))
    )
    tool = SynthesisTool(model_name="claude-haiku-4-5", client=fake_client)
    ctx = ToolContext()

    result = await tool.run(ctx, SynthesizeTracesArguments(trace_ids=["t1", "t2"], focus="errors"))
    assert result.summary == "summary"
    fake_client.chat.completions.create.assert_awaited_once()
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

`engine/engine/tools/synthesis_tool.py`:

```python
from __future__ import annotations

from typing import Any

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

    def __init__(self, model_name: str, client: Any | None = None) -> None:
        self._model_name = model_name
        if client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI()
        else:
            self._client = client

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
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): synthesis tool"
```

---

### Task 8.2: Subagent tool factory — `build_root_sdk_agent` + `build_subagent_as_tool`

**Context:** This task wires subagents. Core responsibilities:
- Build the root SDK `Agent` with all tools. If `maximum_depth > 0`, include a `call_subagent` tool built from `build_subagent_as_tool(depth=1)`.
- `build_subagent_as_tool(depth)`:
  - Constructs the child SDK `Agent` using `engine_config.subagent` + `render_subagent_system_prompt(...)`.
  - If `depth + 1 <= maximum_depth`, the child's tool list includes `build_subagent_as_tool(depth+1)`. Otherwise the child gets only leaf tools (trace tools, context tool, synthesis tool, run_code). **The child at `depth == maximum_depth` has no subagent tool — this is the structural hard stop.**
  - Wraps the child with `child_agent.as_tool(tool_name="call_subagent", tool_description=..., on_stream=..., custom_output_extractor=..., max_turns=subagent.maximum_turns)`.
  - `on_stream` forwards child events to the shared `EngineOutputBus` using `OpenAiEventMapper(is_root=False)`.
  - `custom_output_extractor` walks the completed `RunResultStreaming` and returns a serialized `SubagentToolResult`.
- Enforce `maximum_parallel_subagents` via `asyncio.Semaphore`: wrap the SDK tool's callable with a `semaphore.acquire()`/`release()`. The SDK's `on_stream` + `custom_output_extractor` both run inside a call already orchestrated by the SDK tool framework; the semaphore should gate the *invocation* of that tool. This is done by wrapping the returned `FunctionTool` with a pre/post-hook via `RunHooks` or by intercepting the adapter. For simplicity in v1, wrap the `on_invoke_tool` callable the SDK produces for `as_tool`.

Implementation approach: we get the `FunctionTool` from `Agent.as_tool(...)`, keep its original `on_invoke_tool`, and replace it with a wrapper that does `async with semaphore:` around the call. That guarantees bounded concurrency even if the model calls `call_subagent` in parallel.

**Files:**
- Create: `engine/engine/tools/subagent_tool_factory.py`
- Create: `engine/tests/unit/tools/test_subagent_tool_factory.py`

- [ ] **Step 1: Write failing tests targeting the structural depth limit + semaphore wrapper**

```python
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from engine.agents.agent_config import AgentConfig
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.engine_run_state import EngineRunState
from engine.engine_config import EngineConfig
from engine.model_config import ModelConfig
from engine.tools.subagent_tool_factory import _child_tools_for_depth


def _engine_config(max_depth: int) -> EngineConfig:
    agent = AgentConfig(
        name="a", instructions="",
        model=ModelConfig(name="claude-sonnet-4-5"), maximum_turns=10,
    )
    return EngineConfig(
        root_agent=agent, subagent=agent,
        synthesis_model=ModelConfig(name="claude-haiku-4-5"),
        compaction_model=ModelConfig(name="claude-haiku-4-5"),
        maximum_depth=max_depth,
    )


def test_child_tools_at_max_depth_omits_subagent_tool() -> None:
    cfg = _engine_config(max_depth=2)
    run_state = MagicMock(spec=EngineRunState)
    run_state.config = cfg
    run_state.output_bus = EngineOutputBus()
    # Depth 2 == maximum_depth: do NOT include subagent tool
    tools = _child_tools_for_depth(depth=2, run_state=run_state, semaphore=None)
    names = {t.name for t in tools}
    assert "call_subagent" not in names


def test_child_tools_below_max_depth_includes_subagent_tool() -> None:
    cfg = _engine_config(max_depth=2)
    run_state = MagicMock(spec=EngineRunState)
    run_state.config = cfg
    run_state.output_bus = EngineOutputBus()
    import asyncio
    sem = asyncio.Semaphore(4)
    tools = _child_tools_for_depth(depth=1, run_state=run_state, semaphore=sem)
    names = {t.name for t in tools}
    assert "call_subagent" in names


@pytest.mark.asyncio
async def test_semaphore_wrapper_limits_parallelism() -> None:
    from engine.tools.subagent_tool_factory import _wrap_with_semaphore
    import asyncio

    in_flight = 0
    peak = 0

    async def fake_tool(ctx, args):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return "ok"

    sem = asyncio.Semaphore(2)
    wrapped = _wrap_with_semaphore(fake_tool, sem)

    await asyncio.gather(*[wrapped(None, "{}") for _ in range(6)])
    assert peak <= 2
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

`engine/engine/tools/subagent_tool_factory.py`:

```python
from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from agents import Agent, FunctionTool, RunContextWrapper, Runner

from engine.agents.agent_context import AgentContext
from engine.agents.agent_context_items import AgentContextItem
from engine.agents.agent_execution import AgentExecution
from engine.agents.engine_output_bus import EngineOutputBus
from engine.agents.engine_run_state import EngineRunState
from engine.agents.openai_event_mapper import OpenAiEventMapper
from engine.agents.prompt_templates import render_subagent_system_prompt
from engine.engine_config import EngineConfig
from engine.errors import EngineMaxDepthExceededError
from engine.tools.agent_context_tools import GetContextItemTool
from engine.tools.run_code_tool import RunCodeTool
from engine.tools.synthesis_tool import SynthesisTool
from engine.tools.tool_protocol import ToolContext, to_sdk_function_tool
from engine.tools.trace_tools import (
    CountTracesTool,
    GetDatasetOverviewTool,
    QueryTracesTool,
    SearchTraceTool,
    ViewTraceTool,
)


def build_root_sdk_agent(
    *,
    engine_config: EngineConfig,
    run_state: EngineRunState,
    agent_execution: AgentExecution,
) -> Agent[EngineRunState]:
    semaphore = asyncio.Semaphore(engine_config.maximum_parallel_subagents)
    tools = _child_tools_for_depth(depth=0, run_state=run_state, semaphore=semaphore)

    return Agent[EngineRunState](
        name=engine_config.root_agent.name,
        instructions="",  # system prompt lives in the AgentContext
        model=engine_config.root_agent.model.name,
        tools=tools,
    )


def _child_tools_for_depth(
    *,
    depth: int,
    run_state: EngineRunState,
    semaphore: asyncio.Semaphore | None,
) -> list[FunctionTool]:
    engine_config = run_state.config

    def make_ctx(wrapper: RunContextWrapper[Any]) -> ToolContext:
        return ToolContext(
            run_state=run_state,
            trace_store=run_state.trace_store,
            output_bus=run_state.output_bus,
        )

    leaf_tools: list[FunctionTool] = [
        to_sdk_function_tool(GetDatasetOverviewTool(), context_factory=make_ctx),
        to_sdk_function_tool(QueryTracesTool(), context_factory=make_ctx),
        to_sdk_function_tool(CountTracesTool(), context_factory=make_ctx),
        to_sdk_function_tool(ViewTraceTool(), context_factory=make_ctx),
        to_sdk_function_tool(SearchTraceTool(), context_factory=make_ctx),
        to_sdk_function_tool(GetContextItemTool(), context_factory=make_ctx),
        to_sdk_function_tool(
            SynthesisTool(model_name=engine_config.synthesis_model.name),
            context_factory=make_ctx,
        ),
        to_sdk_function_tool(RunCodeTool(sandbox_config=engine_config.sandbox), context_factory=make_ctx),
    ]

    if depth >= engine_config.maximum_depth:
        return leaf_tools

    assert semaphore is not None, "semaphore required when sub-depth subagent tool is built"
    subagent_tool = _build_subagent_as_tool(
        run_state=run_state,
        child_depth=depth + 1,
        semaphore=semaphore,
    )
    return leaf_tools + [subagent_tool]


def _build_subagent_as_tool(
    *,
    run_state: EngineRunState,
    child_depth: int,
    semaphore: asyncio.Semaphore,
) -> FunctionTool:
    engine_config = run_state.config
    child_agent = Agent[EngineRunState](
        name=engine_config.subagent.name,
        instructions=render_subagent_system_prompt(
            user_instructions=engine_config.subagent.instructions,
            depth=child_depth,
            maximum_depth=engine_config.maximum_depth,
            maximum_parallel_subagents=engine_config.maximum_parallel_subagents,
        ),
        model=engine_config.subagent.model.name,
        tools=_child_tools_for_depth(depth=child_depth, run_state=run_state, semaphore=semaphore),
    )

    mapper = OpenAiEventMapper()
    output_bus = run_state.output_bus

    def on_stream_factory(execution: AgentExecution):
        async def on_stream(event) -> None:
            mapped = mapper.to_mapped_event(event, execution=execution, is_root=False)
            if mapped.output_item is not None:
                await output_bus.emit(mapped.output_item)
            if mapped.delta is not None:
                await output_bus.emit(mapped.delta)
        return on_stream

    async def custom_output_extractor(run_result) -> str:
        from engine.tools.subagent_result import SubagentToolResult  # created below

        final_text = ""
        for item in getattr(run_result, "new_items", []):
            if getattr(item, "type", None) == "message_output_item":
                parts = [
                    getattr(p, "text", "")
                    for p in (getattr(item.message, "content", None) or [])
                    if getattr(p, "type", None) == "output_text"
                ]
                text = "".join(parts).strip()
                if text:
                    final_text = text
        return SubagentToolResult(
            child_agent_id="",  # patched in the wrapper below
            answer=final_text,
            output_start_sequence=0,
            output_end_sequence=0,
            turns_used=0,
            tool_calls_made=0,
        ).model_dump_json()

    sdk_tool = child_agent.as_tool(
        tool_name="call_subagent",
        tool_description="Delegate a focused question to a subagent. Returns the subagent's answer.",
        custom_output_extractor=custom_output_extractor,
        on_stream=None,  # set per-invocation in the wrapper below
        max_turns=engine_config.subagent.maximum_turns,
    )

    original_invoke: Callable[[RunContextWrapper[Any], str], Awaitable[str]] = sdk_tool.on_invoke_tool

    async def guarded_invoke(ctx: RunContextWrapper[Any], raw_arguments: str) -> str:
        # Depth guard (redundant vs structural check; defensive for bugs)
        if child_depth > engine_config.maximum_depth:
            raise EngineMaxDepthExceededError(
                f"subagent invoked at depth={child_depth} > maximum_depth={engine_config.maximum_depth}"
            )

        async with semaphore:
            child_execution = AgentExecution(
                agent_id=f"sub-{uuid.uuid4().hex[:8]}",
                agent_name=engine_config.subagent.name,
                depth=child_depth,
                parent_agent_id=None,
                parent_tool_call_id=None,
            )
            run_state.register(child_execution)
            # Install on_stream for this invocation. The SDK's as_tool wiring
            # receives on_stream via the tool's bound agent; since we created the
            # tool with on_stream=None above, we forward events by patching the
            # child Agent's hooks. Simpler: re-invoke via Runner.run_streamed and
            # forward manually, then hand the completed result to extractor.
            stream = await Runner.run_streamed(
                agent=child_agent,
                input=raw_arguments,
                context=run_state,
            )
            start_seq = None
            end_seq = None
            async for ev in stream.stream_events():
                mapped = mapper.to_mapped_event(ev, execution=child_execution, is_root=False)
                if mapped.output_item is not None:
                    emitted = await output_bus.emit(mapped.output_item)
                    if start_seq is None:
                        start_seq = emitted.sequence
                    end_seq = emitted.sequence
                if mapped.delta is not None:
                    await output_bus.emit(mapped.delta)

            child_execution.output_start_sequence = start_seq
            child_execution.output_end_sequence = end_seq
            run_result = await stream.wait_for_final_output() if hasattr(stream, "wait_for_final_output") else stream

            extracted_json = await custom_output_extractor(run_result)
            from engine.tools.subagent_result import SubagentToolResult

            result = SubagentToolResult.model_validate_json(extracted_json).model_copy(update={
                "child_agent_id": child_execution.agent_id,
                "output_start_sequence": start_seq or 0,
                "output_end_sequence": end_seq or 0,
                "turns_used": child_execution.turns_used,
                "tool_calls_made": child_execution.tool_calls_made,
            })
            return result.model_dump_json()

    sdk_tool.on_invoke_tool = guarded_invoke
    return sdk_tool


def _wrap_with_semaphore(
    fn: Callable[..., Awaitable[Any]],
    semaphore: asyncio.Semaphore,
) -> Callable[..., Awaitable[Any]]:
    async def wrapped(*args, **kwargs):
        async with semaphore:
            return await fn(*args, **kwargs)
    return wrapped
```

> **Implementation note for the executor:** the exact way `Agent.as_tool` wires `on_stream` vs running the child manually may need to be adjusted against the installed `openai-agents` version. The pattern used here — take whatever `as_tool` returns, replace `on_invoke_tool` with a `guarded_invoke` that drives `Runner.run_streamed` ourselves and pumps events to `EngineOutputBus` — gives us full control of streaming, sequencing, and lineage regardless of SDK quirks. If the SDK's own `on_stream` callback honors async delivery and respects lineage, the manual drive loop can be simplified to rely on it. Keep the depth structural check and semaphore unchanged.

- [ ] **Step 4: Write `engine/engine/tools/subagent_result.py`**

```python
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SubagentToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    child_agent_id: str
    answer: str
    output_start_sequence: int
    output_end_sequence: int
    turns_used: int
    tool_calls_made: int
```

- [ ] **Step 5: Run — expect pass**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/tools/test_subagent_tool_factory.py -v`
Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git -C /home/declan/dev/HALO add engine/engine/tools/subagent_tool_factory.py engine/engine/tools/subagent_result.py engine/tests/unit/tools/test_subagent_tool_factory.py
git -C /home/declan/dev/HALO commit -m "feat(engine): subagent tool factory with structural depth limit + semaphore"
```

---

## Phase 9 — Sandbox + run_code tool

### Task 9.1: Sandbox venv build script

**Files:**
- Create: `engine/scripts/build_sandbox_venv.sh`

- [ ] **Step 1: Write script**

```bash
#!/usr/bin/env zsh
# Builds a standalone python 3.12 venv at engine/.sandbox-venv containing
# halo-engine + numpy + pandas. Idempotent: re-runs are a no-op if present.

set -euo pipefail

here="${0:a:h}"
root="${here}/.."
venv="${root}/.sandbox-venv"

if [[ -d "${venv}/bin" ]]; then
  echo "sandbox venv already exists at ${venv}"
  exit 0
fi

uv venv --python 3.12 "${venv}"
"${venv}/bin/pip" install --no-cache-dir \
  "numpy>=2.0" \
  "pandas>=2.2" \
  "pydantic>=2.8"
"${venv}/bin/pip" install --no-cache-dir -e "${root}"

echo "sandbox venv built at ${venv}"
```

- [ ] **Step 2: Mark executable**

Run: `chmod +x /home/declan/dev/HALO/engine/scripts/build_sandbox_venv.sh`

- [ ] **Step 3: Add to `.gitignore` so the venv itself isn't checked in**

Edit `engine/.gitignore` (create if missing):

```text
.venv/
.sandbox-venv/
__pycache__/
.pytest_cache/
*.egg-info/
```

- [ ] **Step 4: Commit**

```bash
git -C /home/declan/dev/HALO add engine/scripts engine/.gitignore
git -C /home/declan/dev/HALO commit -m "build(engine): sandbox venv build script + gitignore"
```

---

### Task 9.2: `sandbox_policy.py` — compose `SandboxPolicy` from run inputs

**Files:**
- Create: `engine/engine/sandbox/sandbox_policy.py`
- Create: `engine/tests/unit/sandbox/test_sandbox_policy.py`

- [ ] **Step 1: Test**

```python
from __future__ import annotations

from pathlib import Path

from engine.sandbox.sandbox_config import SandboxConfig
from engine.sandbox.sandbox_policy import compose_policy


def test_compose_policy_sets_paths(tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    index = tmp_path / "t.idx.jsonl"
    venv = tmp_path / ".sandbox-venv"
    work = tmp_path / "work"

    for p in (trace, index):
        p.write_text("")
    venv.mkdir()
    work.mkdir()

    policy = compose_policy(
        trace_path=trace,
        index_path=index,
        sandbox_venv=venv,
        work_dir=work,
        sandbox_config=SandboxConfig(timeout_seconds=7.0),
    )
    assert trace in policy.readonly_paths
    assert index in policy.readonly_paths
    assert venv in policy.readonly_paths
    assert work in policy.writable_paths
    assert policy.timeout_seconds == 7.0
    assert policy.network_enabled is False
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

`engine/engine/sandbox/sandbox_policy.py`:

```python
from __future__ import annotations

from pathlib import Path

from engine.sandbox.sandbox_config import SandboxConfig, SandboxPolicy


def compose_policy(
    *,
    trace_path: Path,
    index_path: Path,
    sandbox_venv: Path,
    work_dir: Path,
    sandbox_config: SandboxConfig,
) -> SandboxPolicy:
    return SandboxPolicy(
        readonly_paths=[trace_path, index_path, sandbox_venv],
        writable_paths=[work_dir],
        timeout_seconds=sandbox_config.timeout_seconds,
    )
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): sandbox policy composer"
```

---

### Task 9.3: `platform_commands.py` — Linux bubblewrap + macOS sandbox-exec command builders

**Files:**
- Create: `engine/engine/sandbox/platform_commands.py`
- Create: `engine/tests/unit/sandbox/test_platform_commands.py`

- [ ] **Step 1: Tests**

```python
from __future__ import annotations

from pathlib import Path

from engine.sandbox.platform_commands import (
    build_linux_bubblewrap_command,
    build_macos_sandbox_exec_command,
    render_macos_profile,
)
from engine.sandbox.sandbox_config import SandboxPolicy


def _policy(tmp_path: Path) -> SandboxPolicy:
    trace = tmp_path / "t.jsonl"
    index = tmp_path / "t.idx.jsonl"
    venv = tmp_path / "venv"
    work = tmp_path / "work"
    for p in (trace, index):
        p.write_text("")
    for d in (venv, work):
        d.mkdir()
    return SandboxPolicy(
        readonly_paths=[trace, index, venv],
        writable_paths=[work],
        timeout_seconds=10.0,
    )


def test_linux_command_contains_core_flags(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    script = tmp_path / "work" / "bootstrap.py"
    script.write_text("print(1)")
    argv = build_linux_bubblewrap_command(policy=policy, script_path=script)

    joined = " ".join(argv)
    assert argv[0] == "bwrap"
    assert "--unshare-all" in joined
    assert "--unshare-net" in joined
    assert "--clearenv" in joined
    assert "--die-with-parent" in joined
    assert "--ro-bind" in joined
    assert str(script) in joined


def test_macos_profile_denies_by_default(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    profile = render_macos_profile(policy=policy)
    assert "(deny default)" in profile
    assert "(deny network*)" in profile
    assert "(allow file-write*" in profile


def test_macos_command_shape(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    script = tmp_path / "work" / "bootstrap.py"
    script.write_text("")
    profile_path = tmp_path / "profile.sb"
    argv = build_macos_sandbox_exec_command(
        policy=policy, script_path=script, profile_path=profile_path
    )
    assert argv[0] == "sandbox-exec"
    assert "-f" in argv
    assert str(profile_path) in argv
    assert "env" in argv  # env -i chained
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

`engine/engine/sandbox/platform_commands.py`:

```python
from __future__ import annotations

from pathlib import Path

from engine.sandbox.sandbox_config import SandboxPolicy


def build_linux_bubblewrap_command(
    *,
    policy: SandboxPolicy,
    script_path: Path,
) -> list[str]:
    work_dir = policy.writable_paths[0]
    venv = policy.readonly_paths[-1]  # compose_policy appends venv last among RO
    trace_path = policy.readonly_paths[0]
    index_path = policy.readonly_paths[1]

    return [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--unshare-net",
        "--clearenv",
        "--ro-bind", str(trace_path), "/mnt/trace/traces.jsonl",
        "--ro-bind", str(index_path), "/mnt/trace/traces.jsonl.engine-index.jsonl",
        "--ro-bind", str(venv), "/venv",
        "--bind", str(work_dir), "/workspace",
        "--setenv", "PATH", "/venv/bin:/usr/bin:/bin",
        "--setenv", "HOME", "/workspace",
        "--setenv", "LANG", "C.UTF-8",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--setenv", "PYTHONUNBUFFERED", "1",
        "--setenv", "TMPDIR", "/workspace/tmp",
        "--chdir", "/workspace",
        "--proc", "/proc",
        "--dev", "/dev",
        "--",
        "/venv/bin/python",
        str(script_path),
    ]


def render_macos_profile(*, policy: SandboxPolicy) -> str:
    allows_read = "\n".join(
        f'(allow file-read* (subpath "{p}"))' if p.is_dir() else f'(allow file-read* (literal "{p}"))'
        for p in policy.readonly_paths
    )
    allows_write = "\n".join(
        f'(allow file-write* (subpath "{p}"))' for p in policy.writable_paths
    )
    return f"""(version 1)
(deny default)
(allow process*)
(deny network*)
{allows_read}
{allows_write}
"""


def build_macos_sandbox_exec_command(
    *,
    policy: SandboxPolicy,
    script_path: Path,
    profile_path: Path,
) -> list[str]:
    venv = policy.readonly_paths[-1]
    work_dir = policy.writable_paths[0]

    return [
        "sandbox-exec",
        "-f", str(profile_path),
        "env", "-i",
        f"PATH={venv}/bin:/usr/bin:/bin",
        f"HOME={work_dir}",
        "LANG=C.UTF-8",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUNBUFFERED=1",
        f"TMPDIR={work_dir}/tmp",
        f"{venv}/bin/python",
        str(script_path),
    ]
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): linux bubblewrap + macos sandbox-exec command builders"
```

---

### Task 9.4: `sandbox_bootstrap.py` — wrapper script template that exposes `trace_store` to user code

**Files:**
- Create: `engine/engine/sandbox/sandbox_bootstrap.py`
- Create: `engine/tests/unit/sandbox/test_sandbox_bootstrap.py`

- [ ] **Step 1: Test**

```python
from __future__ import annotations

from engine.sandbox.sandbox_bootstrap import render_bootstrap_script


def test_bootstrap_script_includes_user_code() -> None:
    script = render_bootstrap_script(
        user_code="print(trace_store.trace_count)",
        trace_mount_path="/mnt/trace/traces.jsonl",
        index_mount_path="/mnt/trace/traces.jsonl.engine-index.jsonl",
    )
    assert "print(trace_store.trace_count)" in script
    assert "from engine.traces.trace_store import TraceStore" in script
    assert "/mnt/trace/traces.jsonl" in script
    assert "import numpy" in script
    assert "import pandas" in script


def test_user_code_runs_in_isolated_namespace() -> None:
    script = render_bootstrap_script(
        user_code="x = 1",
        trace_mount_path="/a",
        index_mount_path="/b",
    )
    # The wrapper separates bootstrap from user code with an exec call,
    # so user code cannot accidentally shadow bootstrap variables.
    assert 'exec(_USER_CODE' in script
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

`engine/engine/sandbox/sandbox_bootstrap.py`:

```python
from __future__ import annotations


_TEMPLATE = '''\
from __future__ import annotations

import sys
from pathlib import Path

# Core analysis imports, pre-loaded so user code sees them immediately.
import numpy
import pandas

from engine.traces.trace_store import TraceStore

trace_store = TraceStore.load(
    trace_path=Path({trace_mount_path!r}),
    index_path=Path({index_mount_path!r}),
)

_USER_CODE = {user_code!r}

_globals = {{
    "trace_store": trace_store,
    "numpy": numpy,
    "pandas": pandas,
    "np": numpy,
    "pd": pandas,
    "Path": Path,
}}

exec(_USER_CODE, _globals, _globals)
'''


def render_bootstrap_script(
    *,
    user_code: str,
    trace_mount_path: str,
    index_mount_path: str,
) -> str:
    return _TEMPLATE.format(
        user_code=user_code,
        trace_mount_path=trace_mount_path,
        index_mount_path=index_mount_path,
    )
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): sandbox bootstrap wrapper"
```

---

### Task 9.5: `sandbox_results.py` — capped stdout/stderr reads and result assembly

**Files:**
- Create: `engine/engine/sandbox/sandbox_results.py`
- Create: `engine/tests/unit/sandbox/test_sandbox_results.py`

- [ ] **Step 1: Tests**

```python
from __future__ import annotations

import asyncio

import pytest

from engine.sandbox.sandbox_config import CodeExecutionResult, SandboxConfig
from engine.sandbox.sandbox_results import run_process_capped


@pytest.mark.asyncio
async def test_simple_stdout_capture() -> None:
    result = await run_process_capped(
        argv=["/bin/sh", "-c", "echo hello; echo bye"],
        config=SandboxConfig(timeout_seconds=5.0),
    )
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_stdout_cap_truncates() -> None:
    result = await run_process_capped(
        argv=["/bin/sh", "-c", "yes A | head -c 100000"],
        config=SandboxConfig(timeout_seconds=5.0, maximum_stdout_bytes=1000),
    )
    assert len(result.stdout.encode()) <= 1000


@pytest.mark.asyncio
async def test_timeout_kills_and_reports() -> None:
    result = await run_process_capped(
        argv=["/bin/sh", "-c", "sleep 10"],
        config=SandboxConfig(timeout_seconds=0.5),
    )
    assert result.timed_out is True
    assert result.exit_code != 0
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

`engine/engine/sandbox/sandbox_results.py`:

```python
from __future__ import annotations

import asyncio
import os
import signal

from engine.sandbox.sandbox_config import CodeExecutionResult, SandboxConfig


async def run_process_capped(
    *,
    argv: list[str],
    config: SandboxConfig,
) -> CodeExecutionResult:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )

    async def _read_capped(stream: asyncio.StreamReader | None, cap: int) -> bytes:
        if stream is None:
            return b""
        buf = bytearray()
        while len(buf) < cap:
            chunk = await stream.read(min(4096, cap - len(buf)))
            if not chunk:
                break
            buf.extend(chunk)
        # Drain the rest without storing it
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
        return bytes(buf)

    try:
        stdout_task = asyncio.create_task(_read_capped(proc.stdout, config.maximum_stdout_bytes))
        stderr_task = asyncio.create_task(_read_capped(proc.stderr, config.maximum_stderr_bytes))

        try:
            exit_code = await asyncio.wait_for(proc.wait(), timeout=config.timeout_seconds)
            timed_out = False
        except asyncio.TimeoutError:
            _kill_process_group(proc.pid)
            await proc.wait()
            exit_code = proc.returncode if proc.returncode is not None else -1
            timed_out = True

        stdout = await stdout_task
        stderr = await stderr_task
    except BaseException:
        _kill_process_group(proc.pid)
        raise

    return CodeExecutionResult(
        exit_code=exit_code,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
        timed_out=timed_out,
    )


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): capped-output subprocess runner for sandbox"
```

---

### Task 9.6: `sandbox_runner.py` — pick platform + orchestrate run

**Files:**
- Create: `engine/engine/sandbox/sandbox_runner.py`
- Create: `engine/tests/unit/sandbox/test_sandbox_runner.py`

- [ ] **Step 1: Test — runs a trivially-true python snippet on Linux when bubblewrap available**

```python
from __future__ import annotations

import platform
import shutil
from pathlib import Path

import pytest

from engine.sandbox.sandbox_config import SandboxConfig
from engine.sandbox.sandbox_runner import SandboxRunner


@pytest.mark.asyncio
async def test_sandboxed_hello_world(tmp_path: Path, fixtures_dir: Path) -> None:
    system = platform.system()
    if system == "Linux" and shutil.which("bwrap") is None:
        pytest.skip("bubblewrap not installed")
    if system == "Darwin" and shutil.which("sandbox-exec") is None:
        pytest.skip("sandbox-exec unavailable")
    if system not in ("Linux", "Darwin"):
        pytest.skip(f"unsupported platform {system}")

    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = tmp_path / "traces.jsonl.engine-index.jsonl"
    meta_path = tmp_path / "traces.jsonl.engine-index.meta.json"
    # Pre-build index using the builder (sync in test)
    from engine.traces.models.trace_index_config import TraceIndexConfig
    from engine.traces.trace_index_builder import TraceIndexBuilder
    await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig(index_path=index_path)
    )

    sandbox_venv = Path(__file__).resolve().parents[3] / ".sandbox-venv"
    if not (sandbox_venv / "bin" / "python").exists():
        pytest.skip("sandbox venv not built — run engine/scripts/build_sandbox_venv.sh")

    runner = SandboxRunner(sandbox_venv=sandbox_venv)
    result = await runner.run_python(
        code="print('count=', trace_store.trace_count)",
        trace_path=trace_path,
        index_path=index_path,
        config=SandboxConfig(timeout_seconds=15.0),
    )
    assert result.exit_code == 0
    assert "count= 3" in result.stdout
```

This test auto-skips unless the environment is ready. It becomes a real assertion once an operator has run `engine/scripts/build_sandbox_venv.sh`.

- [ ] **Step 2: Run — expect skip on fresh setup**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/unit/sandbox/test_sandbox_runner.py -v`
Expected: `skipped` (no failure).

- [ ] **Step 3: Implement**

`engine/engine/sandbox/sandbox_runner.py`:

```python
from __future__ import annotations

import platform
from pathlib import Path

from engine.sandbox.platform_commands import (
    build_linux_bubblewrap_command,
    build_macos_sandbox_exec_command,
    render_macos_profile,
)
from engine.sandbox.sandbox_bootstrap import render_bootstrap_script
from engine.sandbox.sandbox_config import CodeExecutionResult, SandboxConfig
from engine.sandbox.sandbox_policy import compose_policy
from engine.sandbox.sandbox_results import run_process_capped


class SandboxRunner:
    def __init__(self, sandbox_venv: Path) -> None:
        self._sandbox_venv = sandbox_venv

    async def run_python(
        self,
        *,
        code: str,
        trace_path: Path,
        index_path: Path,
        config: SandboxConfig,
    ) -> CodeExecutionResult:
        system = platform.system()

        import tempfile
        with tempfile.TemporaryDirectory(prefix="halo-sbx-") as tmp:
            work_dir = Path(tmp)
            (work_dir / "tmp").mkdir()
            script = work_dir / "bootstrap.py"

            script_body = render_bootstrap_script(
                user_code=code,
                trace_mount_path="/mnt/trace/traces.jsonl" if system == "Linux" else str(trace_path),
                index_mount_path="/mnt/trace/traces.jsonl.engine-index.jsonl" if system == "Linux" else str(index_path),
            )
            script.write_text(script_body)

            policy = compose_policy(
                trace_path=trace_path,
                index_path=index_path,
                sandbox_venv=self._sandbox_venv,
                work_dir=work_dir,
                sandbox_config=config,
            )

            if system == "Linux":
                # bubblewrap maps the script into /workspace; adjust path accordingly.
                in_sandbox_script = Path("/workspace/bootstrap.py")
                argv = build_linux_bubblewrap_command(policy=policy, script_path=in_sandbox_script)
            elif system == "Darwin":
                profile_path = work_dir / "profile.sb"
                profile_path.write_text(render_macos_profile(policy=policy))
                argv = build_macos_sandbox_exec_command(
                    policy=policy, script_path=script, profile_path=profile_path
                )
            else:
                raise RuntimeError(f"unsupported platform {system}")

            return await run_process_capped(argv=argv, config=config)
```

- [ ] **Step 4: Run — expect skip or pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): SandboxRunner orchestrating platform sandbox"
```

---

### Task 9.7: `run_code_tool.py`

**Files:**
- Create: `engine/engine/tools/run_code_tool.py`
- Create: `engine/tests/unit/tools/test_run_code_tool.py`

- [ ] **Step 1: Test**

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from engine.sandbox.sandbox_config import CodeExecutionResult, RunCodeArguments, SandboxConfig
from engine.tools.run_code_tool import RunCodeTool
from engine.tools.tool_protocol import ToolContext


@pytest.mark.asyncio
async def test_run_code_tool_delegates_to_sandbox_runner(tmp_path: Path, fixtures_dir: Path) -> None:
    trace_path = tmp_path / "t.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    index_path = tmp_path / "t.idx.jsonl"
    index_path.write_text("")

    from engine.traces.trace_store import TraceStore
    # build a real index for trace_store
    from engine.traces.models.trace_index_config import TraceIndexConfig
    from engine.traces.trace_index_builder import TraceIndexBuilder
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    store = TraceStore.load(trace_path=trace_path, index_path=index_path)

    fake_runner = AsyncMock()
    fake_runner.run_python = AsyncMock(return_value=CodeExecutionResult(
        exit_code=0, stdout="ok", stderr="", timed_out=False,
    ))
    ctx = ToolContext(trace_store=store, sandbox_runner=fake_runner)

    tool = RunCodeTool(sandbox_config=SandboxConfig())
    result = await tool.run(ctx, RunCodeArguments(code="print('hello')"))
    assert result.exit_code == 0
    fake_runner.run_python.assert_awaited_once()
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement**

`engine/engine/tools/run_code_tool.py`:

```python
from __future__ import annotations

from pathlib import Path

from engine.sandbox.sandbox_config import (
    CodeExecutionResult,
    RunCodeArguments,
    SandboxConfig,
)
from engine.sandbox.sandbox_runner import SandboxRunner
from engine.tools.tool_protocol import ToolContext


class RunCodeTool:
    name = "run_code"
    description = (
        "Execute Python code in a sandbox with read-only access to the trace dataset. "
        "numpy, pandas, and a preloaded trace_store variable are available."
    )
    arguments_model = RunCodeArguments
    result_model = CodeExecutionResult

    def __init__(
        self,
        sandbox_config: SandboxConfig,
        sandbox_venv: Path | None = None,
    ) -> None:
        self._sandbox_config = sandbox_config
        self._default_venv = sandbox_venv or Path(__file__).resolve().parents[2] / ".sandbox-venv"

    async def run(self, tool_context: ToolContext, arguments: RunCodeArguments) -> CodeExecutionResult:
        runner = tool_context.sandbox_runner or SandboxRunner(sandbox_venv=self._default_venv)
        store = tool_context.require_trace_store()
        # TraceStore holds its own paths; expose them through accessors on TraceStore.
        return await runner.run_python(
            code=arguments.code,
            trace_path=store._trace_path,  # intentional: private accessor ok within package
            index_path=store._index_path,
            config=self._sandbox_config,
        )
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): run_code tool"
```

---

### Task 9.8: Denied-operation tests for the sandbox

**Files:**
- Create: `engine/tests/unit/sandbox/test_sandbox_policy_denials.py`

- [ ] **Step 1: Tests (auto-skip if sandbox venv missing or platform unsupported)**

```python
from __future__ import annotations

import platform
import shutil
from pathlib import Path

import pytest

from engine.sandbox.sandbox_config import SandboxConfig
from engine.sandbox.sandbox_runner import SandboxRunner
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.trace_index_builder import TraceIndexBuilder


def _sandbox_ready(tmp_path: Path, fixtures_dir: Path) -> tuple[SandboxRunner, Path, Path]:
    system = platform.system()
    if system == "Linux" and shutil.which("bwrap") is None:
        pytest.skip("bubblewrap not installed")
    if system not in ("Linux", "Darwin"):
        pytest.skip(f"unsupported platform {system}")
    venv = Path(__file__).resolve().parents[3] / ".sandbox-venv"
    if not (venv / "bin" / "python").exists():
        pytest.skip("sandbox venv not built")

    trace_path = tmp_path / "t.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    return SandboxRunner(sandbox_venv=venv), trace_path, tmp_path


@pytest.mark.asyncio
async def test_cannot_write_outside_workspace(tmp_path: Path, fixtures_dir: Path) -> None:
    runner, trace_path, _ = _sandbox_ready(tmp_path, fixtures_dir)
    idx = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    result = await runner.run_python(
        code="open('/etc/attack', 'w').write('no')",
        trace_path=trace_path, index_path=idx,
        config=SandboxConfig(timeout_seconds=10.0),
    )
    assert result.exit_code != 0
    assert "PermissionError" in result.stderr or "Read-only" in result.stderr or "not permitted" in result.stderr


@pytest.mark.asyncio
async def test_cannot_read_outside_allowed(tmp_path: Path, fixtures_dir: Path) -> None:
    runner, trace_path, _ = _sandbox_ready(tmp_path, fixtures_dir)
    idx = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    result = await runner.run_python(
        code="print(open('/etc/passwd').read()[:10])",
        trace_path=trace_path, index_path=idx,
        config=SandboxConfig(timeout_seconds=10.0),
    )
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_no_network(tmp_path: Path, fixtures_dir: Path) -> None:
    runner, trace_path, _ = _sandbox_ready(tmp_path, fixtures_dir)
    idx = await TraceIndexBuilder.ensure_index_exists(
        trace_path=trace_path, config=TraceIndexConfig()
    )
    result = await runner.run_python(
        code=(
            "import socket; s = socket.socket(); "
            "s.connect(('1.1.1.1', 80))"
        ),
        trace_path=trace_path, index_path=idx,
        config=SandboxConfig(timeout_seconds=5.0),
    )
    assert result.exit_code != 0
```

- [ ] **Step 2: Run — expect skip or pass**

- [ ] **Step 3: Commit**

```bash
git -C /home/declan/dev/HALO add engine/tests/unit/sandbox/test_sandbox_policy_denials.py
git -C /home/declan/dev/HALO commit -m "test(engine): sandbox policy denial assertions"
```

---

## Phase 10 — End-to-end smoke test

### Task 10.1: E2E test against fixture trace + public API

The user will supply a real trace fixture. For now we check that the engine runs to completion on the tiny fixture with fake-but-realistic inputs and emits at least one output item with `final=True`. The assertions are loose (per user: "for now we want it to execute and I will judge success").

**Files:**
- Create: `engine/tests/integration/__init__.py` (empty)
- Create: `engine/tests/integration/test_engine_e2e.py`

- [ ] **Step 1: Write the E2E test**

```python
from __future__ import annotations

import os
from pathlib import Path

import pytest

from engine.agents.agent_config import AgentConfig
from engine.engine_config import EngineConfig
from engine.main import run_engine_async
from engine.model_config import ModelConfig
from engine.models.messages import AgentMessage


@pytest.mark.asyncio
async def test_engine_runs_on_tiny_fixture(tmp_path: Path, fixtures_dir: Path) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; E2E requires real LLM access")

    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())

    agent = AgentConfig(
        name="root",
        instructions="Answer briefly.",
        model=ModelConfig(name="gpt-5.4-mini"),
        maximum_turns=6,
    )
    cfg = EngineConfig(
        root_agent=agent,
        subagent=agent.model_copy(update={"name": "sub"}),
        synthesis_model=ModelConfig(name="gpt-5.4-mini"),
        compaction_model=ModelConfig(name="gpt-5.4-mini"),
        maximum_depth=1,
        maximum_parallel_subagents=2,
    )

    messages = [AgentMessage(
        role="user",
        content="Use dataset_overview to report how many traces exist.",
    )]

    results = await run_engine_async(messages, cfg, trace_path)
    assert len(results) >= 1
    assert any(item.final for item in results)
```

- [ ] **Step 2: Run — expect skip in CI; pass locally with OPENAI_API_KEY**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest tests/integration/test_engine_e2e.py -v`
Expected: skipped unless `OPENAI_API_KEY` is set and sandbox venv is built.

- [ ] **Step 3: Commit**

```bash
git -C /home/declan/dev/HALO add engine/tests/integration
git -C /home/declan/dev/HALO commit -m "test(engine): e2e smoke against tiny fixture"
```

---

### Task 10.2: Final full-suite run + cleanup commit

- [ ] **Step 1: Run the full suite**

Run: `cd /home/declan/dev/HALO/engine && uv run pytest -q`
Expected: all non-skipped tests pass.

- [ ] **Step 2: Run ruff**

Run: `cd /home/declan/dev/HALO/engine && uv run ruff check .`
Expected: no errors. Fix any drift.

- [ ] **Step 3: Final commit if any fix-ups were needed**

```bash
git -C /home/declan/dev/HALO commit -am "chore(engine): green suite + ruff clean" || true
```

---

## Self-Review

**Spec coverage (walked top to bottom against `docs/engine-architecture-plan.md`):**

| Spec section | Task(s) |
|---|---|
| Public entrypoints (`stream_engine_async`, `run_engine_async`, sync) | 7.2 |
| Package structure | 0.1; file tree above |
| Naming (`Engine*` prefix, avoid RLM) | Applied throughout |
| Typing (Pydantic everywhere) | 1.1–1.12, 4.x, 5.x |
| Config (`ModelConfig`, `AgentConfig`, `TraceIndexConfig`, `SandboxConfig`, `EngineConfig`) | 1.4, 1.5, 1.6, 1.7, 1.8 |
| Trace input format (canonical span) | 1.10 |
| `TraceIndexBuilder` build-once + atomic write | 2.1, 2.2, 2.3 |
| `TraceStore` pure API (view/query/count/overview/search/render) | 3.1–3.7 |
| `run_code` tool + sandbox mounts + numpy/pandas + no-network | 9.1–9.8 |
| `AgentContextItem`, `AgentContext`, per-item compaction, system never compacted, tool-call invariant (via individual compaction per user guidance) | 5.1, 5.2, 5.3 |
| Prompt templates + final sentinel | 5.4 |
| `get_context_item` tool | 5.5 |
| `AgentOutputItem` + `AgentTextDelta` for streaming | 1.3, 6.4, 7.1 |
| `EngineOutputBus` single run-level queue + fail path | 6.1 |
| `EngineRunState` + `AgentExecution` | 6.2, 6.3 |
| OpenAI SDK event mapping (final sentinel, deltas, tool call/output) | 6.4 |
| Subagents via `Agent.as_tool` + structural depth hard-stop + semaphore | 8.2 |
| Synthesis tool | 8.1 |
| Parallel subagents interleaved through bus with lineage | 6.1 + 8.2 |
| Error handling (circuit breaker at 10 failures; tool-error returned to parent) | 7.1; tool adapter already propagates typed results |
| Path of execution (index → store → run_state → bus → root agent → stream) | 7.2 |
| Implementation phases 1–10 | Phases 0–10 of this plan |

**Placeholder scan:** no "TBD", "TODO", or "similar to Task N" references remain. Every code block is concrete and runnable as written, with the exception of the SDK attribute names in Task 6.4 and Task 8.2 where the accompanying implementation notes tell the executor to adjust against the installed `openai-agents` version — this is deliberate because those names are the one surface we could not pin without the SDK present.

**Type consistency spot checks:**
- `AgentMessage.content` is always `MessageContent = str | list[ContentPart] | None` — used consistently in Tasks 1.2, 5.2, 6.4.
- `AgentContextItem.is_compacted` / `compaction_summary` — set only in `AgentContext.compact_old_items` (Task 5.3), read in `_render_item` (Task 5.2) and `GetContextItemTool` (Task 5.5).
- `TraceFilters` fields — defined in 1.12, consumed in 3.3–3.5, surfaced in tool args in 4.3.
- `SubagentToolResult` fields — defined in 8.2, populated in the same task's `guarded_invoke`.
- `CodeExecutionResult` — defined in 1.7, returned by 9.5 and 9.7.
- `SandboxPolicy.readonly_paths[-1]` assumed to be the venv; `compose_policy` (9.2) enforces this order, and `platform_commands` (9.3) reads it.

**Known residual risks flagged for the implementer:**
1. `openai-agents` event attribute names (Task 6.4) and `Agent.as_tool` streaming semantics (Task 8.2) are the most SDK-version-sensitive surfaces. The plan provides fallback strategies (injectable `run_streamed`; manual `Runner.run_streamed` drive loop in subagent invoker) so neither blocks progress.
2. `TraceStore._trace_path` / `_index_path` are read by `RunCodeTool` (Task 9.7) via underscore attributes. If a future refactor makes them truly private, add public properties then.
3. Sandbox denial tests (9.8) require a running sandbox — they'll auto-skip in CI until `build_sandbox_venv.sh` is executed on the host. That's intentional.

---

## Phase 11 — Hardening (post-E2E amendments)

Added 2026-04-25 after E2E revealed gaps. Tasks target real-risk issues surfaced in the first live run.

**Design decision (circuit breaker & failure propagation):**
- **4xx errors are not retried.** They are permanent — retrying wastes time and floods logs. Only `openai.APIConnectionError`, `openai.APITimeoutError`, `openai.RateLimitError`, and 5xx `openai.APIStatusError` are retriable.
- **Only root-agent exhaustion raises.** Subagent failures (including circuit-breaker exhaustion or 4xx) are caught inside `guarded_invoke` and returned to the parent as a `SubagentToolResult` with an `answer` that describes the failure. The parent agent then decides whether to retry, try a different approach, or report back.
- `EngineAgentExhaustedError` stays as the exception type for root exhaustion; it closes the stream via `EngineOutputBus.fail(error)`.

### Task 11.1: Classify retriable vs non-retriable errors in `OpenAiAgentRunner`

**Files:**
- Modify: `engine/engine/agents/openai_agent_runner.py`
- Modify: `engine/tests/unit/agents/test_openai_agent_runner.py`

- [ ] **Step 1: Append failing tests**

```python
from openai import APIConnectionError, BadRequestError
import httpx


@pytest.mark.asyncio
async def test_runner_does_not_retry_on_bad_request() -> None:
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root", agent_name="root", depth=0,
        parent_agent_id=None, parent_tool_call_id=None,
    )

    call_count = 0
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    fake_response = httpx.Response(400, request=fake_request)

    async def raise_400(*, agent, input, context):
        nonlocal call_count
        call_count += 1
        raise BadRequestError(
            message="bad field",
            response=fake_response,
            body={"error": {"message": "bad field"}},
        )

    async def noop_compactor(_):
        return ""

    runner = OpenAiAgentRunner(
        run_streamed=raise_400,
        compactor_factory=lambda _: noop_compactor,
    )

    with pytest.raises(BadRequestError):
        await runner.run(
            sdk_agent=object(),
            agent_context=ctx,
            agent_execution=execution,
            output_bus=bus,
            is_root=True,
        )
    assert call_count == 1  # no retries


@pytest.mark.asyncio
async def test_runner_retries_on_connection_error_then_fails() -> None:
    bus = EngineOutputBus()
    ctx = _context()
    execution = AgentExecution(
        agent_id="root", agent_name="root", depth=0,
        parent_agent_id=None, parent_tool_call_id=None,
    )

    call_count = 0
    fake_request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    async def raise_connection(*, agent, input, context):
        nonlocal call_count
        call_count += 1
        raise APIConnectionError(request=fake_request)

    async def noop_compactor(_):
        return ""

    runner = OpenAiAgentRunner(
        run_streamed=raise_connection,
        compactor_factory=lambda _: noop_compactor,
    )

    with pytest.raises(EngineAgentExhaustedError):
        await runner.run(
            sdk_agent=object(),
            agent_context=ctx,
            agent_execution=execution,
            output_bus=bus,
            is_root=True,
        )
    assert call_count == 10  # full retry budget consumed
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Patch impl**

Add at module top:

```python
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError


def _is_retriable_llm_error(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    return False
```

Modify the `except Exception as exc:` branch of `run` to:

```python
            except Exception as exc:
                if not _is_retriable_llm_error(exc):
                    raise
                last_exc = exc
                agent_execution.record_llm_failure()
                logger.warning(
                    "llm call failed for agent_id={} (failure {} of {})",
                    agent_execution.agent_id,
                    agent_execution.consecutive_llm_failures,
                    MAX_CONSECUTIVE_LLM_FAILURES,
                )
                continue
```

- [ ] **Step 4: Run — expect `4 passed`** (2 pre-existing + 2 new)

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "fix(engine): do not retry non-retriable LLM errors (4xx)"
```

---

### Task 11.2: Subagent failures become tool errors, not stream crashes

**Files:**
- Modify: `engine/engine/tools/subagent_tool_factory.py`
- Modify: `engine/tests/unit/tools/test_subagent_tool_factory.py`

The goal: wrap the `guarded_invoke` body in a try/except. On any exception, return a `SubagentToolResult` whose `answer` describes the failure, so the parent agent keeps running.

- [ ] **Step 1: Append failing test**

```python
@pytest.mark.asyncio
async def test_guarded_invoke_returns_failure_on_exception() -> None:
    # Construct a minimal run_state + subagent tool, then monkeypatch Runner.run_streamed
    # to raise. The tool's on_invoke_tool must NOT raise — it must return a serialized
    # SubagentToolResult with the failure described in `answer`.
    import asyncio
    from engine.agents.engine_output_bus import EngineOutputBus
    from engine.agents.engine_run_state import EngineRunState
    from engine.agents.agent_config import AgentConfig
    from engine.engine_config import EngineConfig
    from engine.model_config import ModelConfig
    from engine.tools.subagent_result import SubagentToolResult
    from engine.tools.subagent_tool_factory import _build_subagent_as_tool
    from engine.traces.trace_store import TraceStore

    cfg = EngineConfig(
        root_agent=AgentConfig(name="r", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3),
        subagent=AgentConfig(name="s", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3),
        synthesis_model=ModelConfig(name="gpt-5.4-mini"),
        compaction_model=ModelConfig(name="gpt-5.4-mini"),
        maximum_depth=1,
    )
    fake_store = MagicMock(spec=TraceStore)
    run_state = EngineRunState(trace_store=fake_store, output_bus=EngineOutputBus(), config=cfg)

    sem = asyncio.Semaphore(1)
    tool = _build_subagent_as_tool(run_state=run_state, child_depth=1, semaphore=sem)

    # Patch Runner.run_streamed to raise
    from engine.tools import subagent_tool_factory as mod

    class _Boom:
        pass

    def _raise(*args, **kwargs):
        raise RuntimeError("SDK exploded")

    orig = mod.Runner.run_streamed
    mod.Runner.run_streamed = _raise
    try:
        result_json = await tool.on_invoke_tool(None, "{}")
    finally:
        mod.Runner.run_streamed = orig

    result = SubagentToolResult.model_validate_json(result_json)
    assert "SDK exploded" in result.answer or "failed" in result.answer.lower()
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Wrap `guarded_invoke`'s body in try/except**

Inside the `async with semaphore:` block of `guarded_invoke`, wrap everything in try/except. On failure, build and return a `SubagentToolResult` describing the error. See the patch below:

```python
        async with semaphore:
            child_execution = AgentExecution(
                agent_id=f"sub-{uuid.uuid4().hex[:8]}",
                agent_name=engine_config.subagent.name,
                depth=child_depth,
                parent_agent_id=None,
                parent_tool_call_id=None,
            )
            run_state.register(child_execution)
            start_seq: int | None = None
            end_seq: int | None = None
            try:
                stream = Runner.run_streamed(
                    starting_agent=child_agent,
                    input=raw_arguments,
                    context=run_state,
                )
                async for ev in stream.stream_events():
                    mapped = mapper.to_mapped_event(ev, execution=child_execution, is_root=False)
                    if mapped.output_item is not None:
                        emitted = await output_bus.emit(mapped.output_item)
                        if start_seq is None:
                            start_seq = emitted.sequence
                        end_seq = emitted.sequence
                    if mapped.delta is not None:
                        await output_bus.emit(mapped.delta)

                run_result = stream
                if hasattr(stream, "wait_for_final_output"):
                    run_result = await stream.wait_for_final_output()

                extracted_json = await custom_output_extractor(run_result)
                result = SubagentToolResult.model_validate_json(extracted_json).model_copy(update={
                    "child_agent_id": child_execution.agent_id,
                    "output_start_sequence": start_seq or 0,
                    "output_end_sequence": end_seq or 0,
                    "turns_used": child_execution.turns_used,
                    "tool_calls_made": child_execution.tool_calls_made,
                })
                return result.model_dump_json()
            except Exception as exc:
                from loguru import logger

                logger.warning(
                    "subagent {} failed at depth={}: {}",
                    child_execution.agent_id, child_depth, exc,
                )
                failure = SubagentToolResult(
                    child_agent_id=child_execution.agent_id,
                    answer=f"Subagent failed: {type(exc).__name__}: {exc}",
                    output_start_sequence=start_seq or 0,
                    output_end_sequence=end_seq or 0,
                    turns_used=child_execution.turns_used,
                    tool_calls_made=child_execution.tool_calls_made,
                )
                return failure.model_dump_json()
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "fix(engine): subagent failures return tool errors instead of crashing"
```

---

### Task 11.3: Track `turns_used` and `tool_calls_made` in subagent execution

**Files:**
- Modify: `engine/engine/tools/subagent_tool_factory.py`
- Modify: `engine/tests/unit/tools/test_subagent_tool_factory.py`

- [ ] **Step 1: Append failing test**

```python
@pytest.mark.asyncio
async def test_guarded_invoke_counts_turns_and_tool_calls(monkeypatch) -> None:
    import asyncio
    from types import SimpleNamespace
    from engine.agents.engine_output_bus import EngineOutputBus
    from engine.agents.engine_run_state import EngineRunState
    from engine.agents.agent_config import AgentConfig
    from engine.engine_config import EngineConfig
    from engine.model_config import ModelConfig
    from engine.tools.subagent_result import SubagentToolResult
    from engine.tools.subagent_tool_factory import _build_subagent_as_tool
    from engine.traces.trace_store import TraceStore
    from engine.tools import subagent_tool_factory as mod

    cfg = EngineConfig(
        root_agent=AgentConfig(name="r", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3),
        subagent=AgentConfig(name="s", instructions="", model=ModelConfig(name="gpt-5.4-mini"), maximum_turns=3),
        synthesis_model=ModelConfig(name="gpt-5.4-mini"),
        compaction_model=ModelConfig(name="gpt-5.4-mini"),
        maximum_depth=1,
    )
    fake_store = MagicMock(spec=TraceStore)
    run_state = EngineRunState(trace_store=fake_store, output_bus=EngineOutputBus(), config=cfg)

    # Synthesize 1 tool_call event and 1 message_output event, then completion
    events = [
        SimpleNamespace(type="run_item_stream_event", item=SimpleNamespace(
            type="tool_call_item",
            raw_item=SimpleNamespace(call_id="c1", id="c1", name="query_traces", arguments="{}"),
        )),
        SimpleNamespace(type="run_item_stream_event", item=SimpleNamespace(
            type="message_output_item",
            raw_item=SimpleNamespace(id="m1", role="assistant", content=[
                SimpleNamespace(type="output_text", text="done")
            ]),
        )),
    ]

    class _Stream:
        new_items: list = []

        async def stream_events(self_inner):
            for e in events:
                yield e

        async def wait_for_final_output(self_inner):
            return self_inner

    def fake_run_streamed(*args, **kwargs):
        return _Stream()

    monkeypatch.setattr(mod.Runner, "run_streamed", fake_run_streamed)

    sem = asyncio.Semaphore(1)
    tool = _build_subagent_as_tool(run_state=run_state, child_depth=1, semaphore=sem)
    result_json = await tool.on_invoke_tool(None, "{}")
    result = SubagentToolResult.model_validate_json(result_json)
    assert result.turns_used == 1
    assert result.tool_calls_made == 1
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Update `guarded_invoke` to count events before each dispatch to bus**

Inside the `async for ev in stream.stream_events():` loop, add:

```python
                    # Count turns and tool calls on the child execution
                    if isinstance(mapped.output_item, AgentOutputItem):
                        item = mapped.output_item.item
                        if item.role == "assistant" and item.tool_calls:
                            child_execution.tool_calls_made += len(item.tool_calls)
                        elif item.role == "assistant":
                            child_execution.turns_used += 1
```

Add the import at the top of the module:

```python
from engine.models.engine_output import AgentOutputItem
```

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "feat(engine): subagent counts turns_used and tool_calls_made"
```

---

### Task 11.4: Compaction E2E test

Trigger compaction in a real conversation by setting thresholds very low and asking the agent a multi-turn question. Assert that at least one `AgentContextItem` ends up `is_compacted=True` after the run. Because we can't easily read the internal context from outside, we instead verify compaction fires by using a *fake* compactor callable and counting calls. This is the unit-level test we already have; the E2E version runs the whole engine with a real model and asserts no crash with tight thresholds.

**Files:**
- Create: `engine/tests/integration/test_engine_compaction.py`

```python
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from engine.agents.agent_config import AgentConfig
from engine.engine_config import EngineConfig
from engine.main import run_engine_async
from engine.model_config import ModelConfig
from engine.models.messages import AgentMessage


E2E_MODEL = os.environ.get("HALO_E2E_MODEL", "gpt-5.4-mini")
E2E_TIMEOUT_SECONDS = float(os.environ.get("HALO_E2E_TIMEOUT", "90"))


@pytest.mark.asyncio
async def test_engine_compaction_fires_without_crash(tmp_path: Path, fixtures_dir: Path) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; E2E requires real LLM access")

    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())

    agent = AgentConfig(
        name="root",
        instructions="Answer concisely.",
        model=ModelConfig(name=E2E_MODEL),
        maximum_turns=4,
    )
    cfg = EngineConfig(
        root_agent=agent,
        subagent=agent.model_copy(update={"name": "sub"}),
        synthesis_model=ModelConfig(name=E2E_MODEL),
        compaction_model=ModelConfig(name=E2E_MODEL),
        # Force compaction to trigger even on small conversations
        text_message_compaction_keep_last_messages=1,
        tool_call_compaction_keep_last_messages=1,
        maximum_depth=0,
        maximum_parallel_subagents=1,
    )

    messages = [AgentMessage(
        role="user",
        content=(
            "Call get_dataset_overview, then count_traces with has_errors=true, "
            "then tell me the two numbers and end with <final/>."
        ),
    )]

    async with asyncio.timeout(E2E_TIMEOUT_SECONDS):
        results = await run_engine_async(messages, cfg, trace_path)

    assert any(item.final for item in results)
```

- [ ] **Step 1: Create the test**

- [ ] **Step 2: Run it**

`cd /home/declan/dev/HALO/engine && uv run pytest tests/integration/test_engine_compaction.py -v`
Expected: `1 passed` against live API. If it fails, capture the error — a likely candidate is the compaction prompt shape breaking on the chosen model.

- [ ] **Step 3: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "test(engine): e2e compaction does not crash with tight thresholds"
```

---

### Task 11.5: Depth=1 subagent E2E

**Files:**
- Modify: `engine/tests/integration/test_engine_e2e.py` OR create `engine/tests/integration/test_engine_subagent.py`

Create a new test file rather than modifying the existing happy-path E2E:

```python
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from engine.agents.agent_config import AgentConfig
from engine.engine_config import EngineConfig
from engine.main import run_engine_async
from engine.model_config import ModelConfig
from engine.models.messages import AgentMessage


E2E_MODEL = os.environ.get("HALO_E2E_MODEL", "gpt-5.4-mini")
E2E_TIMEOUT_SECONDS = float(os.environ.get("HALO_E2E_TIMEOUT", "120"))


@pytest.mark.asyncio
async def test_root_delegates_to_subagent(tmp_path: Path, fixtures_dir: Path) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; E2E requires real LLM access")

    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())

    agent = AgentConfig(
        name="root",
        instructions="Answer briefly.",
        model=ModelConfig(name=E2E_MODEL),
        maximum_turns=6,
    )
    cfg = EngineConfig(
        root_agent=agent,
        subagent=agent.model_copy(update={"name": "sub"}),
        synthesis_model=ModelConfig(name=E2E_MODEL),
        compaction_model=ModelConfig(name=E2E_MODEL),
        maximum_depth=1,
        maximum_parallel_subagents=2,
    )

    messages = [AgentMessage(
        role="user",
        content=(
            "Delegate this question to a subagent via call_subagent: "
            "'How many traces have errors? Use count_traces with has_errors=true.' "
            "Then report the subagent's answer to me and end with <final/>."
        ),
    )]

    async with asyncio.timeout(E2E_TIMEOUT_SECONDS):
        results = await run_engine_async(messages, cfg, trace_path)

    # A subagent tool call should have been made at depth 0 by the root
    subagent_calls = [
        item for item in results
        if item.depth == 0 and item.item.tool_calls and any(
            tc.function.name == "call_subagent" for tc in item.item.tool_calls
        )
    ]
    assert subagent_calls, "root did not call call_subagent"

    # The child agent should have emitted output items at depth 1
    child_items = [item for item in results if item.depth == 1]
    assert child_items, "no depth=1 items streamed — child stream not forwarded to bus"

    assert any(item.final for item in results)
```

- [ ] **Step 1: Create the test**

- [ ] **Step 2: Run**

`cd /home/declan/dev/HALO/engine && uv run pytest tests/integration/test_engine_subagent.py -v`

- [ ] **Step 3: If the test reveals more wiring bugs (common expected: `ctx.context` vs `ctx` confusion, tool arg shape for `call_subagent`, etc.), investigate and fix in a follow-up task rather than hacking around it.**

- [ ] **Step 4: Commit**

```bash
git -C /home/declan/dev/HALO commit -am "test(engine): e2e subagent delegation at depth 1"
```

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-24-halo-engine.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using the `superpowers:executing-plans` skill, batch execution with checkpoints.

Which approach?


