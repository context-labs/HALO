# Engine Architecture Plan

## Goal

Build a new standalone `engine` package for the re-architected RLM runtime.

The package receives:

- A chat messages array.
- Runtime configuration.
- A path to canonical flat OTel span traces.

It runs the Engine function and returns a full ordered array of all messages from every agent that ran. Assistant tool calls are represented as assistant messages with a `tool_calls` field. Tool results are represented as `tool` role messages. The final item is always the root agent's final assistant response.

## Public Entrypoints

Async is primary. Sync wrappers exist for convenience.

```python
async def stream_engine_async(messages, engine_config, trace_path): ...
async def run_engine_async(messages, engine_config, trace_path): ...

def stream_engine(messages, engine_config, trace_path): ...
def run_engine(messages, engine_config, trace_path): ...
```

`run_*` only aggregates `stream_*`.

These functions live in `engine/main.py`. The file is named `main.py` because it is the package's main programmatic entrypoint, not an HTTP API layer.

## Package Structure

```text
engine/
  __init__.py
  main.py                 # public run/stream entrypoints
  engine_config.py        # EngineConfig, composed from domain configuration models
  model_config.py         # AvailableModelName union and ModelConfig

  models/
    __init__.py
    engine_output.py      # AgentOutputItem lineage wrapper
    json_value.py         # shared JSON value aliases/wrappers
    messages.py           # OpenAI/HF-compatible AgentMessage shapes

  agents/
    agent_config.py       # AgentConfig
    agent_context.py      # AgentContext + compaction behavior
    agent_context_items.py # AgentContextItem and compaction models
    agent_execution.py    # AgentExecution and per-agent run summaries
    engine_output_bus.py  # async queue + ordered output delivery
    engine_run_state.py   # shared state for one Engine run
    openai_agent_runner.py # OpenAI Agents SDK orchestration
    openai_event_mapper.py # SDK event normalization into AgentOutputItem
    prompt_templates.py   # root/subagent/synthesis prompt builders

  traces/
    models/
      __init__.py
      canonical_span.py     # canonical flat span pydantic model
      trace_index_config.py # TraceIndexConfig
      trace_index_models.py # trace index row/meta models
      trace_query_models.py # trace filter/query/result models
    trace_index_builder.py # TraceIndexBuilder / ensure-index logic
    trace_store.py        # pure TraceStore query/view/count/search/render API

  tools/
    tool_protocol.py      # EngineTool interface and OpenAI SDK adapter
    trace_tools.py        # overview/query/count/view/search tools
    agent_context_tools.py # get context item
    synthesis_tool.py     # summarize selected traces
    subagent_tool_factory.py # Agent.as_tool streaming/extractor wiring
    run_code_tool.py      # run_code tool

  sandbox/
    __init__.py
    sandbox_config.py     # SandboxConfig
    sandbox_runner.py     # chooses backend and executes sandboxed Python
    sandbox_policy.py     # read-only mounts, writable temp dir, no-network policy
    sandbox_bootstrap.py  # script template that exposes TraceStore to user code
    platform_commands.py  # linux/macOS sandbox command builder functions
    sandbox_results.py    # process result parsing/output caps
```

## Naming

Use `Engine` naming for public runtime classes and variables.

Examples:

```python
EngineConfig
AgentOutputItem
AgentMessage
EngineToolCall
EngineRunState
EngineOutputBus
```

Avoid `RLM` prefixes in new code.

## Typing

Everything model-related should be strongly typed with Pydantic models or explicit type aliases. Tool arguments, tool results, trace query inputs, trace query results, context items, emitted output items, and sandbox results should not be raw dictionaries.

Messages stay compatible with OpenAI/HF messages array format. `AgentMessage` uses the standard fields (`role`, `content`, `tool_calls`, `tool_call_id`, `name`) and adds Engine-specific metadata fields for ids, lineage, compaction, and debugging. Tool calls are not separate context items; they live on one assistant message's `tool_calls` array. Tool results are `role="tool"` messages.

The root `engine.models` package is only for public cross-cutting runtime models shared across components. Domain-specific models live with the domain, for example `engine.traces.models.*`, `engine.agents.*`, and `engine.sandbox.*`.

## Config

Keep configuration small until implementation reveals exact SDK needs.

```python
AvailableModelName = Literal[
    "claude-opus-4-7",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "gpt-5.4",
    "gpt-5.4-mini",
]


class ModelConfig(BaseModel):
    name: AvailableModelName
    temperature: float | None = None
    maximum_output_tokens: int | None = None
    parallel_tool_calls: bool = True


class AgentConfig(BaseModel):
    name: str
    instructions: str
    model: ModelConfig
    maximum_turns: int
```

`AgentConfig` is used when constructing OpenAI Agents SDK `Agent` instances.

Location: `engine/agents/agent_config.py`.

`AvailableModelName` and `ModelConfig` live in `engine/model_config.py`. The model name is a literal union so supported models are explicit in code and configuration validation fails fast for unknown names.

```python
class TraceIndexConfig(BaseModel):
    index_path: Path | None = None
    schema_version: int = 1
```

`TraceIndexConfig` only controls sidecar index creation/loading behavior.

Location: `engine/traces/models/trace_index_config.py`.

It answers:

- Where should the index file live?
- Which index schema version should be written for a new index?

It does not control trace query behavior.

The index is build-once. If the index file already exists, the Engine loads it and does not compare source file size, modified time, or content hashes. If the trace file changes, the caller must provide a new index path or delete the old index before running.

```python
class EngineConfig(BaseModel):
    root_agent: AgentConfig
    subagent: AgentConfig
    synthesis_model: ModelConfig
    compaction_model: ModelConfig
    trace_index: TraceIndexConfig = TraceIndexConfig()
    sandbox: SandboxConfig = SandboxConfig()
    text_message_compaction_keep_last_messages: int = 12
    tool_call_compaction_keep_last_messages: int = 6
    maximum_depth: int = 2
    maximum_parallel_subagents: int = 4
```

Use explicit configuration field names without abbreviations. `compaction_model` is used only by `AgentContext` when replacing older context items with compacted summaries. Text messages and tool-call-related messages have separate keep-last thresholds because tool traffic tends to be larger and can be compacted more aggressively.

Location: `engine/engine_config.py`.

## Trace Input Format

There is one supported trace input format: flat JSONL with one canonical span per line.

```json
{
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7",
  "parent_span_id": "",
  "trace_state": "",
  "name": "anthropic.messages.create",
  "kind": "SPAN_KIND_CLIENT",
  "start_time": "2026-04-23T05:32:00.123456789Z",
  "end_time": "2026-04-23T05:32:00.623456789Z",
  "status": { "code": "STATUS_CODE_OK", "message": "" },
  "resource": {
    "attributes": {
      "service.name": "my-agent",
      "service.version": "0.1.0",
      "deployment.environment": "prod"
    }
  },
  "scope": {
    "name": "@arizeai/openinference-instrumentation-anthropic",
    "version": "0.1.10"
  },
  "attributes": {
    // Original upstream attributes, preserved verbatim.
    "openinference.span.kind": "LLM",
    "llm.provider": "anthropic",
    "llm.model_name": "claude-sonnet-4-5",
    "llm.token_count.prompt": 123,
    "llm.token_count.completion": 456,

    // Tool definitions available to the LLM (OpenInference convention).
    "llm.tools.0.tool.json_schema": "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get current weather for a location\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}",

    // Tool calls emitted by the LLM in output messages.
    "llm.output_messages.0.message.role": "assistant",
    "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_abc123",
    "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
    "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{\"location\":\"San Francisco, CA\"}",

    // Tool results appear as input messages with role "tool".
    "llm.input_messages.1.message.role": "tool",
    "llm.input_messages.1.message.content": "{\"temperature\":72,\"condition\":\"sunny\"}",
    "llm.input_messages.1.message.tool_call_id": "call_abc123",
    "llm.input_messages.1.message.name": "get_weather",

    // Our canonical projections (always present when we were able to
    // extract them — router output).
    "inference.export.schema_version": 1,
    "inference.project_id": "prj_...",
    "inference.observation_kind": "LLM",
    "inference.llm.provider": "anthropic",
    "inference.llm.model_name": "claude-sonnet-4-5",
    "inference.llm.input_tokens": 123,
    "inference.llm.output_tokens": 456,
    "inference.llm.cost.total": "0.0123400000",
    "inference.user_id": null,
    "inference.session_id": null,
    "inference.agent_name": ""
  }
}
```

No descriptors. No trace format adapters. No Claude Code / inference.net conversion layer.

### Trace Schema Notes

Each JSONL row is one span. `trace_id` groups spans into a trace, `span_id` identifies the span, and `parent_span_id` preserves the span tree. `attributes` preserves upstream OTel/OpenInference fields verbatim plus the `inference.*` canonical projections we can reliably extract.

The trace models should keep this shape strongly typed without hiding the raw attributes. `SpanRecord` should model the top-level OTel fields, while `attributes` and `resource.attributes` remain typed JSON maps so tools can inspect original exported values when needed.

The canonical `inference.*` attributes are the fields the index should prefer for summaries and filters. Original `llm.*` and OpenInference attributes are still available through `view_trace`, `search_trace`, and `render_trace`.

## Trace Architecture

Trace functionality is split into two responsibilities.

`TraceIndexBuilder` owns build-once index creation. It lives in `engine/traces/trace_index_builder.py`.

`TraceStore` owns pure query/read/render behavior over an existing trace file and index file. It lives in `engine/traces/trace_store.py`.

This split matters because `run_code` must be able to import and initialize `TraceStore` inside the sandbox without pulling in OpenAI SDK, tool code, index-building dependencies, or app/runtime code.

### TraceIndexBuilder

```python
class TraceIndexBuilder:
    @classmethod
    async def ensure_index_exists(
        cls,
        trace_path: Path,
        config: TraceIndexConfig,
    ) -> Path:
        ...

    @classmethod
    async def build_index(
        cls,
        trace_path: Path,
        index_path: Path,
        meta_path: Path,
        schema_version: int,
    ) -> None:
        ...

```

Indexing does:

- Scan flat span JSONL once.
- Group spans by `trace_id`.
- Store byte offsets and byte lengths.
- Store slim summary fields for query/count/overview.
- Write sidecar index and meta files.

Default paths:

- Index: `<trace_path>.engine-index.jsonl`
- Metadata: `<trace_path>.engine-index.meta.json`

Index files are build-once. `ensure_index_exists` only checks whether the index path exists. Existing indexes are loaded as-is. There is no freshness check.

Index writing should be atomic: write temporary index/meta files first, then rename them into place. Read the trace file in binary mode so `byte_offset` and `byte_length` are exact for later `TraceStore` reads.

If an existing index has an unsupported schema version, fail fast instead of rebuilding implicitly.

### TraceStore

`TraceStore` is pure and importable in the sandbox.

```python
class TraceStore:
    def __init__(self, trace_path: Path, index_path: Path) -> None:
        ...

    @classmethod
    def load(cls, trace_path: Path, index_path: Path) -> "TraceStore":
        ...

    def get_overview(self, filters) -> DatasetOverview:
        ...

    def query_traces(self, query) -> TraceQueryResult:
        ...

    def count_traces(self, filters) -> TraceCountResult:
        ...

    def view_trace(self, trace_id: str) -> TraceView:
        ...

    def search_trace(self, trace_id: str, pattern: str) -> TraceSearchResult:
        ...

    def render_trace(self, trace_id: str, budget) -> str:
        ...
```

`view_trace` returns a strongly typed trace object for a trace id. `render_trace` is the prompt/tool-facing text representation of that same trace. Rendering should stay as a `TraceStore` method or a private helper used by `TraceStore`, not as a separate public module.

`TraceStore` should depend only on:

- Python stdlib.
- Pydantic.
- The local `engine.traces.models.*` models.

It should not import OpenAI Agents SDK, tools, sandbox runner, app code, or async runtime code.

## run_code Tool

`run_code` gets read-only access to:

- Trace file.
- Index file.
- Minimal importable Engine trace modules.

It gets writable access to:

- A temp working directory.

The sandbox bootstrap script should make this easy:

```python
from pathlib import Path
from engine.traces.trace_store import TraceStore

trace_store = TraceStore.load(
    trace_path=Path("/mnt/trace/traces.jsonl"),
    index_path=Path("/mnt/trace/traces.jsonl.engine-index.jsonl"),
)

# numpy and pandas are available for analysis code.
import numpy as np
import pandas as pd

# User code can call:
# trace_store.get_overview(...)
# trace_store.query_traces(...)
# trace_store.count_traces(...)
# trace_store.view_trace(...)
# trace_store.search_trace(...)
# trace_store.render_trace(...)
```

Initial sandbox requirements:

- Read-only trace file.
- Read-only index file.
- Read-only access to the minimal importable `engine.traces` modules, either through installed package files or a read-only source mount.
- Writable temp dir.
- No network.
- Timeout.
- Stdout/stderr caps.
- Environment scrubbing: do not inherit host environment variables.
- Read-only Python environment with `numpy`, `pandas`, and importable `TraceStore` modules.
- Linux uses `bubblewrap`.
- macOS uses `sandbox-exec`.

The sandboxed code should only be able to read the trace JSONL file, the sidecar index file, a read-only Python environment, and the minimal Python/runtime files needed to import `engine.traces`, `numpy`, and `pandas`. It should only be able to write inside its temporary working directory. Do not bind the project root, home directory, or filesystem root into the sandbox.

The sandbox Python environment should have `numpy` and `pandas` available. The bootstrap script should initialize and expose a `trace_store` variable so user code can inspect traces through the pure `TraceStore` API without manually loading files.

Basic sandbox models:

```python
class SandboxConfig(BaseModel):
    timeout_seconds: float = 10.0
    maximum_stdout_bytes: int = 64_000
    maximum_stderr_bytes: int = 64_000
    python_executable: Path | None = None


class SandboxPolicy(BaseModel):
    readonly_paths: list[Path]
    writable_paths: list[Path]
    network_enabled: Literal[False] = False
    timeout_seconds: float


class CodeExecutionResult(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
```

Execution flow:

1. `run_code_tool` receives typed `RunCodeArguments`.
2. `SandboxRunner` creates a temporary working directory.
3. `sandbox_bootstrap` writes a wrapper script that initializes `TraceStore`.
4. `platform_commands` builds the Linux or macOS sandbox command.
5. `SandboxRunner` starts the process in its own session/process group.
6. `sandbox_results` captures exit code, timeout, stdout, and stderr with configured output caps while reading output.
7. On timeout, `SandboxRunner` kills the full process group, not just the immediate child process.

### Sandbox Package Components

`engine.sandbox.sandbox_runner` owns the high-level execution flow:

```python
class SandboxRunner:
    async def run_python(
        self,
        code: str,
        trace_path: Path,
        index_path: Path,
        config: SandboxConfig,
    ) -> CodeExecutionResult:
        ...
```

`engine.sandbox.platform_commands` builds platform-specific commands with simple functions, not a backend abstraction:

```python
def build_linux_bubblewrap_command(
    policy: SandboxPolicy,
    script_path: Path,
) -> list[str]:
    ...


def build_macos_sandbox_exec_command(
    policy: SandboxPolicy,
    script_path: Path,
) -> list[str]:
    ...
```

`SandboxRunner` selects the right function based on `platform.system()`.

Linux command construction should use `bubblewrap` with read-only binds for the trace/index inputs, a read-only bind for the Python environment, and a writable bind for the temp directory. The exact Python runtime mounts may vary by environment, but the policy shape should stay narrow:

```text
bwrap
  --die-with-parent
  --new-session
  --unshare-all
  --unshare-net
  --clearenv
  --ro-bind <trace_path> /mnt/trace/traces.jsonl
  --ro-bind <index_path> /mnt/trace/traces.jsonl.engine-index.jsonl
  --ro-bind <python_environment_path> /venv
  --ro-bind <engine_traces_package_path> /opt/engine/traces
  --bind <temporary_work_dir> /workspace
  --setenv PATH /venv/bin:/usr/bin:/bin
  --setenv HOME /workspace
  --setenv LANG C.UTF-8
  --setenv PYTHONDONTWRITEBYTECODE 1
  --setenv PYTHONUNBUFFERED 1
  --setenv TMPDIR /workspace/tmp
  --chdir /workspace
  --proc /proc
  --dev /dev
  -- /venv/bin/python /workspace/bootstrap.py
```

`--unshare-net` is the explicit Linux network isolation. `--clearenv` prevents host secrets such as cloud credentials, API keys, and local configuration paths from leaking into the sandbox.

macOS command construction should use `sandbox-exec` with a generated profile that denies by default, allows read access only to the trace JSONL, index file, read-only Python environment, and required Python/package paths, allows writes only under the temp directory, and denies network access. Use `env -i` to scrub the inherited host environment:

```text
sandbox-exec -f <profile_path>
  env -i
  PATH=<python_environment_path>/bin:/usr/bin:/bin
  HOME=<temporary_work_dir>
  LANG=C.UTF-8
  PYTHONDONTWRITEBYTECODE=1
  PYTHONUNBUFFERED=1
  TMPDIR=<temporary_work_dir>/tmp
  <python_environment_path>/bin/python <temporary_work_dir>/bootstrap.py
```

The profile should follow this shape:

```scheme
(version 1)
(deny default)
(allow process*)
(deny network*)
(allow file-read* (literal "<trace_path>"))
(allow file-read* (literal "<index_path>"))
(allow file-read* (subpath "<python_environment_path>"))
(allow file-read* (subpath "<engine_traces_package_path>"))
(allow file-read* (subpath "<python_runtime_path>"))
(allow file-write* (subpath "<temporary_work_dir>"))
```

Sandbox tests should include denied-operation cases for filesystem writes outside the temp directory, reads outside the allowed paths, and network access. These can be normal unit/integration tests rather than startup checks.

`engine.sandbox.sandbox_bootstrap` owns the generated wrapper script that imports `TraceStore`, initializes it with the mounted trace/index paths, then runs user code.

`engine.sandbox.sandbox_results` owns stdout/stderr truncation, timeout reporting, exit code capture, and result serialization.

## AgentContext

Each agent has its own `AgentContext`.

Parent and child contexts are separate.

```python

class AgentToolFunction(BaseModel):
    name: str
    arguments: str


class AgentToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: AgentToolFunction


class AgentMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: MessageContent | None = None
    tool_calls: list[AgentToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

class AgentContextItem(BaseModel):
    item_id: str
    role: Literal["system", "user", "assistant", "tool"]
    content: MessageContent | None = None
    tool_calls: list[AgentToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    is_compacted: bool = False
    compaction_summary: str | None = None
    agent_id: str | None = None
    parent_agent_id: str | None = None
    parent_tool_call_id: str | None = None


class AgentContext:
    items: list[AgentContextItem]
    compaction_model: ModelConfig
    text_message_compaction_keep_last_messages: int
    tool_call_compaction_keep_last_messages: int

    def __init__(
        self,
        items: list[AgentContextItem],
        compaction_model: ModelConfig,
        text_message_compaction_keep_last_messages: int,
        tool_call_compaction_keep_last_messages: int,
    ) -> None:
        ...

    def append(self, item: AgentContextItem) -> None:
        ...

    def compact_old_items(self) -> None:
        ...

    '''
    Converts items into LLM-compatible messages array format
    '''
    def to_messages_array(self) -> list[AgentMessage]:
        ...

    def get_item(self, item_id: str) -> AgentContextItem:
        ...
```

Compaction uses two independent keep-last thresholds:

- `text_message_compaction_keep_last_messages`: how many recent non-tool-call text messages remain uncompacted.
- `tool_call_compaction_keep_last_messages`: how many recent tool-call-related messages remain uncompacted. Tool-call-related means assistant messages with non-empty `tool_calls` and `tool` role result messages.

System messages are never compacted.

`AgentContextItem` is the stored representation. `AgentMessage` is the OpenAI/HF-compatible message sent to the model. Available tool definitions are not stored on messages; they live on the OpenAI Agents SDK `Agent`.

Compaction never removes an item from `AgentContext.items` and does not strip the original fields. It sets `is_compacted=True` and writes a `compaction_summary`, while preserving the original `content`, `tool_calls`, `tool_call_id`, role, name, and lineage metadata on the item.

`get_item(item_id)` returns the full stored `AgentContextItem`, including original content/tool-call fields and any compaction summary.

`to_messages_array` converts stored items into provider-compatible messages. If `is_compacted=False`, the item maps directly to `AgentMessage` using its original fields. If `is_compacted=True`, the model-facing message uses `compaction_summary` instead of the original content/tool calls.

The compacted summary string should include any tool names, arguments, or result details that matter for future reasoning. The original structured fields remain available on the `AgentContextItem` for context tools and debugging.

Message-array validity matters for tool calls. A `role="tool"` message is only valid when the message array also contains the matching assistant message with a real `tool_calls` entry for the same `tool_call_id`. Compacted assistant tool-call messages render as assistant summary messages with no `tool_calls`, so compacted tool results should also render as assistant summary messages instead of `tool` messages.

### AgentContext Examples

Original user text item:

```python
context_item = AgentContextItem(
    item_id="msg_1",
    role="user",
    content="Find failing traces.",
    tool_call_id=None,
)

message = AgentMessage(
    role="user",
    content="Find failing traces.",
)
```

Original assistant tool-call item:

```python
context_item = AgentContextItem(
    item_id="msg_2",
    role="assistant",
    content=None,
    tool_calls=[
        AgentToolCall(
            id="call_1",
            function=AgentToolFunction(
                name="query_traces",
                arguments='{"filters":{"has_errors":true}}',
            ),
        )
    ],
    tool_call_id=None,
)

message = AgentMessage(
    role="assistant",
    content=None,
    tool_calls=context_item.tool_calls,
)
```

Original tool result item:

```python
context_item = AgentContextItem(
    item_id="msg_3",
    role="tool",
    content='{"trace_ids":["t1","t2","t3"]}',
    tool_calls=None,
    tool_call_id="call_1",
    name="query_traces",
)

message = AgentMessage(
    role="tool",
    content='{"trace_ids":["t1","t2","t3"]}',
    tool_call_id="call_1",
    name="query_traces",
)
```

Compacted text item:

```python
context_item = AgentContextItem(
    item_id="msg_1",
    role="user",
    content="Find failing traces.",
    tool_call_id=None,
    is_compacted=True,
    compaction_summary="User asked to find failing traces.",
)

message = AgentMessage(
    role="user",
    content="Compacted message (id: msg_1): User asked to find failing traces.",
)
```

Compacted assistant tool-call item:

```python
context_item = AgentContextItem(
    item_id="msg_2",
    role="assistant",
    content=None,
    tool_calls=[
        AgentToolCall(
            id="call_1",
            function=AgentToolFunction(
                name="query_traces",
                arguments='{"filters":{"has_errors":true}}',
            ),
        )
    ],
    tool_call_id=None,
    is_compacted=True,
    compaction_summary="Called query_traces with has_errors=true.",
)

message = AgentMessage(
    role="assistant",
    content=(
        "Compacted tool calls (id: msg_2): "
        "Called query_traces with has_errors=true."
    ),
)
```

Compacted tool result item whose matching tool call was also compacted:

```python
context_item = AgentContextItem(
    item_id="msg_3",
    role="tool",
    content='{"trace_ids":["t1","t2","t3"]}',
    tool_calls=None,
    tool_call_id="call_1",
    name="query_traces",
    is_compacted=True,
    compaction_summary="query_traces returned trace ids t1, t2, and t3.",
)

message = AgentMessage(
    role="assistant",
    content=(
        "Compacted tool result (id: msg_3, tool: query_traces): "
        "query_traces returned trace ids t1, t2, and t3."
    ),
)
```

### OpenAI Agents SDK Boundary

The OpenAI Agents SDK should not directly own or mutate `AgentContextItem`. `AgentContextItem` is the Engine's stored representation; `AgentMessage` is the OpenAI/HF-compatible message shape passed to the SDK.

Before a model run, `AgentContext.to_messages_array()` converts stored items into SDK-compatible messages:

```python
messages = agent_context.to_messages_array()
stream = Runner.run_streamed(sdk_agent, input=messages)
```

Available tools are configured on the SDK `Agent`, not on individual messages:

```python
sdk_agent = Agent(
    name=agent_config.name,
    instructions=agent_config.instructions,
    tools=[query_traces_tool, view_trace_tool, run_code_tool],
)
```

During execution, `OpenAiEventMapper` converts SDK stream events back into `AgentContextItem` and `AgentOutputItem`:

```python
context_item = openai_event_mapper.to_context_item(sdk_event, agent_execution)
agent_context.append(context_item)

await output_bus.emit(
    openai_event_mapper.to_output_item(context_item, agent_execution)
)
```

Assistant tool calls become one assistant `AgentContextItem` with `tool_calls`. Tool outputs become `role="tool"` `AgentContextItem`s with `tool_call_id` and `name`. If an item is compacted, `to_messages_array()` sends only the `compaction_summary` to the SDK while the full original fields remain available on the stored context item.

Subagents follow the same boundary with separate contexts. The parent context stores the parent assistant tool-call message and the final subagent tool result message. The child context stores the child execution messages, and child outputs stream to the caller through the shared `EngineOutputBus`.

## Engine Output Item

Keep the public output model small and lineage-rich. The item itself is always an OpenAI/HF-compatible message shape.

```python
class AgentOutputItem(BaseModel):
    sequence: int
    agent_id: str
    parent_agent_id: str | None
    parent_tool_call_id: str | None
    agent_name: str
    depth: int
    item: AgentMessage
    final: bool = False
```

Tool calls and tool results are represented inside `AgentMessage`, not as separate output payload types. This supports interleaved parallel child output while keeping grouping easy and preserving messages-array compatibility.

## Output Bus

Use one run-level async queue. Every agent writes directly to it.

```python
class EngineOutputBus:
    def __init__(self):
        self.queue: asyncio.Queue[AgentOutputItem | None]
        self.next_sequence: int

    async def emit(self, item: AgentOutputItem) -> AgentOutputItem:
        ...

    async def close(self) -> None:
        ...

    async def fail(self, error: BaseException) -> None:
        ...

    async def stream(self) -> AsyncIterator[AgentOutputItem]:
        ...
```

`emit` assigns sequence numbers under a lock, then puts the item on the queue. The bus does not own a full in-memory ledger.

The caller consumes `EngineOutputBus.stream()`.

Parent, child, and grandchild output all flows through the same queue.

`run_engine_async` gets the full ordered result by collecting items from `stream_engine_async`. Streaming callers get the same items directly. If a future API needs an internal replay buffer, add a separate collector around the stream instead of putting storage responsibility inside `EngineOutputBus`.

## RunState

```python
class EngineRunState:
    trace_store: TraceStore
    output_bus: EngineOutputBus
    config: EngineConfig
    executions_by_agent_id: dict[str, AgentExecution]
    executions_by_tool_call_id: dict[str, AgentExecution]
```

`EngineRunState` is shared across all agents in a single Engine run.

No semaphore. Parallel subagent limits are prompt-instructed through `maximum_parallel_subagents`.

## Tools

Use one simple tool interface.

```python
class EngineTool(Protocol):
    name: str
    description: str
    arguments_model: type[BaseModel]
    result_model: type[BaseModel]

    async def run(self, tool_context: ToolContext, arguments: BaseModel) -> BaseModel:
        ...
```

Tool groups:

```text
tools/trace_tools.py          # overview, query, count, view, search
tools/agent_context_tools.py  # get context item
tools/synthesis_tool.py       # summarize selected traces
tools/subagent_tool_factory.py # agent-as-tool wiring
tools/run_code_tool.py        # sandboxed python
```

Keep tool implementation files readable. Each tool should be a small class with a typed arguments model and typed result model.

Core tool functions should have explicit typed boundaries:

```python
async def get_dataset_overview(
    tool_context: ToolContext,
    arguments: DatasetOverviewArguments,
) -> DatasetOverviewResult: ...


async def query_traces(
    tool_context: ToolContext,
    arguments: QueryTracesArguments,
) -> QueryTracesResult: ...


async def count_traces(
    tool_context: ToolContext,
    arguments: CountTracesArguments,
) -> CountTracesResult: ...


async def view_trace(
    tool_context: ToolContext,
    arguments: ViewTraceArguments,
) -> ViewTraceResult: ...


async def search_trace(
    tool_context: ToolContext,
    arguments: SearchTraceArguments,
) -> SearchTraceResult: ...


async def get_context_item(
    tool_context: ToolContext,
    arguments: GetContextItemArguments,
) -> GetContextItemResult: ...


async def synthesize_traces(
    tool_context: ToolContext,
    arguments: SynthesizeTracesArguments,
) -> SynthesizeTracesResult: ...


async def run_code(
    tool_context: ToolContext,
    arguments: RunCodeArguments,
) -> CodeExecutionResult: ...
```

The exact argument/result fields can evolve during implementation, but every tool should expose Pydantic argument and result models. Tool functions should call `TraceStore`, `AgentContext`, or `SandboxRunner`; they should not parse raw dictionaries internally.

## Subagents

Use OpenAI Agents SDK `Agent.as_tool(...)`.

For each subagent tool:

- `on_stream` receives nested child stream events.
- `on_stream` normalizes child events into `AgentOutputItem`.
- `on_stream` emits those items to the shared `EngineOutputBus`.
- `custom_output_extractor` receives the completed child `RunResult`.
- The extractor builds a typed `SubagentToolResult` and serializes it only at the provider tool-message boundary.
- The parent stores only that concise tool result in its own `AgentContext` as a standard `role="tool"` message.

Parent context gets a tool result message whose `content` is the serialized form of this typed payload:

```python
class SubagentToolResult(BaseModel):
    child_agent_id: str
    answer: str
    output_start_sequence: int
    output_end_sequence: int
    turns_used: int
    tool_calls_made: int
```

Caller stream gets the full child execution.

## Parallel Subagents

With SDK `parallel_tool_calls=True`, the parent may call multiple subagent tools concurrently.

Expected stream can interleave:

```text
0 parent assistant message with tool_calls=[subagent_A, subagent_B]
1 child_A assistant message with tool_calls=[query_traces]
2 child_B assistant message with tool_calls=[dataset_overview]
3 child_B tool message for dataset_overview
4 child_A tool message for query_traces
5 child_B final assistant message
6 parent tool message for subagent_B
7 child_A final assistant message
8 parent tool message for subagent_A
9 parent final assistant message
```

This is fine because every item carries:

```text
agent_id
parent_agent_id
parent_tool_call_id
depth
sequence
```

## Error Handling

Root run failures should fail the stream. `stream_engine_async` starts the root agent in a background task and must call `EngineOutputBus.fail(error)` if that task raises, then re-raise the error to the stream consumer.

Tool failures should be represented consistently at the tool boundary. If the SDK converts tool exceptions into tool result messages, use that behavior. Otherwise, catch tool exceptions in the tool adapter and return a typed failed tool result message so the parent model can continue when appropriate.

Subagent failures should return typed failure information to the parent tool call when recovery is reasonable. Fatal runtime errors, cancellation, and invalid SDK state should fail the stream rather than being hidden inside a summary string.

`EngineOutputBus.close()` is only for successful completion. `EngineOutputBus.fail(error)` closes the queue and stores the error so `stream()` raises after yielding already-emitted items.

`run_engine_async` should not have separate error behavior. It collects `stream_engine_async`, so it returns collected items on success and raises the same stream error on failure.

## Path Of Execution

1. Caller calls `stream_engine_async(messages, engine_config, trace_path)`.
2. `TraceIndexBuilder.ensure_index_exists(...)` creates the sidecar index if missing, otherwise reuses it as-is.
3. `TraceStore.load(trace_path, index_path)` creates the pure trace query/read object.
4. `EngineRunState` and `EngineOutputBus` are created.
5. Root `AgentContext` is initialized from input messages.
6. Root SDK `Agent` is built with trace, context, synthesis, subagent, and code tools.
7. Root agent starts in a background task.
8. `stream_engine_async` yields from `EngineOutputBus.stream()`.
9. Root SDK events emit parent output items.
10. If root calls subagents, child SDK stream events emit child output items into the same bus.
11. Child final result returns to parent as a normal tool result.
12. Parent continues.
13. Root final response is emitted with `final=True`.
14. Bus closes.
15. `run_engine_async` returns the collected stream items.

## Implementation Phases

1. Add package skeleton and key Pydantic runtime models.
2. Implement `TraceIndexBuilder`.
3. Implement pure `TraceStore`.
4. Implement trace tools against `TraceStore`.
5. Implement `AgentContext` and context tools.
6. Implement `EngineOutputBus`, `EngineRunState`, and SDK event normalization.
7. Implement root agent runner with OpenAI Agents SDK.
8. Implement subagents through `Agent.as_tool(...)`, `on_stream`, and `custom_output_extractor`.
9. Implement minimal sandbox runner and `run_code` tool with `TraceStore` bootstrap.
10. Rewrite tests around the new `engine/` package only.
