# utils

Reusable utilities shared across all projects in this workspace: batch processing and LLM completion.

## Install

All workspace projects get `utils` automatically. When you create a project, add `utils` as a workspace dependency:

```toml
# projects/<name>/pyproject.toml
[tool.uv.sources]
utils = { workspace = true }
```

Then `uv sync` makes it importable.

Outside the workspace (or without uv), install directly:

```bash
pip install -e lib/utils/
```

Either way, imports look the same:

```python
from utils.batch import run_batch
from utils.llm import complete, CompletionResult, Message
from utils.hypers import Hypers, TBD
```

---

## `utils.batch`

Concurrent batch processing with a rich progress bar, cost tracking, rate limiting, and graceful shutdown.

### `run_batch(items, process_fn, *, workers, max_rps, ...)`

Main entry point. Runs `process_fn` over `items` with optional parallelism, rate limiting, and checkpointing.

```python
from utils.batch import run_batch

tracker = run_batch(
    items=my_items,
    process_fn=process_one,
    workers=8,
    max_rps=5.0,
    desc="Processing",
    cost_fn=lambda r: r.cost,
    error_fn=lambda r: r.error is not None,
    on_checkpoint=save_results,
    checkpoint_every=50,
)

print(tracker.total_cost, tracker.samples_processed, tracker.samples_failed)
```

| Parameter | Description |
|---|---|
| `items` | Any iterable of items to process |
| `process_fn` | `item -> result` (called once per item) |
| `workers` | Concurrency level (1 = sequential) |
| `max_rps` | Rate limit in requests/sec (`None` = unlimited) |
| `cost_fn` | `result -> float` to extract cost (`None` = no cost tracking) |
| `error_fn` | `result -> bool` to flag failures (`None` = all successes) |
| `on_result` | `(item, result) -> None` after each completion |
| `on_checkpoint` | `list[(item, result)] -> None` for batched disk I/O |
| `checkpoint_every` | Buffer size before flushing to `on_checkpoint` |
| `shutdown` | `GracefulShutdown` instance for SIGINT handling |

Returns a `ProgressTracker` with `samples_processed`, `samples_failed`, `total_cost`, `avg_cost_per_sample`, and `estimate_total_cost(n)`.

### `ProgressTracker`

Thread-safe accumulator for success/failure counts and cost. Serialisable via `to_dict()` / `from_dict()` for checkpointing.

```python
from utils.batch import ProgressTracker

tracker = ProgressTracker()
tracker.record_success(cost=0.001)
tracker.record_failure(cost=0.0005)

print(tracker.samples_processed)    # 1
print(tracker.samples_failed)       # 1
print(tracker.total_cost)           # 0.0015
print(tracker.avg_cost_per_sample)  # 0.00075

# Serialize / restore
data = tracker.to_dict()
restored = ProgressTracker.from_dict(data)
```

### `RateLimiter(max_rps)`

Thread-safe fixed-interval rate limiter. Call `acquire()` before each request.

```python
from utils.batch import RateLimiter

limiter = RateLimiter(max_rps=10.0)
limiter.acquire()  # blocks until the interval has elapsed
```

### `GracefulShutdown`

Context manager that intercepts SIGINT/SIGTERM and sets a flag instead of raising.

```python
from utils.batch import GracefulShutdown

with GracefulShutdown() as shutdown:
    while not shutdown.is_set():
        do_work()
```

---

## `utils.llm`

Thin wrapper over LiteLLM proxy for cloud models, with direct OpenAI-compatible client for local models. Cost tracked automatically from proxy response headers.

### `complete(model, messages, *, local, endpoints, tools, ...)`

Send a chat completion request and return a structured `CompletionResult`.

```python
from utils.llm import complete

result = complete("gpt-5.4", [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "What is 2+2?"},
])

print(result.content)  # "4"
print(result.cost)     # 0.000012
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | Model identifier (e.g. "gpt-5.4") |
| `messages` | `list[Message]` | required | Conversation history in OpenAI format |
| `local` | `bool` | `False` | Route to local endpoints instead of LiteLLM proxy |
| `endpoints` | `list[str] \| None` | `None` | Base URLs for local servers (required when `local=True`) |
| `tools` | `list[dict] \| None` | `None` | Tool definitions in OpenAI function-calling format |
| `tool_choice` | `str \| None` | `None` | Tool selection strategy ("auto", "required", etc.) |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `max_tokens` | `int` | `8192` | Maximum tokens in the response |
| `response_format` | `dict \| None` | `None` | Response format constraint |
| `max_retries` | `int` | `3` | Retry attempts for local model calls |
| `on_complete` | `Callable \| None` | `None` | Callback invoked with the result |

**Returns:** `CompletionResult`

### `CompletionResult`

```python
@dataclass
class CompletionResult:
    content: str | None       # text output
    tool_calls: list[dict] | None  # tool calls (if any)
    message: dict             # full assistant message (OpenAI format)
    tokens: dict              # {input, output, thinking, total}
    cost: float               # from LiteLLM header, $0 for local
    latency: float            # wall-clock seconds
    model: str                # model name
    error: str | None         # per-request error (None = success)
```

### `list_models(*, filter=None)`

Query the LiteLLM proxy `/v1/models` endpoint and return a sorted list of model ID strings. Optional substring filter (e.g. `filter="claude"`).

### `print_models(*, filter=None)`

Query and pretty-print available models from the LiteLLM proxy in a Rich table.

### Error handling

**Infrastructure errors** (bad API key, model not found, endpoint unreachable) raise immediately.

**Per-request errors** (content filtered, context length exceeded) are captured in `result.error` so batch runs can skip and continue.

### Environment variables

- `LITELLM_API_KEY` -- required for cloud models
- `LOCAL_API_KEY` -- for local model endpoints (defaults to "no-key")
- `LITELLM_BASE_URL` -- override proxy URL (default: `https://litellm.inference.cool/v1`)

See [utils/llm/README.md](llm/README.md) for full documentation: local models, tool calling, agent loops, error handling.

---

## `utils.hypers`

Dataclass-based configuration with layered overrides: **defaults -> config file -> CLI args**. Define your config once as a dataclass, and every field automatically becomes a CLI argument.

### Defining a config

Subclass `Hypers` and add typed fields with defaults:

```python
from dataclasses import dataclass
from utils.hypers import Hypers, TBD

@dataclass
class InferenceConfig(Hypers):
    model: str = "gpt-4o"
    temperature: float = 0.0
    workers: int = 8
    max_rps: float = 10.0
    output_path: str = "data/traces.jsonl"
```

Instantiate it and the layered override kicks in automatically:

```python
config = InferenceConfig()
print(config)  # prints a color-coded table
```

### Config files

Config files are plain Python with variable assignments. Any subset of fields can be overridden:

```python
# configs/fast.py
model = "gpt-4o-mini"
temperature = 0.2
workers = 16
```

Pass config files as positional arguments on the command line:

```bash
uv run my-project generate configs/fast.py
```

### CLI overrides

Every field becomes a `--field_name` CLI argument automatically. CLI args take highest priority:

```bash
# Override a single field
uv run my-project generate --temperature 0.5

# Config file + CLI override (CLI wins)
uv run my-project generate configs/fast.py --temperature 0.5
```

### `TBD()` for computed fields

Use `TBD()` for fields that are computed or set programmatically after init. These fields are excluded from CLI args and `__init__`:

```python
from dataclasses import dataclass
from pathlib import Path
from utils.hypers import Hypers, TBD

@dataclass
class InferenceConfig(Hypers):
    model: str = "gpt-4o"
    output_dir: str = "data"
    # Computed after init
    output_path: Path = TBD()

    def init(self):
        self.output_path = Path(self.output_dir) / "traces.jsonl"
```

`TBD(default)` accepts an optional default value (scalar or list).

### The `init()` pattern

After constructing the config, call a custom `init()` method (defined by you, not by Hypers) to resolve computed fields like paths:

```python
config = InferenceConfig()
config.init()
print(config.output_path)  # Path("data/traces.jsonl")
```

### Color-coded table output

Printing a config displays a Rich table showing each parameter's value and where it came from:

- **Blue**: default value
- **Magenta**: from a config file
- **Yellow**: from a CLI argument

```python
config = InferenceConfig()
print(config)
```

```
         Config
┏━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃ Parameter   ┃ Value  ┃ Source  ┃
┡━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│ model       │ gpt-4o │ default │
│ temperature │ 0.5    │ cli     │
│ workers     │ 16     │ config  │
└─────────────┴────────┴─────────┘
```

### API

| Method | Description |
|---|---|
| `get(name)` | Get a config value by field name |
| `set(name, val)` | Set a config value by field name |
| `to_dict()` | Return all config values as a dict |
| `update(dict)` | Update multiple config values from a dict |
