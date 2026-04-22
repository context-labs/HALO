# utils.llm

Thin wrapper over LiteLLM proxy for cloud models, with direct OpenAI-compatible client for local models.

## Quick start

```python
from utils.llm import complete, Message

messages: list[Message] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]
result = complete("gpt-5.4", messages)

print(result.content)    # "4"
print(result.cost)       # 0.00001 (from LiteLLM proxy header)
print(result.tokens)     # {"input": 25, "output": 3, "thinking": 0, "total": 28}
```

## How it works

```
complete("model-name", messages)
    |
    local=False (default)  -->  LiteLLM proxy (litellm.inference.cool/v1)
    local=True             -->  Direct OpenAI client, round-robin across endpoints
```

Cloud models go through the LiteLLM proxy, which handles provider routing, retries, rate limits, and fallbacks server-side. Local models hit your endpoints directly with client-side retry.

## Local models

```python
result = complete(
    "openai/my-fine-tune",
    messages,
    local=True,
    endpoints=[
        "http://10.0.1.28:8011/v1",
        "http://10.0.1.28:8012/v1",
        "http://10.0.1.28:8013/v1",
    ],
)
```

Endpoints are round-robined across calls. Retries with exponential backoff on failure.

## Tool calling

```python
result = complete("gpt-5.4", messages=[...], tools=[{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}])

if result.tool_calls:
    for tc in result.tool_calls:
        print(tc["function"]["name"], tc["function"]["arguments"])
```

## Agent loops

`result.message` is the full assistant message dict in OpenAI format, ready to append back to the conversation:

```python
messages = [
    {"role": "system", "content": "You have access to tools."},
    {"role": "user", "content": "Find the weather in SF"},
]

for turn in range(10):
    result = complete("gpt-5.4", messages, tools=my_tools)
    messages.append(result.message)

    if not result.tool_calls:
        break

    for tc in result.tool_calls:
        output = execute_tool(tc["function"]["name"], tc["function"]["arguments"])
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": output})
```

## Batch integration

```python
from utils.llm import complete
from utils.batch import run_batch, GracefulShutdown

def process(item):
    return complete("gpt-5.4", messages=[
        {"role": "user", "content": item["text"]},
    ])

with GracefulShutdown() as shutdown:
    tracker = run_batch(
        items=samples,
        process_fn=process,
        cost_fn=lambda r: r.cost,
        error_fn=lambda r: r.error is not None,
        shutdown=shutdown,
    )
```

## Error handling

**Infrastructure errors** (bad API key, model not found, endpoint unreachable) raise immediately. These indicate broken config — no point continuing a batch.

**Per-request errors** (content filtered, context length exceeded) are captured in `result.error` so batch runs can skip and continue.

```python
result = complete("gpt-5.4", messages=[...])
if result.error:
    print(f"Skipped: {result.error}")
else:
    print(result.content)
```

## Environment variables

- `LITELLM_API_KEY` — required for cloud models (get a virtual key from the LiteLLM admin UI)
- `LOCAL_API_KEY` — for local model endpoints (defaults to "no-key" if not set)
- `LITELLM_BASE_URL` — override proxy URL (default: `https://litellm.inference.cool/v1`)

## CompletionResult

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
