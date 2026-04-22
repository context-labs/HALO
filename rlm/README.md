# HALO RLM

The HALO Recursive Language Model harness — an LLM-driven agent for
exploring large agent-trace datasets.

Point it at a JSONL of OTLP span trees (local path, or the file that
[`otel-interceptor`](https://github.com/context-labs/otel-interceptor) or
the HALO hosted ingest dumps), register it with a ~30-line *descriptor*,
and ask natural-language questions. The agent plans tool calls, drills
into specific traces, runs sandboxed Python against the index, and
synthesizes answers grounded in the data.

**Supported trace formats** — auto-detected on ingest:

| Format | When it's produced | Detected via |
|---|---|---|
| **OpenInference** | OpenAI Agents SDK, Anthropic, custom framework with [OpenInference](https://github.com/Arize-ai/openinference) conventions | `openinference.span.kind` attribute on spans |
| **Claude Code** | Claude Agent SDK (Claude Code CLI's native OTel) | span names under the `claude_code.*` namespace |
| **HF** | Flat JSONL (HuggingFace-style record-per-line) — no OTel | Fallback when no OTLP span tree is present |

`halo ingest` sniffs the first parseable record and picks the right
mapping automatically. No flags required.

Answers stream token-by-token. Conversations are multi-turn with
disk-persisted state and context compaction so long sessions don't blow
past the model's window. Multiple runs can stream concurrently in the UI.

```
                    ┌───────────────────────────┐
  traces.jsonl ───▶ │  summary index  (RAM)     │
                    │  + random-access reader   │
                    └────────────┬──────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │  LLM analyst (halo harness)     │
                │  tools: overview, find, inspect,│
                │         synthesize, compare…    │
                └────────────────┬────────────────┘
                                 │  SSE
                          ┌──────┴──────┐
                          │ React UI    │
                          └─────────────┘
```

## Quick start

```bash
cd rlm
uv sync                      # Python deps (creates .venv)
cp .env.example .env         # add LITELLM_API_KEY + auth creds
```

### 1. Point it at a JSONL

```bash
uv run halo ingest /path/to/traces.jsonl
```

One command. `ingest` peeks at the first ~200 records, infers the field
mappings (id, query, messages, outcome score, ground truth, documents,
categorical labels), writes a descriptor to `catalog/<id>.py`, and
builds the summary index. It prints a table of every mapping it picked
so you can spot-check the inference.

Edit the generated descriptor file afterwards to add seed questions or
fix any mis-guessed field (then `uv run halo index --dataset <id>` to
rebuild the index).

Alternative (manual): drop a descriptor file into `catalog/` yourself
(see `catalog/_template.py.example` and `catalog/README.md`), then
`uv run halo index --dataset <id>`.

### 2. Verify it registered

```bash
uv run halo datasets     # lists everything auto-discovered
```

The indexer streams the source file once and writes a compact summary
JSONL (tens of MB for a million traces). The raw bodies stay on disk
and are fetched via byte-offset seek on demand, so you can interrogate
300 GB datasets without loading them into memory.

### 3. Ask from the CLI

```bash
uv run halo ask "Where does this agent fail, and why?" --dataset <id>
```

### 4. Run the UI

The React UI's static assets ship pre-built under `web/static/`, so
`halo serve` works out of the box:

```bash
uv run halo serve            # http://localhost:8000
```

Set `HALO_AUTH_USER` / `HALO_AUTH_PASS` in `.env` to gate the UI with
HTTP Basic Auth.

### Tests

```bash
uv sync --group dev          # adds pytest + datasets
uv run pytest tests/         # 26 tests; covers indexer, autodetect,
                             # compactor, sandbox, both OpenInference
                             # and Claude Code views.
```

## Hosted endpoint

The same harness is productionized as a Modal HTTP endpoint so you can
call it over HTTPS without running anything locally. Input: an S3 /
R2 / HTTPS URL to a JSONL of OTLP span trees (OpenInference or Claude
Code native — auto-detected per request) + an OpenAI-format `messages`
list. Output: the full agent run — events, tool calls, tool results,
final assistant message, cost. The endpoint also exposes an
OpenAI-compatible `/v1/chat/completions` shim so any OpenAI SDK can
talk to it.

See the Modal deployment repo for the endpoint URL and token, or use
the OpenAI SDK pattern:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<workspace>--halo-rlm-prod-rlm-http-api.modal.run/v1",
    api_key="<HALO_RLM_TOKEN>",
)
resp = client.chat.completions.create(
    model="rlm:s3://your-bucket/traces.jsonl",
    messages=[{"role": "user", "content": "Where does the agent fail?"}],
)
print(resp.choices[0].message.content)
```

## Architecture

| Module | Responsibility |
|---|---|
| `dataset/descriptor.py` | `DatasetDescriptor` — central abstraction for describing any trace JSONL; holds a `HFMapping` / `OpenInferenceMapping` / `ClaudeCodeMapping`. |
| `dataset/formats/openinference.py` | View over OpenInference-shaped OTLP span trees — `llm.input_messages.*`, `tool.name`, `openinference.span.kind`. |
| `dataset/formats/claude_code.py` | View over Claude Code native OTel (`claude_code.interaction` / `llm_request` / `tool` spans + `tool.output` span events). |
| `dataset/autodetect.py` | `halo ingest` sniff path — picks HF / OpenInference / Claude Code by looking at the first parseable record. |
| `dataset/indexer.py` | Descriptor-driven one-pass scan; emits a `TraceSummary` per line with byte offsets. |
| `dataset/reader.py` | Random-access fetch of one full record by byte offset. |
| `dataset/store.py` | In-memory store + filter + `overview()` aggregates. |
| `inference/tools.py` | Tool implementations + OpenAI schemas (10 tools including `synthesize`, `run_code`, `inspect_result`). |
| `inference/sandbox.py` | Subprocess Python sandbox behind the `run_code` tool. |
| `inference/harness.py` | Streaming agent loop; yields per-token `delta` events plus `tool_call` / `final` / `usage`. |
| `catalog/` | Registered datasets — one Python module per dataset. |
| `registry.py` | Auto-discovery of catalog modules with `DESCRIPTOR` attributes. |
| `web/server.py` | FastAPI + SSE bridge between agent events and the React UI. |
| `web/conversations.py` | Disk-persisted multi-turn conversation store (messages + result-key map). |
| `web/history.py` | Append-only run history for the UI's "past questions" panel. |
| `utils/` | Inlined LLM client (LiteLLM proxy) + `Hypers` config system. |

## The RLM primitives

Two tools in the analyst's kit are borrowed from the [RLM](https://github.com/alexzhang13/rlm)
(Recursive Language Models) design and load-bearing for population-scale
reasoning:

- **`synthesize`** — pass 10–25 trace ids plus a sub-question; a secondary
  LLM (the *rlm sub-agent*, configurable per-run) reads compact
  renderings of those traces and answers in one paragraph. The analyst
  never has to stuff 25 full traces into its own context window.
- **`run_code`** — sandboxed Python REPL with `store`, `reader`, and
  `descriptor` pre-bound. The analyst writes ad-hoc `store.filter(...)`
  aggregations, custom groupings, multi-pass filters — anything the
  eight fixed tools don't express. Runs in a subprocess with
  `cwd=/tmp`, a 10 s timeout, and a 100 KB stdout cap.

Plus, to keep long conversations affordable:

- **Context compaction** — after each turn, older tool messages collapse
  to a one-line summary `[compacted · r_N] <tool> → <summary>`. The
  full JSON stays in a per-conversation result store.
- **`inspect_result(key)`** — fetch back the full JSON of any earlier,
  compacted tool call when its details matter to the current step.

## Streaming

The top-level LLM call streams via `stream=True`. The harness coalesces
fast-arriving tokens into ~30 ms windows (cuts SSE event count ~20× for
fast generators without losing content), and the frontend batches rAF
flushes to cap re-renders at 60 fps. The Markdown renderer is memoized
so already-rendered earlier turns don't re-parse while a later turn
streams.

Cost is computed per turn from LiteLLM's `/model/info` pricing table —
streaming responses don't carry the `x-litellm-response-cost` header
that the non-streaming path uses.

## Concurrent runs

The UI can drive several runs in parallel against the same dataset.
Each run has its own SSE connection, epoch counter, and row in the
**Progressing** sidebar section; switching focus between them never
cancels a stream in flight. Follow-up questions continue the focused
conversation; clicking a **history** entry loads its saved messages
and auto-switches to the dataset it belongs to.

## License

MIT.
</content>
</invoke>
