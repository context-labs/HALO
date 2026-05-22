# HALO for Care AI Langfuse Traces

Care AI already sends agent execution traces to Langfuse through the Care Agent SDK. HALO reads OTel-shaped JSONL, so the first integration layer is a converter from Care AI Langfuse exports into HALO's existing trace format.

## What The Converter Preserves

The `halo-careai convert-langfuse` helper maps:

| Care AI Langfuse field | HALO span field |
|---|---|
| Trace `id` | `attributes.langfuse.trace_id` and a stable OTel-style `trace_id` |
| Trace `sessionId` | `attributes.langfuse.session_id` and `attributes.care_ai.conversation_id` |
| `metadata.ucid` | `attributes.care_ai.ucid` |
| `metadata.sessionExecutorName` | `attributes.care_ai.session_executor_name` |
| `metadata.traceKind = agent_execution` | `inference.observation_kind = AGENT` |
| `agent_execution.output.toolCallCount` | `care_ai.agent_reported_tool_call_count` |
| Langfuse `GENERATION` / `traceKind = llm_generation` | `inference.observation_kind = LLM` |
| `metadata.traceKind = tool_call` or `agent_tool_invocation` | `inference.observation_kind = TOOL` |
| Span names `tool_call: ...`, `agent_tool: ...`, `transfer_to_*` | `inference.observation_kind = TOOL` even when `metadata.traceKind` is absent |
| `usage` / `usageDetails` | `inference.llm.input_tokens`, `inference.llm.output_tokens`, `llm.token_count.*` |
| Tool `input` / `output` | `input.value`, `output.value` |
| Tool metadata | `tool.name`, `care_ai.tool_call_id`, `care_ai.parent_agent`, `care_ai.target_agent` |
| `GENERATION.output` function calls | `care_ai.llm_planned_tool_names`, `care_ai.llm_planned_tool_count` |

Each Langfuse trace becomes one HALO trace. That matches Care AI's production shape: one trace per customer message, grouped into a conversation by `sessionId`.

Care AI emits a new trace for each customer message. `inspect`, `evidence-pack`, and the local pipeline therefore include both trace-level and session-level metrics: traces per session, spans per session, session input/output token distributions, session tool-call distributions, and the top sessions by input tokens. This lets HALO distinguish one expensive turn from a multi-turn conversation that is accumulating routing cost.

## Convert An Export

Export a trace plus observations from Langfuse into a JSON file shaped like:

```json
{
  "trace": { "id": "...", "sessionId": "...", "metadata": { "ucid": "..." } },
  "observations": []
}
```

Then convert it:

```bash
uv run halo-careai convert-langfuse ./langfuse-export.json ./care-ai-traces.jsonl
```

The converter also accepts:

- `{"traces": [...], "observations": [...]}`
- a plain observation list from `lf api observations list`
- JSONL with one observation per line
- a list of `{ "trace": ..., "observations": [...] }` bundles

## Sanitize For Metadata-Only Diagnosis

If a trace set has not been approved for raw transcript or tool-payload analysis, create a metadata-only copy before running HALO:

```bash
uv run halo-careai sanitize \
  .halo-careai/care-ai-traces.jsonl \
  .halo-careai/care-ai-traces.sanitized.jsonl
```

`sanitize` removes raw `input.value`, `output.value`, `llm.input_messages`, and `llm.output_messages` attributes. It also redacts non-empty `status.message` by default. Each removed value is replaced with a short SHA-256 fingerprint and byte count, so repeated-shape analysis still works without exposing payload text.

By default, `sanitize` also hashes trace, span, parent-span, Langfuse session/observation, UCID, request, tool-call, and user/customer identifiers while preserving joins within the file. Use `--keep-identifiers` only when those raw ids are explicitly approved for the model-backed analysis.

Use the sanitized file for structural questions:

```bash
uv run halo-careai diagnose .halo-careai/care-ai-traces.sanitized.jsonl \
  --focus routing \
  --executor orchestrator-test
```

Do not use sanitized traces for questions that require literal user text, tool payloads, model wording, or exact error strings.

### Pull With `lf`

For Langfuse data access, use the local `lf` CLI. Do not add Python or TypeScript API pull scripts.
The `.halo-careai/` directory is ignored by git so local trace exports do not become repo artifacts.

Generate a copy-paste batch recipe for the approved `lf` + `jq` workflow:

```bash
uv run halo-careai lf-batch-recipe .halo-careai/export/orchestrator-test \
  --environment test \
  --executor orchestrator-test \
  --limit 25 \
  --sample 12 \
  --focus cost
```

`lf-batch-recipe` does not call Langfuse. It prints shell that uses `lf api traces list`, `lf api traces get`, `lf api observations list`, `jq`, and the local `halo-careai` converter/sanitize/evidence/diagnose commands. Paste and run the recipe when you want to pull fresh traces. `convert-lf-pair` uses the separate observations payload when it is non-empty; if `lf api observations list --trace-id` returns an empty page but `lf api traces get` includes embedded observations, it falls back to the embedded observations so fresh traces do not collapse to root-only spans.

Single trace:

```bash
TRACE_ID="..."
mkdir -p .halo-careai/export

lf api traces get "$TRACE_ID" --json \
  > .halo-careai/export/trace.json

lf api observations list \
  --trace-id "$TRACE_ID" \
  --fields "core,basic,time,io,metadata,model,usage,prompt,metrics" \
  --expand-metadata "traceKind,toolName,toolCallId,parentAgent,targetAgent,targetAgentDisplayName,conversationId,ucid,sessionExecutorName,conversationModel,requestType,requestId,promptName,promptVersion,status" \
  --limit 1000 \
  --json \
  > .halo-careai/export/observations.json

uv run halo-careai convert-lf-pair \
  .halo-careai/export/trace.json \
  .halo-careai/export/observations.json \
  .halo-careai/care-ai-traces.jsonl
```

Session or conversation:

```bash
SESSION_ID="..."
mkdir -p .halo-careai/export/session

lf api traces list \
  --session-id "$SESSION_ID" \
  --fields "core,io,metrics" \
  --limit 100 \
  --json \
  > .halo-careai/export/session/traces.json

: > .halo-careai/care-ai-session-traces.jsonl

jq -r '(.body.data // .data // .items // .)[] | .id' .halo-careai/export/session/traces.json |
while read -r TRACE_ID; do
  lf api traces get "$TRACE_ID" --json \
    > ".halo-careai/export/session/${TRACE_ID}.trace.json"
  lf api observations list \
    --trace-id "$TRACE_ID" \
    --fields "core,basic,time,io,metadata,model,usage,prompt,metrics" \
    --expand-metadata "traceKind,toolName,toolCallId,parentAgent,targetAgent,targetAgentDisplayName,conversationId,ucid,sessionExecutorName,conversationModel,requestType,requestId,promptName,promptVersion,status" \
    --limit 1000 \
    --json \
    > ".halo-careai/export/session/${TRACE_ID}.observations.json"
  uv run halo-careai convert-lf-pair \
    ".halo-careai/export/session/${TRACE_ID}.trace.json" \
    ".halo-careai/export/session/${TRACE_ID}.observations.json" \
    .halo-careai/care-ai-session-traces.jsonl \
    --append
done
```

If an observations response includes a cursor, rerun `lf api observations list` with `--cursor` and append the next page before conversion.

## Inspect Locally

Before running HALO's LLM-backed diagnosis, inspect the converted JSONL locally:

```bash
uv run halo-careai inspect .halo-careai/care-ai-traces.jsonl
```

This prints metadata-level counts only: trace count, span count, model names, agent names, executor names, executed tool names, planned LLM function-call names, status codes, and token totals. It does not print raw user input, tool input, model output, or tool output. Use it to confirm the export has the expected executor, model, and tool coverage before spending an RLM run.

For machine-readable output:

```bash
uv run halo-careai inspect .halo-careai/care-ai-traces.jsonl --json
```

## Check Local Readiness

Before spending a HALO diagnosis run, check the local prerequisites:

```bash
uv run halo-careai doctor .halo-careai/care-ai-traces.sanitized.jsonl
```

For machine-readable output:

```bash
uv run halo-careai doctor .halo-careai/care-ai-traces.sanitized.jsonl --json
```

`doctor` reports whether `OPENAI_API_KEY` is present, whether `OPENAI_BASE_URL` is customized, whether the Deno/Pyodide sandbox is available for `run_code`, and the same metadata-level trace/audit counts used by `inspect` and `audit`. It does not print the API key.

To validate credentials with a live model-provider request:

```bash
uv run halo-careai doctor --check-model --model gpt-5.4-mini
```

For Care AI work that should use Tommy's GoCode provider:

```bash
uv run halo-careai doctor --check-model --gocode --model gpt-5.4-mini
```

`--gocode` sets the OpenAI-compatible base URL to GoCode for that command and uses `GOCODE_CODEX_API_KEY` as the OpenAI API key when present. The live check uses the Responses API. Do not treat `models.retrieve` failures as authoritative for GoCode readiness.

## Audit Locally

Run the deterministic audit before spending an RLM pass:

```bash
uv run halo-careai audit .halo-careai/care-ai-traces.jsonl
```

The audit flags metadata-only signals that should either seed a HALO question or verify a proposed fix:

| Signal | Meaning |
|---|---|
| Planned tool call missing executed `TOOL` span | `GENERATION.output` included a function call name, but no matching tool span appeared in the trace |
| Repeated tool call input fingerprint | Same `tool.name` plus same input fingerprint repeated within a trace, a likely exploration loop |
| `transfer_to_human` without diagnostic tool | Human transfer occurred with no non-transfer tool span in the same trace. This is a heuristic, not automatically a bug |
| Tool errors | `STATUS_CODE_ERROR` tool spans grouped with trace and span ids |
| High-token LLM span | LLM span crossed the configured input or output token threshold |
| Reported tool-count mismatch | `agent_execution.output.toolCallCount` disagrees with the converted executed `TOOL` span count, usually a sign the export or converter missed child-agent/tool evidence |

For machine-readable output:

```bash
uv run halo-careai audit .halo-careai/care-ai-traces.jsonl --json
```

The audit does not print raw user input, tool input, model output, or tool output. Repeated inputs are represented by a short SHA-256 fingerprint plus byte count.

## Compare Before And After

After applying a prompt or configuration change and collecting a candidate trace set, compare deterministic signals against the baseline:

```bash
uv run halo-careai compare-audits \
  .halo-careai/baseline-traces.jsonl \
  .halo-careai/candidate-traces.jsonl
```

For machine-readable output:

```bash
uv run halo-careai compare-audits \
  .halo-careai/baseline-traces.jsonl \
  .halo-careai/candidate-traces.jsonl \
  --json
```

The comparison reports count deltas and per-trace-rate deltas for the audit signals, input/output token total deltas, LLM-span token distributions, and session-level token distributions including median, p95, p99, max, and mean. Use it as a guardrail, not as the only shipping gate. A lower failure-signal rate should still be checked against the relevant Care AI evals, trace samples, and product-specific behavior.

## Build An Evidence Pack

Before running model-backed diagnosis on a batch, write a deterministic evidence pack:

```bash
uv run halo-careai evidence-pack \
  .halo-careai/care-ai-traces.sanitized.jsonl \
  .halo-careai/evidence/care-ai-routing-pack.json \
  --focus routing \
  --executor orchestrator-test
```

The evidence pack combines:

| Section | Contents |
|---|---|
| `inspect` | Metadata counts: traces, spans, sessions, models, executors, tools, planned tool calls, token totals, LLM token distributions, and session-level distributions |
| `care_ai_surface` | Deterministic prompt name, route, and `care-ai-agents` session-executor path for known executors |
| `audit` | Deterministic signals with raw `status.message` replaced by SHA-256 fingerprint and byte count |
| `safety` | Counts of raw payload attributes, raw status messages, known identifier attributes, and redaction markers |
| `doctor` | Local readiness summary, including sandbox status and optional GoCode model check |
| `diagnostic_prompt` | The exact Care AI HALO prompt for the selected focus, executor, session, and question |
| `candidate` / `comparison` | Optional before/after inspect, audit, safety, signal deltas, and token distribution deltas when `--candidate` is provided |

The pack is designed to stop the RLM from spending turns rediscovering basic facts through `TraceStore` APIs. It is also reviewable before any live model call. Use sanitized traces for packs that may be shared or sent through a model path.

To include a candidate trace set:

```bash
uv run halo-careai evidence-pack \
  .halo-careai/baseline-traces.sanitized.jsonl \
  .halo-careai/evidence/baseline-vs-candidate.json \
  --candidate .halo-careai/candidate-traces.sanitized.jsonl \
  --focus cost \
  --executor orchestrator-test
```

To include a live GoCode readiness check in the pack:

```bash
uv run halo-careai evidence-pack \
  .halo-careai/care-ai-traces.sanitized.jsonl \
  .halo-careai/evidence/care-ai-pack.json \
  --check-model \
  --gocode \
  --model gpt-5.4-mini
```

`evidence-pack` never prints raw payloads, status messages, or known identifier values. It only reports counts, fingerprints, metadata, and the generated prompt. It does not sanitize the input file for you. If `safety.safe_for_metadata_only_diagnosis` is `false`, run `sanitize` first or get explicit approval for raw payload analysis.

## Run A Local Pipeline

For a repeatable local run from an existing HALO JSONL file, use `local-pipeline`.
By default it writes deterministic artifacts only and does not call a model:

```bash
uv run halo-careai local-pipeline \
  .halo-careai/care-ai-traces.sanitized.jsonl \
  .halo-careai/runs/orchestrator-test-cost \
  --focus cost \
  --executor orchestrator-test
```

This writes:

| Artifact | Contents |
|---|---|
| `*-evidence-pack.json` | Inspect, audit, safety, doctor, and prompt context |
| `reports/*-experiment-plan.md` | Deterministic harness experiment plan |
| `reports/*-candidate-handoff.md` | Non-mutating prompt/config candidate handoff with proposed addendum and `lf` commands |
| `reports/*-loop-status.md` | Read-only loop status showing the proven local stage and next missing step |
| `*-manifest.json` | Run settings, artifact paths, diagnosis status, and summary counts |

Add `--diagnose` when you want the same run to call HALO and feed the resulting report into the plan:

```bash
uv run halo-careai local-pipeline \
  .halo-careai/care-ai-traces.sanitized.jsonl \
  .halo-careai/runs/orchestrator-test-cost \
  --focus cost \
  --executor orchestrator-test \
  --diagnose \
  --gocode \
  --tool-mode summary \
  --reasoning-effort low \
  --max-depth 0 \
  --max-turns 4 \
  --timeout-seconds 60
```

`local-pipeline` is intentionally local-file based. It does not fetch Langfuse data and does not mutate Care AI repos. Pull traces with the `lf` recipe, sanitize them, then use this command to produce the evidence, live diagnosis report when requested, experiment plan, candidate handoff, loop status, and manifest in one directory.

## Build An Experiment Plan

After HALO writes a diagnosis report, turn the finding into a deterministic experiment plan:

```bash
uv run halo-careai experiment-plan \
  .halo-careai/evidence/care-ai-pack.json \
  .halo-careai/reports/care-ai-experiment-plan.md \
  --diagnosis-report .halo-careai/reports/orchestrator-test-cost.md
```

The plan does not call a model. It combines the evidence pack and optional diagnosis report into:

| Section | Contents |
|---|---|
| Scope | Focus, executor, session, source artifacts |
| Care AI surface | Prompt name, endpoint route, and `care-ai-agents` session-executor file to inspect when the executor is known |
| Evidence summary | Trace/span/token totals, LLM token distributions, session-level distributions, top sessions by input tokens, plus audit counts |
| Hypothesis | The harness-level behavior to validate |
| Candidate change | Prompt/config/context change to try |
| Success metrics | Deterministic and review-based acceptance signals |
| Guardrails | What not to infer or change from the current evidence |
| Verification commands | `inspect`, `audit`, `compare-audits`, and candidate `evidence-pack` commands |

## Build A Candidate Handoff

After the plan identifies the Care AI surface, create a non-mutating handoff for the prompt or config owner:

```bash
uv run halo-careai candidate-handoff \
  .halo-careai/evidence/care-ai-pack.json \
  .halo-careai/reports/care-ai-candidate-handoff.md \
  --diagnosis-report .halo-careai/reports/orchestrator-test-cost.md
```

The handoff includes:

| Section | Contents |
|---|---|
| Care AI surface | Prompt name, endpoint route, and session-executor file |
| Prompt snapshot | Optional metadata-only prompt/version/body fingerprint from `prompt-snapshot` |
| Baseline evidence | Trace counts, token distributions, session-level cost shape, and audit counts |
| Proposed prompt addendum | A scoped candidate text block derived from the HALO finding |
| Langfuse commands | `lf api prompts get`, local `prompt-snapshot`, local `candidate-prompt-file`, local `candidate-review`, plus a `prompts create --curl` command for review |
| Verification commands | The same deterministic candidate trace comparison commands as the plan |

The generated Langfuse create command uses `--curl`, so it writes the request to an ignored local file instead of mutating Langfuse. The `--curl` output includes the full prompt body, so do not print it in shared terminals or paste it into model transcripts. The generated `prompts get` command uses `--label latest` because Care AI runtime prompt fetches use the current runtime label, not Conversation Design's `production` metadata label. Review the current prompt and candidate body before creating a new prompt version or label.

To include safe prompt provenance in the handoff, snapshot a local `lf` prompt export. This does not retain prompt body text:

```bash
lf api prompts get c1/airo-care/orchestrator-test/orchestrator \
  --label latest \
  --json \
  > .halo-careai/prompts/orchestrator-test-current.json

uv run halo-careai candidate-local-pipeline \
  .halo-careai/prompts/orchestrator-test-current.json \
  .halo-careai/evidence/care-ai-pack.json \
  .halo-careai/prompts \
  --environment test
```

`candidate-local-pipeline` is the default local path from evidence pack to candidate review. It writes the metadata-only prompt snapshot, candidate prompt file, candidate metadata, candidate review Markdown/JSON, approval-gated runtime plan Markdown/JSON, preflight Markdown/JSON, loop status, and a metadata-only manifest. It does not call Langfuse. The candidate prompt file itself contains the full prompt body for local human review, but the manifest, review packets, and status files retain only hashes and metadata.

When `--diagnosis-report` is provided, the candidate addendum includes a short HALO evidence block and sanitized diagnosis direction. This keeps the proposed prompt change tied to the RLM finding, for example token distribution and tool-loop evidence, without copying raw trace payloads into metadata artifacts.

For lower-level debugging, run the same pieces individually:

```bash
uv run halo-careai prompt-snapshot \
  .halo-careai/prompts/orchestrator-test-current.json \
  .halo-careai/prompts/orchestrator-test-current.snapshot.json

uv run halo-careai candidate-prompt-file \
  .halo-careai/prompts/orchestrator-test-current.json \
  .halo-careai/evidence/care-ai-pack.json \
  .halo-careai/prompts/orchestrator-test-cost-candidate.txt \
  --metadata-output .halo-careai/prompts/orchestrator-test-cost-candidate.metadata.json

lf api prompts create \
  --name c1/airo-care/orchestrator-test/orchestrator \
  --type text \
  --prompt-file .halo-careai/prompts/orchestrator-test-cost-candidate.txt \
  --labels halo-candidate \
  --tags halo,care-ai,cost \
  --commit-message 'HALO candidate: cost improvement for orchestrator-test' \
  --curl \
  > .halo-careai/prompts/orchestrator-test-cost-candidate.create.curl

uv run halo-careai candidate-review \
  .halo-careai/prompts/orchestrator-test-current.json \
  .halo-careai/prompts/orchestrator-test-cost-candidate.txt \
  .halo-careai/prompts/orchestrator-test-cost-candidate.metadata.json \
  .halo-careai/prompts/orchestrator-test-cost-candidate-review.md \
  --create-curl .halo-careai/prompts/orchestrator-test-cost-candidate.create.curl

uv run halo-careai candidate-handoff \
  .halo-careai/evidence/care-ai-pack.json \
  .halo-careai/reports/care-ai-candidate-handoff.md \
  --prompt-snapshot .halo-careai/prompts/orchestrator-test-current.snapshot.json
```

`candidate-prompt-file` writes a local prompt body for human review. It does not call Langfuse and its metadata file stores only fingerprints and provenance, not the prompt body. `candidate-review` checks that the candidate body equals the current prompt plus the HALO addendum, validates hashes, and writes a review packet without including the full prompt body.

`halo-candidate` is a review label. It does not automatically produce Care AI runtime traces because the service prompt manifest currently fetches the runtime label, usually `latest`. After a candidate review is ready, generate an approval-gated runtime plan before collecting candidate traces:

```bash
uv run halo-careai candidate-review \
  .halo-careai/prompts/orchestrator-test-current.json \
  .halo-careai/prompts/orchestrator-test-cost-candidate.txt \
  .halo-careai/prompts/orchestrator-test-cost-candidate.metadata.json \
  .halo-careai/prompts/orchestrator-test-cost-candidate-review.json \
  --create-curl .halo-careai/prompts/orchestrator-test-cost-candidate.create.curl \
  --format json

uv run halo-careai candidate-runtime-plan \
  .halo-careai/prompts/orchestrator-test-cost-candidate-review.json \
  .halo-careai/prompts/orchestrator-test-cost-runtime-plan.md \
  --environment test \
  --executor orchestrator-test \
  --focus cost

uv run halo-careai candidate-runtime-plan \
  .halo-careai/prompts/orchestrator-test-cost-candidate-review.json \
  .halo-careai/prompts/orchestrator-test-cost-runtime-plan.json \
  --environment test \
  --executor orchestrator-test \
  --focus cost \
  --format json

lf api prompts get c1/airo-care/orchestrator-test/orchestrator \
  --label latest \
  --json \
  > .halo-careai/prompts/orchestrator-test-current.json

uv run halo-careai candidate-preflight \
  .halo-careai/prompts/orchestrator-test-current.json \
  .halo-careai/prompts/orchestrator-test-cost-candidate-review.json \
  .halo-careai/prompts/orchestrator-test-cost-runtime-plan.json \
  .halo-careai/prompts/orchestrator-test-cost-candidate-preflight.md

uv run halo-careai candidate-preflight \
  .halo-careai/prompts/orchestrator-test-current.json \
  .halo-careai/prompts/orchestrator-test-cost-candidate-review.json \
  .halo-careai/prompts/orchestrator-test-cost-runtime-plan.json \
  .halo-careai/prompts/orchestrator-test-cost-candidate-preflight.json \
  --format json
```

The runtime plan is intentionally non-mutating. It prints the commands that would move the runtime `latest` label, rehydrate prompt cache, collect candidate traces, compare baseline against candidate, and roll back. The preflight step fetches `latest` again and verifies that the reviewed candidate still matches the current prompt version and hash. Do not run the approved mutation commands without explicit approval.

After an approved runtime push writes the created prompt JSON, verify that the pushed version is still the reviewed candidate before rehydrating prompt cache:

```bash
uv run halo-careai candidate-runtime-check \
  .halo-careai/prompts/orchestrator-test-cost-candidate.created.json \
  .halo-careai/prompts/orchestrator-test-cost-candidate-review.json \
  .halo-careai/prompts/orchestrator-test-cost-runtime-plan.json \
  .halo-careai/prompts/orchestrator-test-cost-runtime-check.md \
  --preflight .halo-careai/prompts/orchestrator-test-cost-candidate-preflight.json

uv run halo-careai candidate-runtime-check \
  .halo-careai/prompts/orchestrator-test-cost-candidate.created.json \
  .halo-careai/prompts/orchestrator-test-cost-candidate-review.json \
  .halo-careai/prompts/orchestrator-test-cost-runtime-plan.json \
  .halo-careai/prompts/orchestrator-test-cost-runtime-check.json \
  --preflight .halo-careai/prompts/orchestrator-test-cost-candidate-preflight.json \
  --format json
```

`candidate-runtime-check` stores only metadata and hashes. It verifies the prompt name, that the created version is newer than the reviewed source version, that labels include both `latest` and `halo-candidate`, that the created prompt body hash matches the reviewed candidate hash, and that the runtime plan expected the `latest` label move. A passing runtime check is the proof point before prompt-cache rehydrate and candidate traffic collection.

After candidate traces are collected, write the full post-candidate evaluation bundle:

```bash
uv run halo-careai candidate-evaluate \
  .halo-careai/evidence/care-ai-pack.json \
  .halo-careai/export/candidate-runtime/orchestrator-test/orchestrator-test-traces.sanitized.jsonl \
  .halo-careai/export/candidate-runtime/orchestrator-test \
  --runtime-check .halo-careai/prompts/orchestrator-test-cost-runtime-check.json \
  --runtime-plan .halo-careai/prompts/orchestrator-test-cost-runtime-plan.json \
  --executor orchestrator-test \
  --focus cost \
  --candidate-traffic-note "direct visitor API traffic; runtime cost proof, not authenticated shopper UI parity"
```

`candidate-evaluate` writes comparison JSON, a candidate evidence pack, decision JSON, decision Markdown, and a manifest. When `--runtime-check` is supplied, it first verifies that the candidate trace file includes spans for the pushed prompt name and version, so stale pre-candidate traffic cannot be evaluated as candidate traffic. If Langfuse omits `promptVersion`, it can fall back to proving every target prompt span starts after the created prompt timestamp; mixed pre-push traces still fail. For cost work, the embedded `candidate-decision` requires improved LLM-span input-token p95, improved session input-token p95, no increase in `high_token_llm_spans`, and a non-empty candidate set of token-bearing LLM spans when the baseline has LLM spans. Use `--candidate-traffic-note` when the collection path matters, for example direct visitor API traffic versus authenticated shopper UI traffic; the note is stored with the metadata-only candidate trace profile so reviewers do not overread the evidence. Use the session input-token distribution and top sessions by input tokens to separate one expensive turn from a multi-turn conversation that accumulated cost. It treats guardrail regressions in planned-missing tool calls, repeated tool calls, unsupported transfers, or tool errors as rollback signals.

`promote_candidate` is a deterministic candidate recommendation. It means the candidate trace set passed local target checks and still requires human approval. It does not authorize applying a Langfuse `production` label, production deployment, or prod prompt-cache rehydrate.

To audit where the loop currently stands without mutating external systems:

```bash
uv run halo-careai loop-status \
  --evidence-pack .halo-careai/evidence/care-ai-pack.json \
  --candidate-review .halo-careai/prompts/orchestrator-test-cost-candidate-review.json \
  --runtime-plan .halo-careai/prompts/orchestrator-test-cost-runtime-plan.json \
  --preflight .halo-careai/prompts/orchestrator-test-cost-candidate-preflight.json \
  --runtime-check .halo-careai/prompts/orchestrator-test-cost-runtime-check.json \
  --evaluation .halo-careai/export/candidate-runtime/orchestrator-test/orchestrator-test-cost-candidate-evaluation-manifest.json
```

`loop-status` reads local metadata artifacts only. It reports whether the next missing step is local artifact generation or the external approval-gated sequence: move the runtime `latest` label, verify the created prompt, rehydrate prompt cache, collect candidate traffic, then run `candidate-evaluate`.

For machine-readable candidate handoff:

```bash
uv run halo-careai candidate-handoff \
  .halo-careai/evidence/care-ai-pack.json \
  .halo-careai/reports/care-ai-candidate-handoff.json \
  --format json
```

For a machine-readable experiment plan:

```bash
uv run halo-careai experiment-plan \
  .halo-careai/evidence/care-ai-pack.json \
  .halo-careai/reports/care-ai-experiment-plan.json \
  --format json
```

## Run HALO

Once converted, use the normal engine:

```bash
uv run halo ./care-ai-traces.jsonl \
  --prompt "Across these Care AI traces, which tool-call or agent-routing failure patterns recur? Cite trace ids, tool names, and literal error strings." \
  --model gpt-5.4-mini \
  --max-turns 12
```

Ask HALO data questions, not code-change questions. HALO can cite failing tool spans, repeated loops, and model/token patterns. Verify any proposed harness change in `care-ai-agents` or `care-agent-sdk` before editing.

The Care AI helper includes prebuilt diagnostic prompts:

```bash
uv run halo-careai diagnose .halo-careai/care-ai-traces.jsonl \
  --focus tool-errors \
  --executor airo-care-orchestrator \
  --gocode \
  --max-turns 12
```

`diagnose` prepends a metadata-only local evidence summary by default. The summary includes inspect counts, audit counts and examples, and safety counts, so HALO can start from known facts instead of spending turns discovering dataset basics through `TraceStore` tools. Limit examples with `--evidence-top`; disable this prelude with `--no-include-evidence` when you intentionally want the raw HALO behavior.

The Care AI wrapper also defaults to `--tool-mode summary`. In summary mode HALO can use dataset overview, trace summaries, counts, context, and synthesis tools, but cannot call `view_trace`, `view_spans`, `search_trace`, `search_span`, or `run_code`. That default is intentional for small sanitized batches where the deterministic evidence pack already contains the important signals. Use `--tool-mode full` only when a diagnosis truly needs raw per-span inspection.

For live GoCode-backed batch diagnosis, start with constrained settings:

```bash
uv run halo-careai diagnose .halo-careai/care-ai-traces.sanitized.jsonl \
  --focus cost \
  --executor orchestrator-test \
  --gocode \
  --tool-mode summary \
  --reasoning-effort low \
  --max-depth 0 \
  --max-turns 4 \
  --timeout-seconds 60 \
  --output .halo-careai/reports/orchestrator-test-cost.md \
  --events-jsonl .halo-careai/reports/orchestrator-test-cost.events.jsonl
```

Raise the timeout or use `--tool-mode full` only after the summary-mode run shows a concrete gap that needs per-span detail.
`--output` writes the final root answer. `--events-jsonl` writes durable HALO agent events for replay/debugging. Use sanitized traces when writing events that may be shared.

Review the generated prompt before sending trace content through HALO's model path:

```bash
uv run halo-careai diagnose .halo-careai/care-ai-traces.jsonl \
  --focus routing \
  --executor orchestrator-test \
  --print-prompt
```

The prompt includes Care AI harness context so findings map to the right surface: production orchestrator prompts, DNS or lifecycle specialist prompts, variant specialist tools, help-center variant tools, or code/config wiring. Treat it as bounding context only. The trace evidence still has to prove the failure pattern.

Available focus modes:

| Focus | Use for |
|---|---|
| `overview` | Recurring harness-level failure patterns across the dataset |
| `tool-errors` | Failed tool spans, exact error strings, recovery behavior |
| `routing` | Misrouting, unnecessary agent-as-tool calls, premature transfer |
| `loops` | Repeated calls, repeated arguments, inefficient exploration |
| `cost` | Token outliers and model usage tied to waste patterns |
| `mcp` | Product/domain/case-management context gaps and redundant calls |

## Care AI Notes

- Langfuse `GENERATION.input` usually contains only the latest user message. It does not prove the agent lacked system prompt or prior-turn context.
- The converter extracts function-call names from `GENERATION.output` into `care_ai.llm_planned_tool_names`. Compare planned calls with actual `TOOL` spans before calling a tool missing or hallucinated.
- Group by `langfuse.session_id` or `care_ai.conversation_id` when you need multi-turn conversation analysis. A single HALO trace still represents one customer-message execution. The Care AI wrapper's inspect/evidence outputs include session-level distributions for this reason.
- The built-in diagnosis prompt distinguishes production `airo-care-orchestrator` from test, sandbox, TCF, and email-MCP-bot variants. Pass `--executor` when your trace file mixes executors.
- Current orchestrator executors have domain lifecycle case creation disabled. Treat a direct human transfer after lifecycle statuses that require specialist intervention as potentially correct, not automatically premature.
- `diagnose` sends trace contents through HALO's model path unless `--print-prompt` is set. Use `inspect` first. If raw payloads are not approved for model analysis, run `sanitize` and diagnose the sanitized copy.
- Use `--project-id` if you want separate HALO filters for prod, test, sandbox, or a specific executor:

```bash
uv run halo-careai convert-langfuse export.json traces.jsonl \
  --project-id care-ai-orchestrator-test \
  --service-name care-ai-agents
```
