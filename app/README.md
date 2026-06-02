# HALO

HALO is a local desktop trace monitor for developers building AI agents. It runs as an ElectroBun app with a Bun/Hono/tRPC backend, stores OpenTelemetry trace data in SQLite, and renders a real-time tracing workspace in React.

Use it when you want to point a locally running agent at a desktop app and watch traces, spans, sessions, model calls, tool calls, inputs, outputs, token usage, errors, imported Langfuse history, and local HALO analysis runs.

## What You Can Do

- Receive local OTLP/JSON traces at `http://127.0.0.1:8799/v1/traces`.
- Watch traces update live through tRPC WebSocket subscriptions.
- Open trace detail sheets with tree, timeline, span, and raw JSON tabs.
- View sessions, which group multiple traces by `session.id`.
- Enable Follow Latest to keep the detail sheet locked to the newest trace.
- Import historical traces from Langfuse.
- Run local HALO analysis over filtered trace groups or session groups.
- Search and filter by status, source, service, agent, model, and time window.
- Clear local telemetry data from the UI.
- Browse the copied UI component gallery at `/components`.

## Tech Stack

- Desktop shell: ElectroBun
- Runtime/backend: Bun, Hono, tRPC
- Database: SQLite, Drizzle
- Frontend: React, TanStack Start/Router, TanStack Query, Tailwind CSS
- Background jobs: bunqueue for Langfuse import and HALO analysis tasks

## Prerequisites

- Bun installed locally.
- macOS is the primary development target right now because the app is an ElectroBun desktop app.
- Access to any local agent or script that can emit OTLP/JSON traces.
- Optional: a Langfuse instance and Langfuse API keys if you want to test imports.
- Optional for HALO analysis: `git`, `uv`, Python 3.12, and an OpenAI-compatible model provider key.

## Install The Desktop App

The v1 distribution target is macOS Apple Silicon and Ubuntu/Debian x64:

```bash
curl -fsSL https://inference.net/halo/install.sh | sh
```

Installer options:

```bash
HALO_CHANNEL=canary curl -fsSL https://inference.net/halo/install.sh | sh
HALO_NO_OPEN=1 curl -fsSL https://inference.net/halo/install.sh | sh
HALO_INSTALL_DIR="$HOME/Applications" curl -fsSL https://inference.net/halo/install.sh | sh
```

Release engineering details live in [`docs/distribution.md`](docs/distribution.md).

## First-Time Setup

Install dependencies:

```bash
bun install
```

Start the desktop app:

```bash
bun run dev
```

`bun run dev` will:

- push the SQLite schema with Drizzle
- start the Vite/TanStack Start dev server on `127.0.0.1:5173`
- start the local telemetry backend on `127.0.0.1:8799`
- start the live WebSocket server on `127.0.0.1:8800`
- launch ElectroBun in watch mode

In development, the local SQLite database defaults to:

```text
data/halo-canvas.sqlite
```

The `data/` directory is ignored by git.

Packaged desktop builds store mutable app data outside the `.app` bundle:

```text
~/Library/Application Support/net.inference.halo/
```

That directory contains the local SQLite database, SQLite WAL/SHM files,
bunqueue state, imported trace metadata, model provider settings, and the local
HALO engine install.

## Agent Configuration

Configure local agents to send OTLP traces here:

```bash
export CATALYST_OTLP_ENDPOINT=http://127.0.0.1:8799/v1/traces
```

The current ingest endpoint supports:

- `POST /v1/traces`
- OTLP JSON payloads
- gzip-compressed JSON via `Content-Encoding: gzip`

The current ingest endpoint does not support OTLP protobuf yet.

## Generate Test Traces

With the app running, generate a local test trace with 10 spans:

```bash
bun run fire:test-spans
```

You can also point this script at another endpoint:

```bash
bun scripts/fire-test-spans.ts --endpoint http://127.0.0.1:8799/v1/traces
```

## Langfuse Import

Click **Import Data** in the top bar to import historical Langfuse traces.

You need:

- API URL, for example `http://localhost:3001`
- Langfuse public key
- Langfuse secret key

The import flow will:

- smoke test the Langfuse connection
- discover available trace facets
- let you choose filters such as date range, environment, trace name, tags, user, session, version, and release
- enqueue an import job with bunqueue
- show progress in the dialog
- write imported traces into the same local SQLite span tables

Credentials are stored in the local SQLite database so the app can reuse saved connections. Do not commit anything under `data/`, and do not commit `.env` files.

## Local HALO Analysis

Open **Settings** to install/update the local HALO engine and save a model provider. HALO auto-clones the HALO repo into local app data, validates `uv`/Python/imports, and stores provider keys in the local SQLite database with masked UI display.

Open **Analysis** to create a HALO run over:

- filtered trace groups
- filtered session groups

Single-trace HALO runs are intentionally not part of v1. Each run exports matching spans to local JSONL under `data/halo-runs/`, runs the HALO engine locally through a structured Python bridge, streams progress back to the UI, and stores the final result as a standalone run record linked to the source filters. HALO orchestration is local, but model calls may leave the machine depending on the provider you configure.

## Useful Commands

```bash
bun run dev
```

Run the full desktop development loop.

```bash
bun run dev:web
```

Run only the Vite/TanStack Start web server.

```bash
bun run db:push
```

Push the Drizzle schema to the configured SQLite database.

```bash
bun run typecheck
```

Run TypeScript checks.

```bash
bun test
```

Run the test suite.

```bash
bun run build:web
```

Build the TanStack Start frontend.

```bash
bun run build:desktop
```

Build the desktop app bundle.

```bash
bun run build:stable
```

Build stable distribution artifacts. Production macOS builds require Developer ID signing and notarization secrets; use `HALO_SKIP_CODESIGN=1` for an unsigned local dry run.

```bash
bun run release:manifest
bun run release:verify
```

Generate and verify release metadata for the current `artifacts/` directory.

## Configuration

Most development works without any `.env` file. These environment variables are available when needed:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HALO_APP_DATA_DIR` | `~/Library/Application Support/net.inference.halo` in packaged macOS builds | Directory for packaged desktop app data. |
| `HALO_DB_PATH` | `data/halo-canvas.sqlite` in development; `HALO_APP_DATA_DIR/halo-canvas.sqlite` in packaged builds | SQLite database path. Use this to point at a separate local DB. |
| `HALO_VIEW_URL` | `http://127.0.0.1:5173` in `bun run dev`; bundled `views://` in builds | Frontend URL ElectroBun should load. Useful when running a separate web dev server. |
| `VITE_TRPC_HTTP_URL` | `http://127.0.0.1:8799/trpc` | Browser-side tRPC HTTP URL. |
| `VITE_TRPC_WS_URL` | `ws://127.0.0.1:8800` | Browser-side tRPC WebSocket URL. |
| `CATALYST_OTLP_ENDPOINT` | `http://127.0.0.1:8799/v1/traces` in helper scripts | OTLP endpoint used by local trace emitters and `fire:test-spans`. |

Example with a temporary database:

```bash
HALO_DB_PATH=/tmp/halo-canvas.sqlite bun run dev
```

## Routes

- `/` redirects to the trace monitor.
- `/traces` shows trace list and trace details.
- `/sessions` shows conversations grouped by `session.id`.
- `/analysis` shows local HALO analysis runs.
- `/settings` configures the HALO engine and model providers.
- `/components` shows the local UI component gallery.

The desktop app normally uses hash-style routing under ElectroBun, so direct browser testing may use URLs such as:

```text
http://127.0.0.1:5173/#/traces
http://127.0.0.1:5173/#/sessions
http://127.0.0.1:5173/#/analysis
http://127.0.0.1:5173/#/settings
```

## Local Data And Safety

Ignored local-only paths include:

- `data/`
- `langfuse/`
- `.env`
- `*.sqlite`, `*.sqlite-shm`, `*.sqlite-wal`

Before publishing changes, check staged content with:

```bash
git diff --cached --check
git status --short
```

Do not commit real Langfuse keys, agent tokens, SQLite databases, Docker compose env files, or trace exports containing private customer data.

## Troubleshooting

If ports are already in use, stop the existing process using:

- `127.0.0.1:5173` for the frontend
- `127.0.0.1:8799` for the telemetry API
- `127.0.0.1:8800` for live updates

If the app opens but shows no traces, confirm your agent is sending OTLP JSON to:

```text
http://127.0.0.1:8799/v1/traces
```

If you want a clean local database, use **Clear Data** in the app, or stop the app and remove the local SQLite files under `data/`.

If Langfuse import fails, verify:

- the API URL is reachable from your machine
- the public key and secret key belong to the project you want to import
- the Langfuse server supports the public trace APIs used by the importer

If HALO analysis cannot start, verify:

- **Settings** shows the HALO engine as installed
- `git`, `uv`, and Python 3.12 are available
- at least one model provider is saved and passes the provider test
- the selected filters match at least one trace or session

## Optional Gator Fixtures

`tests/gator/` contains direct-run helper scripts for sending traces from a local Gator development stack. They require local Docker services and environment-specific auth, so they are intentionally guarded during normal `bun test`.

Run them only when the local Gator stack is available:

```bash
bun run tests/gator/gator-multiturn.test.ts
bun run tests/gator/gator-singleshot.ts
```
