# HALO Security Audit — New Findings

**Scope:** `app/` directory (TypeScript/Bun server, Electron desktop, React frontend)
**Date:** 2026-08-19

All previously known issues (wildcard CORS, unauthenticated tRPC, SSRF, credential exfiltration, etc.) are excluded from this report.

---

## Finding 1: Stored XSS via Mermaid Diagram Rendering (securityLevel: "loose")

**Severity:** HIGH
**Location:** `app/src/mainview/lib/ui/components/custom/MermaidDiagram.tsx`, lines 215 and 366; `app/src/mainview/lib/ui/utils/mermaidTheme.ts`, line 51

**Description:**
The `MermaidDiagram` component renders Mermaid-generated SVG output directly into the DOM using `dangerouslySetInnerHTML={{ __html: renderState.svg }}`. Critically, the Mermaid library is configured with `securityLevel: "loose"` in `mermaidTheme.ts` (line 51). With this setting, Mermaid explicitly enables:
- **Click event handlers** (`click nodeId callback`) in diagram definitions, which generate inline `onclick` JavaScript in the SVG output
- **HTML labels** that can contain arbitrary HTML elements

The Mermaid library is initialized with a global configuration that persists across all `mermaid.render()` calls in the process. This setting enables Mermaid's `click` callback syntax and HTML label parsing, which produce SVG output containing inline `onclick` handlers and raw HTML elements.

Currently, `MermaidDiagram` is only rendered in the gallery/demo page (`GalleryPage.tsx`) with hardcoded sample data — it is **not yet wired** into the HALO run report markdown renderer (`RunReportView.tsx`). However, the component is exported from the UI library (`lib/ui/index.ts`) and is clearly designed to render LLM-generated mermaid diagram code blocks from run answers. The `securityLevel: "loose"` setting is a latent HIGH-severity issue that becomes immediately exploitable the moment `MermaidDiagram` is used to render any user- or LLM-controlled content.

Additionally, `mermaid.initialize()` sets **global** state. Any other call to `mermaid.render()` anywhere in the application (even outside `MermaidDiagram`) will inherit the `securityLevel: "loose"` setting after the first render of a `MermaidDiagram`.

**Impact:**
When connected to LLM-controlled content (the intended use): arbitrary JavaScript execution in the context of the HALO desktop application or web UI. In the Electron/ElectroBun desktop context, this escalates to full native code execution via `Utils.openExternal`, `Utils.openPath`, filesystem access, and the IPC bridge. An attacker could exfiltrate all stored API keys, provider credentials, and database contents.

**Attack Path (once wired to dynamic content):**
1. Attacker crafts OTLP span data containing a `final_answer` with a malicious Mermaid diagram (e.g., `graph LR; A-->B; click A "javascript:alert(document.cookie)"`) or manipulates a HALO model provider to return malicious output.
2. Mermaid's `securityLevel: "loose"` processes click callbacks and HTML labels, generating SVG with inline JavaScript event handlers.
3. The SVG is injected into the DOM via `dangerouslySetInnerHTML`.
4. Malicious script executes, accesses IPC bridge, exfiltrates credentials.

**Evidence:**
```typescript
// mermaidTheme.ts:49-51
return {
  startOnLoad: false,
  securityLevel: "loose",  // Enables click callbacks and HTML labels in SVG output
```
```typescript
// MermaidDiagram.tsx:215
dangerouslySetInnerHTML={{ __html: renderState.svg }}
// MermaidDiagram.tsx:366
dangerouslySetInnerHTML={{ __html: renderState.svg }}
```

**Remediation:**
1. Change `securityLevel` to `"strict"` (or at minimum `"sandbox"`) in `mermaidTheme.ts`. This disables click callbacks and encodes HTML in text labels.
2. Additionally, sanitize SVG output with DOMPurify before injecting into the DOM, or render Mermaid into an `<img>` tag via a data URI.

---

## Finding 2: Arbitrary File Read via `fileImport.imports.preview` tRPC Endpoint

**Severity:** HIGH
**Location:** `app/src/server/router.ts`, lines 953–967; `app/src/server/fileimport/parser.ts`, lines 100–148

**Description:**
The `fileImport.imports.preview` tRPC endpoint accepts a `filePath` parameter with no path validation or sandboxing. It calls `previewJsonlFile(input.filePath)` which uses `Bun.file(filePath)` to open any file on the filesystem. While the parser expects JSONL format, the error messages and behavior still reveal whether arbitrary files exist and can leak partial file contents through error messages. More critically, the `fileImport.imports.start` endpoint (line 969–984) passes the attacker-supplied `filePath` directly to `service.start({ filePath: input.filePath })`, which reads and processes the file.

Combined with the known unauthenticated API, any cross-origin page can probe and read arbitrary files on the system.

**Impact:**
Arbitrary file existence checking and partial content leakage via error messages. Combined with the `start` endpoint, contents of any file that happens to contain valid JSON lines would be ingested into the database and become readable via the traces API.

**Attack Path:**
1. Attacker sends `fileImport.imports.preview` with `filePath: "/etc/passwd"` — gets existence confirmation.
2. Attacker sends `fileImport.imports.start` with `filePath: "/path/to/sensitive.jsonl"` — file is read and ingested.
3. Attacker reads ingested data via `traces.list` or `spans.list`.

**Evidence:**
```typescript
// router.ts:954-955
.input(z.object({ filePath: z.string().min(1) }))
.query(async ({ input }) => {
  return await previewJsonlFile(input.filePath); // No path validation
```

**Remediation:**
Restrict `filePath` to a whitelist of allowed directories (e.g., the uploads directory and app data directory). Validate that the resolved path is within the allowed directory using `path.resolve()` and prefix checking.

---

## Finding 3: Sensitive Data Exposure via `/health` Endpoint

**Severity:** MEDIUM
**Location:** `app/src/server/app.ts`, lines 49–55

**Description:**
The `/health` endpoint returns the full database file path (`database.path`) in its response body. This leaks the user's filesystem layout, including their username (e.g., `/Users/johnsmith/Library/Application Support/net.inference.halo/halo-canvas.sqlite`).

**Impact:**
Information disclosure of the host filesystem path structure and potentially the user's system username. This aids path-traversal attacks and social engineering.

**Attack Path:**
1. Any cross-origin request to `http://127.0.0.1:8799/health` returns the full database path.
2. Attacker learns the username and filesystem layout.

**Evidence:**
```typescript
// app.ts:49-55
app.get("/health", (c) =>
  c.json({
    dbPath: database.path, // Leaks full filesystem path
    ok: true,
    service: "halo-canvas-telemetry",
  }),
);
```

**Remediation:**
Remove `dbPath` from the health endpoint response, or restrict it to authenticated requests.

---

## Finding 4: Sensitive Data Exposure via `telemetry.info` tRPC Endpoint

**Severity:** MEDIUM
**Location:** `app/src/server/telemetry/storage.ts`, lines 746–799

**Description:**
The `getTelemetryInfo` function returns `dbPath` (full filesystem database path), `ingestUrl`, and `liveUrl` in its response. The database path leaks internal filesystem structure.

**Impact:**
Information disclosure of internal paths and service URLs, aiding further attacks.

**Evidence:**
```typescript
// storage.ts:779-780
return {
  dbPath, // Full filesystem path exposed to any caller
  ingestUrl,
```

**Remediation:**
Remove or redact filesystem paths from API responses.

---

## Finding 5: `runner-config.json` Written with Provider API Keys in Plaintext — World-Readable

**Severity:** MEDIUM
**Location:** `app/src/server/halo/runQueue.ts`, lines 418–446

**Description:**
When a HALO analysis run executes, the runner configuration (including the model provider's API key, base URL, and custom headers) is written to a JSON file on disk at a predictable path (`<data-dir>/halo-runs/<run-id>/runner-config.json`). This file is created with default permissions (typically world-readable on most systems). Any local process or user can read these credentials.

**Impact:**
Plaintext API keys for LLM providers (OpenAI, Anthropic, etc.) stored on disk with predictable paths and default permissions. Any local attacker or malware can harvest these keys.

**Attack Path:**
1. User configures a HALO model provider with their API key.
2. User starts a HALO analysis run.
3. Runner config is written to `<data-dir>/halo-runs/<run-id>/runner-config.json`.
4. Any process on the system reads the file and extracts the API key.

**Evidence:**
```typescript
// runQueue.ts:423-446
writeFileSync(
  configPath,
  JSON.stringify({
    // ...
    provider: {
      apiKey: provider.apiKey,    // Plaintext API key
      baseUrl: provider.baseUrl,
      headers: provider.headers,  // May contain auth tokens
    },
    // ...
  }, null, 2),
  "utf8",  // No restrictive file permissions
);
```

**Remediation:**
Write `runner-config.json` with mode `0o600` (owner-read-only). Better yet, pass credentials via environment variables or stdin to the Python subprocess instead of writing them to disk.

---

## Finding 6: Arbitrary URL Opening in Desktop App Without Validation

**Severity:** MEDIUM
**Location:** `app/src/bun/index.ts`, line 82; `app/src/mainview/desktop/desktopBridge.ts`, line 173

**Description:**
The `openExternal` IPC handler calls `Utils.openExternal(url)` with the URL received from the renderer process without any validation or allowlisting. An attacker who achieves XSS (e.g., via Finding 1) or controls content rendered in the webview can open arbitrary URLs, including `file://` URIs and custom protocol handlers.

**Impact:**
In the desktop context, opening arbitrary URLs can launch applications via protocol handlers (`tel:`, `mailto:`, `ssh:`, custom schemes), potentially leading to further exploitation depending on installed protocol handlers.

**Attack Path:**
1. Attacker achieves XSS via Mermaid injection (Finding 1).
2. Malicious JS calls the IPC bridge: `rpc.request.openExternal({ url: "file:///etc/passwd" })`.
3. System opens the file or triggers a protocol handler.

**Evidence:**
```typescript
// bun/index.ts:82
openExternal: ({ url }) => ({ ok: Utils.openExternal(url) }),
```

**Remediation:**
Validate URLs in the `openExternal` handler — only allow `https:` and `http:` schemes, and optionally maintain a domain allowlist.

---

## Finding 7: Race Condition in HALO Run State Transitions

**Severity:** MEDIUM
**Location:** `app/src/server/halo/runQueue.ts`, lines 116–131 (cancel), 138–173 (continueRun)

**Description:**
The `cancel` and `continueRun` methods perform a read-then-write on the run status without transactional protection. Between the `getHaloRun()` check and the `updateHaloRun()` call, a concurrent request can change the run's state. For example, `continueRun` checks the status is in a terminal set but another request could cancel the run between the check and the queue addition, leading to a cancelled run being re-enqueued and processed.

**Impact:**
A run that has been cancelled could be silently re-activated, or a run already in progress could have its state corrupted. This could lead to concurrent Python processes running against the same data, consuming LLM API credits unexpectedly.

**Attack Path:**
1. Attacker sends `halo.runs.continue` and `halo.runs.cancel` in rapid succession.
2. The continue operation passes the status check before cancel takes effect.
3. A cancelled run gets re-enqueued and resumes execution.

**Evidence:**
```typescript
// runQueue.ts:139-140
const run = getHaloRun(database.sqlite, runId);  // Read
if (!run) return null;
// ... status check ...
// Gap here: another request can cancel the run
const queued = await queue.add(HALO_ROUTE, ...);  // Write
```

**Remediation:**
Wrap the read-check-write pattern in a SQLite transaction, or use an atomic compare-and-swap update: `UPDATE halo_runs SET status = 'queued' WHERE id = ? AND status IN ('completed', 'failed', ...)`.

---

## Finding 8: FTS5 Injection via Search Query

**Severity:** LOW
**Location:** `app/src/server/telemetry/storage.ts`, lines 1318–1325

**Description:**
The `searchFtsQuery` function builds an FTS5 MATCH expression by splitting user input on whitespace, stripping some special characters, and appending `*` for prefix matching. However, the character stripping (`/[^a-zA-Z0-9_./:-]/g`) still allows FTS5 operators like `.` and `:` through, and the constructed query is used directly in a `MATCH` expression. While the code has a try/catch fallback to LIKE search, crafted queries can trigger expensive FTS5 operations.

Additionally, the `searchSessionIds` function (line 1383) uses `%${q}%` in LIKE patterns without escaping the `%` and `_` wildcard characters in the user input itself, allowing users to craft LIKE patterns with unexpected wildcards.

**Impact:**
Potential for expensive FTS5 queries causing performance degradation (minor DoS), and LIKE pattern injection allowing broader searches than intended.

**Evidence:**
```typescript
// storage.ts:1319-1324
function searchFtsQuery(q: string) {
  return q
    .split(/\s+/)
    .map((term) => term.replace(/[^a-zA-Z0-9_./:-]/g, ""))  // Allows FTS operators
    .filter(Boolean)
    .map((term) => `${term}*`)
    .join(" ");
}
```

**Remediation:**
Escape or quote FTS5 special characters properly. For LIKE queries, escape `%` and `_` in user input.

---

## Finding 9: Database Path Exposure in Error Messages

**Severity:** LOW
**Location:** `app/src/server/router.ts`, lines 571–578; `app/src/server/langfuse/client.ts`, line 481

**Description:**
Several tRPC error handlers include raw error messages from SQLite and external services in their responses. SQLite errors frequently contain file paths, table names, and column names. The Langfuse client returns up to 300 characters of the remote server's response body in error messages, which could contain sensitive internal information from the Langfuse instance.

**Impact:**
Information leakage about internal database schema, file paths, and third-party service details via error responses.

**Evidence:**
```typescript
// router.ts:572-573
const detail = error instanceof Error ? error.message : "Unknown SQLite error.";
throw new TRPCError({
  code: "INTERNAL_SERVER_ERROR",
  message: `Connected to Langfuse, but could not save the connection locally: ${detail}`,
});
```

```typescript
// langfuse/client.ts:481-483
return trimmed
  ? `Langfuse returned HTTP ${status}: ${trimmed.slice(0, 300)}`
  : `Langfuse returned HTTP ${status}`;
```

**Remediation:**
Sanitize error messages before returning them in API responses. Log detailed errors server-side and return generic messages to clients.

---

## Finding 10: Unbounded Live Event Replay Leading to Memory Pressure

**Severity:** LOW
**Location:** `app/src/server/router.ts`, lines 1260–1296; `app/src/server/live/events.ts`

**Description:**
The `streamLiveEvents` generator tracks seen event IDs in a `Set<number>` (`seenIds`) that grows unboundedly for the lifetime of the subscription. A long-running WebSocket subscription with high event throughput will accumulate event IDs in memory indefinitely. Additionally, the replay function loads historical events without a cap.

**Impact:**
Memory growth proportional to the total number of events published during a subscription's lifetime. Under sustained high ingest rates, this can contribute to memory pressure and eventual OOM.

**Evidence:**
```typescript
// router.ts:1267
const seenIds = new Set<number>();  // Grows without bound
// ...
seenIds.add(event.id);
```

**Remediation:**
Use a bounded data structure (e.g., a fixed-size ring buffer or LRU set) for `seenIds`, or periodically trim IDs below the current minimum queue position.

---

## Finding 11: SQLite `PRAGMA table_info` Injection via Table Name

**Severity:** LOW (currently unexploitable — all call sites use hardcoded table names)
**Location:** `app/src/server/db/client.ts`, lines 463–468

**Description:**
The `ensureColumn` function interpolates `tableName` and `columnName` directly into SQL strings without parameterization. While currently all callers use hardcoded string literals, this pattern is fragile: any future caller passing user-controlled input would create a SQL injection vulnerability.

**Impact:**
No current impact since all inputs are hardcoded. However, this is a latent vulnerability that could be activated by future code changes.

**Evidence:**
```typescript
// db/client.ts:464-467
const columns = sqlite
  .query<{ name: string }, []>(`PRAGMA table_info(${tableName})`)  // Unparameterized
  .all();
sqlite.run(`ALTER TABLE ${tableName} ADD COLUMN ${columnName} ${definition}`);
```

**Remediation:**
Validate table/column names against an allowlist, or at minimum add a comment documenting that these must never accept user input.
