import { dirname, resolve } from "node:path";
import type { Database } from "bun:sqlite";
import type { LiveEventStore } from "../live/events";
import type { TelemetryFilters } from "../telemetry/types";
import {
  HALO_REPO_URL,
  type HaloEngineStatus,
  type HaloModelProvider,
  type HaloProviderType,
  type HaloRun,
  type HaloRunEvent,
  type HaloRunSnapshot,
  type HaloRunStatus,
  type HaloRunTargetType,
  type StoredHaloModelProvider,
} from "./types";

type ProviderRow = {
  id: string;
  name: string;
  provider_type: HaloProviderType;
  base_url: string;
  api_key: string;
  model: string;
  headers_json: string;
  last_status: string;
  last_error: string | null;
  last_tested_at: number | null;
  created_at: number;
  updated_at: number;
};

type RunRow = {
  id: string;
  bunqueue_job_id: string | null;
  title: string;
  status: HaloRunStatus;
  target_type: HaloRunTargetType;
  filters_json: string;
  provider_id: string | null;
  provider_name: string;
  model: string;
  prompt: string;
  max_depth: number;
  max_turns: number;
  max_parallel: number;
  trace_count: number;
  session_count: number;
  span_count: number;
  progress: number;
  export_path: string | null;
  result_path: string | null;
  final_answer: string | null;
  final_answer_source: string | null;
  error_message: string | null;
  created_at: number;
  updated_at: number;
  started_at: number | null;
  finished_at: number | null;
};

type EventRow = {
  id: number;
  run_id: string;
  sequence: number;
  event_type: string;
  payload_json: string;
  created_at: number;
};

type EngineRow = {
  id: string;
  repo_url: string;
  install_path: string;
  status: "not_installed" | "installing" | "installed" | "error";
  commit_sha: string | null;
  last_error: string | null;
  installed_at: number | null;
  updated_at: number;
};

export function defaultHaloInstallPath(dbPath: string) {
  if (dbPath === ":memory:") return resolve("data/halo-engine");
  return resolve(dirname(dbPath), "halo-engine");
}

export function getHaloEngineSettings(
  sqlite: Database,
  dbPath: string,
): HaloEngineStatus {
  const row = sqlite
    .query<EngineRow, []>(
      `SELECT *
       FROM halo_engine_settings
       WHERE id = 'default'
       LIMIT 1`,
    )
    .get();
  const defaultInstallPath = defaultHaloInstallPath(dbPath);
  const installPath = normalizedHaloInstallPath(
    row?.install_path ?? null,
    defaultInstallPath,
  );
  return {
    checks: {
      git: null,
      importable: false,
      python: null,
      uv: null,
    },
    commitSha: row?.commit_sha ?? null,
    defaultInstallPath,
    installedAt: row?.installed_at ? isoFromMs(row.installed_at) : null,
    installPath,
    lastError: row?.last_error ?? null,
    repoUrl: row?.repo_url ?? HALO_REPO_URL,
    status: row?.status ?? "not_installed",
    updatedAt: row?.updated_at ? isoFromMs(row.updated_at) : null,
  };
}

export function normalizedHaloInstallPath(
  storedPath: string | null | undefined,
  defaultInstallPath: string,
) {
  if (!storedPath?.trim()) return defaultInstallPath;

  const resolved = resolve(storedPath);
  if (isLegacyMacBundleDataPath(resolved)) return defaultInstallPath;
  return resolved;
}

function isLegacyMacBundleDataPath(path: string) {
  const normalized = path.replaceAll("\\", "/");
  return (
    normalized.includes(".app/Contents/MacOS/data/") ||
    normalized.endsWith(".app/Contents/MacOS/data")
  );
}

export function saveHaloEngineSettings(
  sqlite: Database,
  input: {
    dbPath: string;
    status: EngineRow["status"];
    commitSha?: string | null;
    error?: string | null;
    installPath?: string;
    repoUrl?: string;
  },
) {
  const now = Date.now();
  const existing = getHaloEngineSettings(sqlite, input.dbPath);
  sqlite
    .query(
      `INSERT INTO halo_engine_settings (
        id, repo_url, install_path, status, commit_sha, last_error,
        installed_at, updated_at
      ) VALUES ('default', ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        repo_url = excluded.repo_url,
        install_path = excluded.install_path,
        status = excluded.status,
        commit_sha = excluded.commit_sha,
        last_error = excluded.last_error,
        installed_at = excluded.installed_at,
        updated_at = excluded.updated_at`,
    )
    .run(
      input.repoUrl ?? existing.repoUrl,
      input.installPath ?? existing.installPath,
      input.status,
      input.commitSha ?? existing.commitSha,
      input.error ?? null,
      input.status === "installed"
        ? Date.parse(existing.installedAt ?? "") || now
        : existing.installedAt
          ? Date.parse(existing.installedAt)
          : null,
      now,
    );
}

export function listHaloProviders(sqlite: Database): HaloModelProvider[] {
  return sqlite
    .query<ProviderRow, []>(
      `SELECT *
       FROM halo_model_providers
       ORDER BY updated_at DESC`,
    )
    .all()
    .map((row) => mapProvider(row, false));
}

export function getHaloProvider(
  sqlite: Database,
  id: string,
): StoredHaloModelProvider | null {
  const row = sqlite
    .query<ProviderRow, [string]>(
      `SELECT *
       FROM halo_model_providers
       WHERE id = ?
       LIMIT 1`,
    )
    .get(id);
  return row ? mapProvider(row, true) : null;
}

export function saveHaloProvider(
  sqlite: Database,
  input: {
    apiKey: string;
    baseUrl: string;
    headers?: Record<string, string>;
    id?: string;
    model: string;
    name: string;
    providerType: HaloProviderType;
  },
): HaloModelProvider {
  const now = Date.now();
  const id = input.id ?? crypto.randomUUID();
  const existing = input.id ? getHaloProvider(sqlite, input.id) : null;
  sqlite
    .query(
      `INSERT INTO halo_model_providers (
        id, name, provider_type, base_url, api_key, model, headers_json,
        last_status, last_error, last_tested_at, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        name = excluded.name,
        provider_type = excluded.provider_type,
        base_url = excluded.base_url,
        api_key = excluded.api_key,
        model = excluded.model,
        headers_json = excluded.headers_json,
        updated_at = excluded.updated_at`,
    )
    .run(
      id,
      input.name,
      input.providerType,
      normalizeBaseUrl(input.baseUrl),
      input.apiKey,
      input.model,
      JSON.stringify(input.headers ?? {}),
      existing?.lastStatus ?? "unknown",
      existing?.lastError ?? null,
      existing?.lastTestedAt ? Date.parse(existing.lastTestedAt) : null,
      existing ? Date.parse(existing.createdAt) : now,
      now,
    );
  const saved = getHaloProvider(sqlite, id);
  if (!saved) throw new Error("Failed to save provider");
  return maskProvider(saved);
}

export function deleteHaloProvider(sqlite: Database, id: string) {
  sqlite.query("DELETE FROM halo_model_providers WHERE id = ?").run(id);
}

export function updateHaloProviderTestStatus(
  sqlite: Database,
  id: string,
  patch: { status: "connected" | "error"; error?: string | null },
) {
  sqlite
    .query(
      `UPDATE halo_model_providers
       SET last_status = ?, last_error = ?, last_tested_at = ?, updated_at = ?
       WHERE id = ?`,
    )
    .run(patch.status, patch.error ?? null, Date.now(), Date.now(), id);
  const provider = getHaloProvider(sqlite, id);
  if (!provider) throw new Error("Provider not found");
  return maskProvider(provider);
}

export function createHaloRun(
  sqlite: Database,
  input: {
    filters: TelemetryFilters;
    maxDepth: number;
    maxParallel: number;
    maxTurns: number;
    model: string;
    prompt: string;
    providerId: string;
    providerName: string;
    targetType: HaloRunTargetType;
    title: string;
  },
): HaloRun {
  const now = Date.now();
  const id = crypto.randomUUID();
  sqlite
    .query(
      `INSERT INTO halo_runs (
        id, title, status, target_type, filters_json, provider_id,
        provider_name, model, prompt, max_depth, max_turns, max_parallel,
        created_at, updated_at
      ) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      id,
      input.title,
      input.targetType,
      JSON.stringify(input.filters),
      input.providerId,
      input.providerName,
      input.model,
      input.prompt,
      input.maxDepth,
      input.maxTurns,
      input.maxParallel,
      now,
      now,
    );
  const run = getHaloRun(sqlite, id);
  if (!run) throw new Error("Failed to create HALO run");
  return run;
}

export function updateHaloRun(
  sqlite: Database,
  id: string,
  patch: Partial<{
    bunqueueJobId: string | null;
    errorMessage: string | null;
    exportPath: string | null;
    finalAnswer: string | null;
    finalAnswerSource: string | null;
    finishedAt: number | null;
    progress: number;
    resultPath: string | null;
    sessionCount: number;
    spanCount: number;
    startedAt: number | null;
    status: HaloRunStatus;
    traceCount: number;
  }>,
): HaloRun {
  const sets = ["updated_at = :updatedAt"];
  const params: Record<string, string | number | null> = {
    id,
    updatedAt: Date.now(),
  };
  const add = (column: string, key: keyof typeof patch) => {
    if (!(key in patch)) return;
    sets.push(`${column} = :${String(key)}`);
    params[String(key)] = patch[key] ?? null;
  };
  add("bunqueue_job_id", "bunqueueJobId");
  add("status", "status");
  add("trace_count", "traceCount");
  add("session_count", "sessionCount");
  add("span_count", "spanCount");
  add("progress", "progress");
  add("export_path", "exportPath");
  add("result_path", "resultPath");
  add("final_answer", "finalAnswer");
  add("final_answer_source", "finalAnswerSource");
  add("error_message", "errorMessage");
  add("started_at", "startedAt");
  add("finished_at", "finishedAt");

  sqlite
    .query(`UPDATE halo_runs SET ${sets.join(", ")} WHERE id = :id`)
    .run(params);
  const run = getHaloRun(sqlite, id);
  if (!run) throw new Error("HALO run not found");
  return run;
}

export function getHaloRun(sqlite: Database, id: string): HaloRun | null {
  const row = sqlite
    .query<RunRow, [string]>(
      `SELECT *
       FROM halo_runs
       WHERE id = ?
       LIMIT 1`,
    )
    .get(id);
  return row ? mapRun(row) : null;
}

export function listHaloRuns(sqlite: Database, limit = 50): HaloRun[] {
  return sqlite
    .query<RunRow, [number]>(
      `SELECT *
       FROM halo_runs
       ORDER BY updated_at DESC
       LIMIT ?`,
    )
    .all(limit)
    .map(mapRun);
}

export function markInterruptedHaloRuns(sqlite: Database) {
  const now = Date.now();
  sqlite
    .query(
      `UPDATE halo_runs
       SET status = 'interrupted',
           error_message = 'The app stopped before this HALO run finished.',
           finished_at = ?,
           updated_at = ?
       WHERE status IN ('queued', 'exporting', 'running')`,
    )
    .run(now, now);
}

export function isHaloRunCancelled(sqlite: Database, id: string) {
  const status = sqlite
    .query<{ status: string }, [string]>(
      `SELECT status FROM halo_runs WHERE id = ? LIMIT 1`,
    )
    .get(id)?.status;
  return status === "cancelled";
}

export function addHaloRunEvent(
  sqlite: Database,
  input: {
    eventType: string;
    payload: Record<string, unknown>;
    runId: string;
  },
): HaloRunEvent {
  const sequence =
    (sqlite
      .query<{ value: number | null }, [string]>(
        `SELECT max(sequence) AS value
         FROM halo_run_events
         WHERE run_id = ?`,
      )
      .get(input.runId)?.value ?? 0) + 1;
  const createdAt = Date.now();
  sqlite
    .query(
      `INSERT INTO halo_run_events (
        run_id, sequence, event_type, payload_json, created_at
      ) VALUES (?, ?, ?, ?, ?)`,
    )
    .run(
      input.runId,
      sequence,
      input.eventType,
      JSON.stringify(input.payload),
      createdAt,
    );
  const row = sqlite
    .query<EventRow, []>(
      `SELECT *
       FROM halo_run_events
       WHERE id = last_insert_rowid()
       LIMIT 1`,
    )
    .get();
  if (!row) throw new Error("Failed to save HALO event");
  return mapEvent(row);
}

export function listHaloRunEvents(
  sqlite: Database,
  runId: string,
  limit = 500,
): HaloRunEvent[] {
  return sqlite
    .query<EventRow, [string, number]>(
      `SELECT *
       FROM halo_run_events
       WHERE run_id = ?
       ORDER BY sequence ASC
       LIMIT ?`,
    )
    .all(runId, limit)
    .map(mapEvent);
}

export function publishHaloRun(live: LiveEventStore, run: HaloRun) {
  live.publish({
    eventType: "halo.run.updated",
    payload: {
      run: haloRunToSnapshot(run),
      type: "halo.run.updated",
    },
  });
}

export function publishHaloRunEvent(
  live: LiveEventStore,
  run: HaloRun,
  event: HaloRunEvent,
) {
  live.publish({
    eventType:
      event.eventType === "completed"
        ? "halo.run.completed"
        : event.eventType === "failed"
          ? "halo.run.failed"
          : "halo.run.event",
    payload: {
      event,
      run: haloRunToSnapshot(run),
      type:
        event.eventType === "completed"
          ? "halo.run.completed"
          : event.eventType === "failed"
            ? "halo.run.failed"
            : "halo.run.event",
    },
  });
}

function haloRunToSnapshot(run: HaloRun): HaloRunSnapshot {
  return {
    errorMessage: run.errorMessage,
    finalAnswer: run.finalAnswer,
    finishedAt: run.finishedAt,
    id: run.id,
    model: run.model,
    progress: run.progress,
    providerName: run.providerName,
    sessionCount: run.sessionCount,
    spanCount: run.spanCount,
    startedAt: run.startedAt,
    status: run.status,
    targetType: run.targetType,
    title: run.title,
    traceCount: run.traceCount,
    updatedAt: run.updatedAt,
  };
}

function mapProvider<T extends boolean>(
  row: ProviderRow,
  includeSecret: T,
): T extends true ? StoredHaloModelProvider : HaloModelProvider {
  const provider = {
    apiKeyMasked: maskSecret(row.api_key),
    baseUrl: row.base_url,
    createdAt: isoFromMs(row.created_at),
    headers: parseJson(row.headers_json, {}),
    id: row.id,
    lastError: row.last_error,
    lastStatus: row.last_status,
    lastTestedAt: row.last_tested_at ? isoFromMs(row.last_tested_at) : null,
    model: row.model,
    name: row.name,
    providerType: row.provider_type,
    updatedAt: isoFromMs(row.updated_at),
  };
  return (
    includeSecret ? { ...provider, apiKey: row.api_key } : provider
  ) as T extends true ? StoredHaloModelProvider : HaloModelProvider;
}

function maskProvider(provider: StoredHaloModelProvider): HaloModelProvider {
  const { apiKey: _apiKey, ...safe } = provider;
  return safe;
}

function mapRun(row: RunRow): HaloRun {
  return {
    bunqueueJobId: row.bunqueue_job_id,
    createdAt: isoFromMs(row.created_at),
    errorMessage: row.error_message,
    exportPath: row.export_path,
    filters: parseJson<TelemetryFilters>(row.filters_json, {}),
    finalAnswer: row.final_answer,
    finalAnswerSource: row.final_answer_source,
    finishedAt: row.finished_at ? isoFromMs(row.finished_at) : null,
    id: row.id,
    maxDepth: row.max_depth,
    maxParallel: row.max_parallel,
    maxTurns: row.max_turns,
    model: row.model,
    progress: row.progress,
    prompt: row.prompt,
    providerId: row.provider_id,
    providerName: row.provider_name,
    resultPath: row.result_path,
    sessionCount: row.session_count,
    spanCount: row.span_count,
    startedAt: row.started_at ? isoFromMs(row.started_at) : null,
    status: row.status,
    targetType: row.target_type,
    title: row.title,
    traceCount: row.trace_count,
    updatedAt: isoFromMs(row.updated_at),
  };
}

function mapEvent(row: EventRow): HaloRunEvent {
  return {
    createdAt: isoFromMs(row.created_at),
    eventType: row.event_type,
    id: row.id,
    payload: parseJson(row.payload_json, {}),
    runId: row.run_id,
    sequence: row.sequence,
  };
}

function normalizeBaseUrl(value: string) {
  return value.trim().replace(/\/+$/, "");
}

function maskSecret(value: string) {
  if (value.length <= 8) return "••••";
  return `${value.slice(0, 4)}…${value.slice(-4)}`;
}

function parseJson<T>(raw: string, fallback: T): T {
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function isoFromMs(ms: number): string {
  return new Date(ms).toISOString();
}
