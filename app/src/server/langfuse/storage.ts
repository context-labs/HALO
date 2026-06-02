import type { Database } from "bun:sqlite";
import type { ImportJobSnapshot, LiveEventStore } from "../live/events";
import type {
  LangfuseConnection,
  LangfuseDiscovery,
  LangfuseImportJob,
  LangfuseImportStatus,
  LangfuseTraceFilters,
  StoredLangfuseConnection,
} from "./types";

type ConnectionRow = {
  id: string;
  name: string;
  base_url: string;
  public_key: string;
  secret_key: string;
  project_id: string | null;
  project_name: string | null;
  organization_id: string | null;
  organization_name: string | null;
  discovered_facets_json: string;
  last_status: string;
  last_error: string | null;
  last_connected_at: number | null;
  created_at: number;
  updated_at: number;
};

type JobRow = {
  id: string;
  connection_id: string;
  connection_name: string | null;
  bunqueue_job_id: string | null;
  status: LangfuseImportStatus;
  filters_json: string;
  progress: number;
  total_traces: number;
  imported_traces: number;
  total_observations: number;
  imported_observations: number;
  failed_traces: number;
  error_message: string | null;
  current_trace_id: string | null;
  current_trace_name: string | null;
  created_at: number;
  updated_at: number;
  started_at: number | null;
  finished_at: number | null;
};

export function listLangfuseConnections(sqlite: Database): LangfuseConnection[] {
  return sqlite
    .query<ConnectionRow, []>(
      `SELECT *
       FROM langfuse_connections
       ORDER BY updated_at DESC`,
    )
    .all()
    .map((row) => mapConnection(row, false));
}

export function getLangfuseConnection(
  sqlite: Database,
  id: string,
): StoredLangfuseConnection | null {
  const row = sqlite
    .query<ConnectionRow, [string]>(
      `SELECT *
       FROM langfuse_connections
       WHERE id = ?
       LIMIT 1`,
    )
    .get(id);
  return row ? mapConnection(row, true) : null;
}

export function saveLangfuseConnection(
  sqlite: Database,
  input: {
    baseUrl: string;
    discovery: LangfuseDiscovery;
    id?: string;
    name: string;
    publicKey: string;
    secretKey: string;
  },
): LangfuseConnection {
  const now = Date.now();
  const id =
    input.id ??
    findLangfuseConnectionIdByKey(sqlite, input.baseUrl, input.publicKey) ??
    crypto.randomUUID();
  const project = input.discovery.project;
  const existing = getLangfuseConnection(sqlite, id);

  sqlite
    .query(
      `INSERT INTO langfuse_connections (
        id, name, base_url, public_key, secret_key, project_id, project_name,
        organization_id, organization_name, discovered_facets_json, last_status,
        last_error, last_connected_at, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'connected', NULL, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        name = excluded.name,
        base_url = excluded.base_url,
        public_key = excluded.public_key,
        secret_key = excluded.secret_key,
        project_id = excluded.project_id,
        project_name = excluded.project_name,
        organization_id = excluded.organization_id,
        organization_name = excluded.organization_name,
        discovered_facets_json = excluded.discovered_facets_json,
        last_status = excluded.last_status,
        last_error = excluded.last_error,
        last_connected_at = excluded.last_connected_at,
        updated_at = excluded.updated_at`,
    )
    .run(
      id,
      input.name,
      input.baseUrl,
      input.publicKey,
      input.secretKey,
      project?.id ?? null,
      project?.name ?? null,
      project?.organization?.id ?? null,
      project?.organization?.name ?? null,
      JSON.stringify(input.discovery.facets),
      now,
      existing ? Date.parse(existing.createdAt) : now,
      now,
    );

  const saved = getLangfuseConnection(sqlite, id);
  if (!saved) throw new Error("Failed to save Langfuse connection");
  return saved;
}

function findLangfuseConnectionIdByKey(
  sqlite: Database,
  baseUrl: string,
  publicKey: string,
): string | null {
  return (
    sqlite
      .query<{ id: string }, [string, string]>(
        `SELECT id
         FROM langfuse_connections
         WHERE base_url = ? AND public_key = ?
         LIMIT 1`,
      )
      .get(baseUrl, publicKey)?.id ?? null
  );
}

export function markLangfuseConnectionError(
  sqlite: Database,
  input: { id?: string; baseUrl?: string; publicKey?: string; error: string },
) {
  const now = Date.now();
  if (input.id) {
    sqlite
      .query(
        `UPDATE langfuse_connections
         SET last_status = 'error', last_error = ?, updated_at = ?
         WHERE id = ?`,
      )
      .run(input.error, now, input.id);
    return;
  }
  if (input.baseUrl && input.publicKey) {
    sqlite
      .query(
        `UPDATE langfuse_connections
         SET last_status = 'error', last_error = ?, updated_at = ?
         WHERE base_url = ? AND public_key = ?`,
      )
      .run(input.error, now, input.baseUrl, input.publicKey);
  }
}

export function deleteLangfuseConnection(sqlite: Database, id: string) {
  sqlite.query("DELETE FROM langfuse_connections WHERE id = ?").run(id);
}

export function createLangfuseImportJob(
  sqlite: Database,
  input: { connectionId: string; filters: LangfuseTraceFilters },
): LangfuseImportJob {
  const now = Date.now();
  const id = crypto.randomUUID();
  sqlite
    .query(
      `INSERT INTO langfuse_import_jobs (
        id, connection_id, status, filters_json, progress, created_at, updated_at
      ) VALUES (?, ?, 'queued', ?, 0, ?, ?)`,
    )
    .run(id, input.connectionId, JSON.stringify(input.filters), now, now);
  const job = getLangfuseImportJob(sqlite, id);
  if (!job) throw new Error("Failed to create Langfuse import job");
  return job;
}

export function updateLangfuseImportJob(
  sqlite: Database,
  id: string,
  patch: Partial<{
    bunqueueJobId: string | null;
    currentTraceId: string | null;
    currentTraceName: string | null;
    errorMessage: string | null;
    failedTraces: number;
    finishedAt: number | null;
    importedObservations: number;
    importedTraces: number;
    progress: number;
    startedAt: number | null;
    status: LangfuseImportStatus;
    totalObservations: number;
    totalTraces: number;
  }>,
): LangfuseImportJob {
  const sets: string[] = ["updated_at = :updatedAt"];
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
  add("progress", "progress");
  add("total_traces", "totalTraces");
  add("imported_traces", "importedTraces");
  add("total_observations", "totalObservations");
  add("imported_observations", "importedObservations");
  add("failed_traces", "failedTraces");
  add("error_message", "errorMessage");
  add("current_trace_id", "currentTraceId");
  add("current_trace_name", "currentTraceName");
  add("started_at", "startedAt");
  add("finished_at", "finishedAt");

  sqlite
    .query(
      `UPDATE langfuse_import_jobs
       SET ${sets.join(", ")}
       WHERE id = :id`,
    )
    .run(params);

  const job = getLangfuseImportJob(sqlite, id);
  if (!job) throw new Error("Langfuse import job not found");
  return job;
}

export function getLangfuseImportJob(
  sqlite: Database,
  id: string,
): LangfuseImportJob | null {
  const row = sqlite
    .query<JobRow, [string]>(
      `SELECT
        j.*,
        c.name AS connection_name
       FROM langfuse_import_jobs j
       LEFT JOIN langfuse_connections c ON c.id = j.connection_id
       WHERE j.id = ?
       LIMIT 1`,
    )
    .get(id);
  return row ? mapImportJob(row) : null;
}

export function listLangfuseImportJobs(
  sqlite: Database,
  limit = 20,
): LangfuseImportJob[] {
  return sqlite
    .query<JobRow, [number]>(
      `SELECT
        j.*,
        c.name AS connection_name
       FROM langfuse_import_jobs j
       LEFT JOIN langfuse_connections c ON c.id = j.connection_id
       ORDER BY j.updated_at DESC
       LIMIT ?`,
    )
    .all(limit)
    .map(mapImportJob);
}

export function markInterruptedLangfuseImports(sqlite: Database) {
  const now = Date.now();
  sqlite
    .query(
      `UPDATE langfuse_import_jobs
       SET status = 'interrupted',
           error_message = 'The app stopped before this import finished.',
           finished_at = ?,
           updated_at = ?
       WHERE status IN ('queued', 'running')`,
    )
    .run(now, now);
}

export function isLangfuseImportCancelled(sqlite: Database, id: string) {
  const row = sqlite
    .query<{ status: string }, [string]>(
      `SELECT status
       FROM langfuse_import_jobs
       WHERE id = ?
       LIMIT 1`,
    )
    .get(id);
  return row?.status === "cancelled";
}

export function publishLangfuseImportJob(
  live: LiveEventStore,
  job: LangfuseImportJob,
) {
  live.publish({
    eventType: "import.job.updated",
    payload: {
      job: importJobToSnapshot(job),
      type: "import.job.updated",
    },
  });
}

function importJobToSnapshot(job: LangfuseImportJob): ImportJobSnapshot {
  return {
    bunqueueJobId: job.bunqueueJobId,
    connectionId: job.connectionId,
    connectionName: job.connectionName,
    currentTraceId: job.currentTraceId,
    currentTraceName: job.currentTraceName,
    errorMessage: job.errorMessage,
    failedTraces: job.failedTraces,
    finishedAt: job.finishedAt,
    id: job.id,
    importedObservations: job.importedObservations,
    importedTraces: job.importedTraces,
    progress: job.progress,
    startedAt: job.startedAt,
    status: job.status,
    totalObservations: job.totalObservations,
    totalTraces: job.totalTraces,
    updatedAt: job.updatedAt,
  };
}

function mapConnection<T extends boolean>(
  row: ConnectionRow,
  includeSecret: T,
): T extends true ? StoredLangfuseConnection : LangfuseConnection {
  const connection = {
    baseUrl: row.base_url,
    createdAt: isoFromMs(row.created_at),
    discoveredFacets: parseJson(row.discovered_facets_json, {}),
    id: row.id,
    lastConnectedAt: row.last_connected_at ? isoFromMs(row.last_connected_at) : null,
    lastError: row.last_error,
    lastStatus: row.last_status,
    name: row.name,
    organizationId: row.organization_id,
    organizationName: row.organization_name,
    projectId: row.project_id,
    projectName: row.project_name,
    publicKey: row.public_key,
    updatedAt: isoFromMs(row.updated_at),
  };
  return (
    includeSecret ? { ...connection, secretKey: row.secret_key } : connection
  ) as T extends true ? StoredLangfuseConnection : LangfuseConnection;
}

function mapImportJob(row: JobRow): LangfuseImportJob {
  return {
    bunqueueJobId: row.bunqueue_job_id,
    connectionId: row.connection_id,
    connectionName: row.connection_name,
    createdAt: isoFromMs(row.created_at),
    currentTraceId: row.current_trace_id,
    currentTraceName: row.current_trace_name,
    errorMessage: row.error_message,
    failedTraces: row.failed_traces,
    filters: parseJson(row.filters_json, {}),
    finishedAt: row.finished_at ? isoFromMs(row.finished_at) : null,
    id: row.id,
    importedObservations: row.imported_observations,
    importedTraces: row.imported_traces,
    progress: row.progress,
    startedAt: row.started_at ? isoFromMs(row.started_at) : null,
    status: row.status,
    totalObservations: row.total_observations,
    totalTraces: row.total_traces,
    updatedAt: isoFromMs(row.updated_at),
  };
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
