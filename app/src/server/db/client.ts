import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { Database } from "bun:sqlite";
import { drizzle } from "drizzle-orm/bun-sqlite";
import * as schema from "./schema";

const DEFAULT_DB_PATH = "data/halo-canvas.sqlite";

export function resolveDatabasePath(
  input = process.env.HALO_DB_PATH ?? DEFAULT_DB_PATH,
) {
  return input === ":memory:" ? input : resolve(input);
}

export function createDatabase(inputPath?: string) {
  const path = resolveDatabasePath(inputPath);

  if (path !== ":memory:") {
    mkdirSync(dirname(path), { recursive: true });
  }

  const sqlite = new Database(path, {
    create: true,
    strict: true,
  });

  sqlite.run("PRAGMA foreign_keys = ON;");
  sqlite.run("PRAGMA journal_mode = WAL;");
  sqlite.run("PRAGMA synchronous = NORMAL;");

  const db = drizzle({
    client: sqlite,
    schema,
  });

  return {
    db,
    path,
    sqlite,
  };
}

export function ensureSchema(sqlite: Database) {
  sqlite.run(`
    CREATE TABLE IF NOT EXISTS spans (
      id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
      project_id TEXT NOT NULL,
      team_id TEXT NOT NULL,
      api_key_id TEXT NOT NULL,
      trace_id TEXT NOT NULL,
      span_id TEXT NOT NULL,
      parent_span_id TEXT NOT NULL DEFAULT '',
      trace_state TEXT NOT NULL DEFAULT '',
      start_time INTEGER NOT NULL,
      end_time INTEGER NOT NULL,
      start_time_unix_nano TEXT NOT NULL,
      end_time_unix_nano TEXT NOT NULL,
      duration_ns TEXT NOT NULL,
      duration_ms REAL NOT NULL,
      ingested_at INTEGER NOT NULL,
      span_name TEXT NOT NULL DEFAULT '',
      span_kind TEXT NOT NULL DEFAULT 'SPAN_KIND_UNSPECIFIED',
      service_name TEXT NOT NULL DEFAULT '',
      service_version TEXT NOT NULL DEFAULT '',
      deployment_environment TEXT NOT NULL DEFAULT '',
      scope_name TEXT NOT NULL DEFAULT '',
      scope_version TEXT NOT NULL DEFAULT '',
      status_code TEXT NOT NULL DEFAULT 'STATUS_CODE_UNSET',
      status_message TEXT NOT NULL DEFAULT '',
      observation_kind TEXT NOT NULL DEFAULT 'SPAN',
      llm_provider TEXT NOT NULL DEFAULT '',
      llm_model_name TEXT NOT NULL DEFAULT '',
      llm_response_model TEXT NOT NULL DEFAULT '',
      input_tokens INTEGER,
      output_tokens INTEGER,
      total_tokens INTEGER,
      cache_read_tokens INTEGER,
      cache_write_tokens INTEGER,
      reasoning_tokens INTEGER,
      cost_total REAL,
      cost_input REAL,
      cost_output REAL,
      cost_cache_read REAL,
      cost_cache_write REAL,
      cost_reasoning REAL,
      user_id TEXT,
      session_id TEXT,
      conversation_id TEXT NOT NULL DEFAULT '',
      chat_id TEXT NOT NULL DEFAULT '',
      agent_name TEXT NOT NULL DEFAULT '',
      agent_id TEXT NOT NULL DEFAULT '',
      input_messages TEXT,
      output_messages TEXT,
      input TEXT,
      output TEXT,
      retrieval_documents TEXT,
      resource_attributes TEXT NOT NULL,
      resource_attributes_int TEXT NOT NULL,
      resource_attributes_double TEXT NOT NULL,
      span_attributes TEXT NOT NULL,
      span_attributes_int TEXT NOT NULL,
      span_attributes_double TEXT NOT NULL,
      events_json TEXT NOT NULL,
      links_json TEXT NOT NULL
    );
  `);

  sqlite.run(`
    CREATE TABLE IF NOT EXISTS trace_summaries (
      id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
      project_id TEXT NOT NULL,
      trace_id TEXT NOT NULL,
      session_id TEXT,
      start_time INTEGER NOT NULL,
      end_time INTEGER NOT NULL,
      duration_ns TEXT NOT NULL,
      duration_ms REAL NOT NULL,
      root_span_name TEXT NOT NULL DEFAULT '',
      root_observation_kind TEXT NOT NULL DEFAULT 'SPAN',
      span_count INTEGER NOT NULL,
      llm_span_count INTEGER NOT NULL,
      total_tokens INTEGER,
      cache_read_tokens INTEGER,
      total_cost REAL,
      has_error INTEGER DEFAULT false NOT NULL,
      service_name TEXT NOT NULL DEFAULT '',
      service_version TEXT NOT NULL DEFAULT '',
      deployment_environment TEXT NOT NULL DEFAULT '',
      agent_name TEXT NOT NULL DEFAULT '',
      agent_id TEXT NOT NULL DEFAULT '',
      source TEXT NOT NULL DEFAULT 'local',
      source_trace_id TEXT,
      source_connection_id TEXT,
      source_connection_name TEXT,
      source_import_job_id TEXT,
      source_imported_at INTEGER,
      source_url TEXT,
      source_tags_json TEXT NOT NULL DEFAULT '[]',
      updated_at INTEGER NOT NULL
    );
  `);

  sqlite.run(`
    CREATE TABLE IF NOT EXISTS ingest_batches (
      id TEXT PRIMARY KEY NOT NULL,
      received_at INTEGER NOT NULL,
      size_bytes INTEGER NOT NULL,
      accepted_span_count INTEGER NOT NULL,
      trace_count INTEGER NOT NULL,
      content_encoding TEXT NOT NULL,
      status TEXT NOT NULL,
      error_message TEXT
    );
  `);

  sqlite.run(`
    CREATE TABLE IF NOT EXISTS live_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
      created_at INTEGER NOT NULL,
      trace_id TEXT,
      event_type TEXT NOT NULL,
      payload_json TEXT NOT NULL
    );
  `);

  sqlite.run(`
    CREATE TABLE IF NOT EXISTS langfuse_connections (
      id TEXT PRIMARY KEY NOT NULL,
      name TEXT NOT NULL,
      base_url TEXT NOT NULL,
      public_key TEXT NOT NULL,
      secret_key TEXT NOT NULL,
      project_id TEXT,
      project_name TEXT,
      organization_id TEXT,
      organization_name TEXT,
      discovered_facets_json TEXT NOT NULL DEFAULT '{}',
      last_status TEXT NOT NULL DEFAULT 'unknown',
      last_error TEXT,
      last_connected_at INTEGER,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL
    );
  `);

  sqlite.run(`
    CREATE TABLE IF NOT EXISTS langfuse_import_jobs (
      id TEXT PRIMARY KEY NOT NULL,
      connection_id TEXT NOT NULL,
      bunqueue_job_id TEXT,
      status TEXT NOT NULL,
      filters_json TEXT NOT NULL DEFAULT '{}',
      progress INTEGER NOT NULL DEFAULT 0,
      total_traces INTEGER NOT NULL DEFAULT 0,
      imported_traces INTEGER NOT NULL DEFAULT 0,
      total_observations INTEGER NOT NULL DEFAULT 0,
      imported_observations INTEGER NOT NULL DEFAULT 0,
      failed_traces INTEGER NOT NULL DEFAULT 0,
      error_message TEXT,
      current_trace_id TEXT,
      current_trace_name TEXT,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL,
      started_at INTEGER,
      finished_at INTEGER
    );
  `);

  sqlite.run(`
    CREATE TABLE IF NOT EXISTS halo_engine_settings (
      id TEXT PRIMARY KEY NOT NULL DEFAULT 'default',
      repo_url TEXT NOT NULL DEFAULT 'https://github.com/context-labs/HALO',
      install_path TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'not_installed',
      commit_sha TEXT,
      last_error TEXT,
      installed_at INTEGER,
      updated_at INTEGER NOT NULL
    );
  `);

  sqlite.run(`
    CREATE TABLE IF NOT EXISTS halo_model_providers (
      id TEXT PRIMARY KEY NOT NULL,
      name TEXT NOT NULL,
      provider_type TEXT NOT NULL,
      base_url TEXT NOT NULL,
      api_key TEXT NOT NULL,
      model TEXT NOT NULL,
      headers_json TEXT NOT NULL DEFAULT '{}',
      last_status TEXT NOT NULL DEFAULT 'unknown',
      last_error TEXT,
      last_tested_at INTEGER,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL
    );
  `);

  sqlite.run(`
    CREATE TABLE IF NOT EXISTS halo_runs (
      id TEXT PRIMARY KEY NOT NULL,
      bunqueue_job_id TEXT,
      title TEXT NOT NULL,
      status TEXT NOT NULL,
      target_type TEXT NOT NULL,
      filters_json TEXT NOT NULL DEFAULT '{}',
      provider_id TEXT,
      provider_name TEXT NOT NULL DEFAULT '',
      model TEXT NOT NULL DEFAULT '',
      prompt TEXT NOT NULL,
      max_depth INTEGER NOT NULL DEFAULT 1,
      max_turns INTEGER NOT NULL DEFAULT 8,
      max_parallel INTEGER NOT NULL DEFAULT 2,
      trace_count INTEGER NOT NULL DEFAULT 0,
      session_count INTEGER NOT NULL DEFAULT 0,
      span_count INTEGER NOT NULL DEFAULT 0,
      progress INTEGER NOT NULL DEFAULT 0,
      export_path TEXT,
      result_path TEXT,
      final_answer TEXT,
      final_answer_source TEXT,
      error_message TEXT,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL,
      started_at INTEGER,
      finished_at INTEGER
    );
  `);

  sqlite.run(`
    CREATE TABLE IF NOT EXISTS halo_run_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
      run_id TEXT NOT NULL,
      sequence INTEGER NOT NULL,
      event_type TEXT NOT NULL,
      payload_json TEXT NOT NULL,
      created_at INTEGER NOT NULL
    );
  `);

  sqlite.run(`
    CREATE TABLE IF NOT EXISTS halo_run_artifacts (
      id TEXT PRIMARY KEY NOT NULL,
      run_id TEXT NOT NULL,
      artifact_type TEXT NOT NULL,
      path TEXT NOT NULL,
      size_bytes INTEGER NOT NULL DEFAULT 0,
      created_at INTEGER NOT NULL
    );
  `);

  sqlite.run(`
    CREATE VIRTUAL TABLE IF NOT EXISTS span_search_fts USING fts5(
      project_id UNINDEXED,
      trace_id UNINDEXED,
      span_id UNINDEXED,
      content
    );
  `);

  ensureColumn(sqlite, "trace_summaries", "source", "TEXT NOT NULL DEFAULT 'local'");
  ensureColumn(sqlite, "trace_summaries", "source_trace_id", "TEXT");
  ensureColumn(sqlite, "trace_summaries", "source_connection_id", "TEXT");
  ensureColumn(sqlite, "trace_summaries", "source_connection_name", "TEXT");
  ensureColumn(sqlite, "trace_summaries", "source_import_job_id", "TEXT");
  ensureColumn(sqlite, "trace_summaries", "source_imported_at", "INTEGER");
  ensureColumn(sqlite, "trace_summaries", "source_url", "TEXT");
  ensureColumn(
    sqlite,
    "trace_summaries",
    "source_tags_json",
    "TEXT NOT NULL DEFAULT '[]'",
  );

  const indexes = [
    "CREATE UNIQUE INDEX IF NOT EXISTS spans_project_trace_span_uidx ON spans(project_id, trace_id, span_id)",
    "CREATE INDEX IF NOT EXISTS spans_project_start_idx ON spans(project_id, start_time)",
    "CREATE INDEX IF NOT EXISTS spans_project_trace_idx ON spans(project_id, trace_id)",
    "CREATE INDEX IF NOT EXISTS spans_project_kind_idx ON spans(project_id, observation_kind)",
    "CREATE INDEX IF NOT EXISTS spans_project_status_idx ON spans(project_id, status_code)",
    "CREATE INDEX IF NOT EXISTS spans_project_model_idx ON spans(project_id, llm_model_name)",
    "CREATE INDEX IF NOT EXISTS spans_project_agent_idx ON spans(project_id, agent_name)",
    "CREATE INDEX IF NOT EXISTS spans_project_session_idx ON spans(project_id, session_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS trace_summaries_project_trace_uidx ON trace_summaries(project_id, trace_id)",
    "CREATE INDEX IF NOT EXISTS trace_summaries_project_start_idx ON trace_summaries(project_id, start_time)",
    "CREATE INDEX IF NOT EXISTS trace_summaries_project_status_idx ON trace_summaries(project_id, has_error)",
    "CREATE INDEX IF NOT EXISTS trace_summaries_project_agent_idx ON trace_summaries(project_id, agent_name)",
    "CREATE INDEX IF NOT EXISTS trace_summaries_project_session_idx ON trace_summaries(project_id, session_id)",
    "CREATE INDEX IF NOT EXISTS trace_summaries_project_source_idx ON trace_summaries(project_id, source)",
    "CREATE INDEX IF NOT EXISTS ingest_batches_received_at_idx ON ingest_batches(received_at)",
    "CREATE INDEX IF NOT EXISTS live_events_created_at_idx ON live_events(created_at)",
    "CREATE INDEX IF NOT EXISTS live_events_trace_id_idx ON live_events(trace_id, id)",
    "CREATE INDEX IF NOT EXISTS live_events_event_type_idx ON live_events(event_type, id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS langfuse_connections_base_public_uidx ON langfuse_connections(base_url, public_key)",
    "CREATE INDEX IF NOT EXISTS langfuse_connections_updated_at_idx ON langfuse_connections(updated_at)",
    "CREATE INDEX IF NOT EXISTS langfuse_import_jobs_connection_idx ON langfuse_import_jobs(connection_id)",
    "CREATE INDEX IF NOT EXISTS langfuse_import_jobs_status_idx ON langfuse_import_jobs(status, updated_at)",
    "CREATE INDEX IF NOT EXISTS langfuse_import_jobs_updated_at_idx ON langfuse_import_jobs(updated_at)",
    "CREATE INDEX IF NOT EXISTS halo_model_providers_updated_at_idx ON halo_model_providers(updated_at)",
    "CREATE INDEX IF NOT EXISTS halo_runs_status_idx ON halo_runs(status, updated_at)",
    "CREATE INDEX IF NOT EXISTS halo_runs_updated_at_idx ON halo_runs(updated_at)",
    "CREATE INDEX IF NOT EXISTS halo_run_events_run_idx ON halo_run_events(run_id, sequence)",
    "CREATE INDEX IF NOT EXISTS halo_run_events_created_at_idx ON halo_run_events(created_at)",
    "CREATE INDEX IF NOT EXISTS halo_run_artifacts_run_idx ON halo_run_artifacts(run_id)",
  ];

  for (const sql of indexes) {
    sqlite.run(sql);
  }

  reconcileTraceSummarySources(sqlite);
  reconcileLangfuseResourceMetadata(sqlite);
}

export type DatabaseHandle = ReturnType<typeof createDatabase>;
export type AppDatabase = DatabaseHandle["db"];

function ensureColumn(
  sqlite: Database,
  tableName: string,
  columnName: string,
  definition: string,
) {
  const columns = sqlite
    .query<{ name: string }, []>(`PRAGMA table_info(${tableName})`)
    .all();
  if (columns.some((column) => column.name === columnName)) return;
  sqlite.run(`ALTER TABLE ${tableName} ADD COLUMN ${columnName} ${definition}`);
}

function reconcileTraceSummarySources(sqlite: Database) {
  sqlite.run(`
    UPDATE trace_summaries
    SET
      source = 'local',
      source_trace_id = NULL,
      source_connection_id = NULL,
      source_connection_name = NULL,
      source_import_job_id = NULL,
      source_imported_at = NULL,
      source_url = NULL,
      source_tags_json = '[]'
    WHERE EXISTS (
      SELECT 1
      FROM spans s
      WHERE s.project_id = trace_summaries.project_id
        AND s.trace_id = trace_summaries.trace_id
        AND s.parent_span_id = ''
        AND s.span_attributes NOT LIKE '%"halo.source":"langfuse"%'
    )
    OR EXISTS (
      SELECT 1
      FROM live_events le
      WHERE le.trace_id = trace_summaries.trace_id
        AND le.event_type = 'trace.upserted'
        AND le.payload_json LIKE '%"source":"local"%'
    );
  `);
}

function reconcileLangfuseResourceMetadata(sqlite: Database) {
  const rootRows = sqlite
    .query<{
      deployment_environment: string | null;
      project_id: string;
      resource_attributes: string;
      service_name: string | null;
      service_version: string | null;
      span_attributes: string;
      trace_id: string;
    }, []>(
      `SELECT
        project_id,
        trace_id,
        service_name,
        service_version,
        deployment_environment,
        resource_attributes,
        span_attributes
       FROM spans
       WHERE parent_span_id = ''
         AND span_attributes LIKE '%"halo.source":"langfuse"%'
         AND span_attributes LIKE '%"langfuse.trace.metadata"%'`,
    )
    .all();

  if (rootRows.length === 0) return;

  const updateSpans = sqlite.query<
    unknown,
    [string, string, string, string, string, string]
  >(
    `UPDATE spans
     SET
       service_name = ?,
       service_version = ?,
       deployment_environment = ?,
       resource_attributes = ?
     WHERE project_id = ? AND trace_id = ?`,
  );
  const updateSummary = sqlite.query<
    unknown,
    [string, string, string, string, string]
  >(
    `UPDATE trace_summaries
     SET
       service_name = ?,
       service_version = ?,
       deployment_environment = ?
     WHERE project_id = ? AND trace_id = ? AND source = 'langfuse'`,
  );

  const transaction = sqlite.transaction(() => {
    for (const row of rootRows) {
      const resourceMetadata = langfuseResourceMetadata(row.span_attributes);
      const serviceName = firstStringValue(
        resourceMetadata["service.name"],
        row.service_name,
      );
      if (!serviceName) continue;

      const serviceVersion =
        firstStringValue(resourceMetadata["service.version"], row.service_version) ?? "";
      const deploymentEnvironment =
        firstStringValue(
          resourceMetadata["deployment.environment"],
          row.deployment_environment,
        ) ?? "";
      const resourceAttributes = parseJsonRecord(row.resource_attributes);
      resourceAttributes["service.name"] = serviceName;
      if (serviceVersion) resourceAttributes["service.version"] = serviceVersion;
      if (deploymentEnvironment) {
        resourceAttributes["deployment.environment"] = deploymentEnvironment;
      }

      updateSpans.run(
        serviceName,
        serviceVersion,
        deploymentEnvironment,
        JSON.stringify(resourceAttributes),
        row.project_id,
        row.trace_id,
      );
      updateSummary.run(
        serviceName,
        serviceVersion,
        deploymentEnvironment,
        row.project_id,
        row.trace_id,
      );
    }
  });

  transaction();
}

function langfuseResourceMetadata(
  spanAttributesJson: string,
): Record<string, unknown> {
  const spanAttributes = parseJsonRecord(spanAttributesJson);
  const metadata = parseJsonRecord(spanAttributes["langfuse.trace.metadata"]);
  return parseObjectRecord(metadata.resourceAttributes);
}

function parseJsonRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  if (typeof value !== "string" || !value.trim()) return {};
  try {
    return parseObjectRecord(JSON.parse(value));
  } catch {
    return {};
  }
}

function parseObjectRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function firstStringValue(...values: unknown[]): string | undefined {
  for (const value of values) {
    if (typeof value !== "string") continue;
    const trimmed = value.trim();
    if (trimmed) return trimmed;
  }
  return undefined;
}
