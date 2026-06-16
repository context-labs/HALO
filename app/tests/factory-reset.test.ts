import { afterEach, describe, expect, test } from "bun:test";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createDatabase,
  ensureSchema,
  type DatabaseHandle,
} from "../src/server/db/client";
import { createLiveEventStore } from "../src/server/live/events";
import { appRouter } from "../src/server/router";
import { ingestTelemetry } from "../src/server/telemetry/storage";
import { makeTracePayload } from "./support/otlp-fixtures";

let tempDir: string | null = null;
let database: DatabaseHandle | null = null;

afterEach(() => {
  database?.sqlite.close(false);
  database = null;
  if (tempDir) {
    rmSync(tempDir, { force: true, recursive: true });
    tempDir = null;
  }
});

describe("factory reset", () => {
  test("removes local records and app-owned HALO files", async () => {
    tempDir = `${tmpdir()}/halo-reset-${crypto.randomUUID()}`;
    mkdirSync(tempDir, { recursive: true });
    const dbPath = join(tempDir, "halo-canvas.sqlite");
    database = createDatabase(dbPath);
    ensureSchema(database.sqlite);

    const engineDir = join(tempDir, "halo-engine");
    const runsDir = join(tempDir, "halo-runs");
    const reportPath = join(runsDir, "run-1", "report.md");
    const artifactPath = join(runsDir, "run-1", "artifact.json");
    mkdirSync(engineDir, { recursive: true });
    mkdirSync(join(runsDir, "run-1"), { recursive: true });
    writeFileSync(join(engineDir, "README.md"), "engine");
    writeFileSync(reportPath, "report");
    writeFileSync(artifactPath, "{}");

    seedNonTelemetryData(database, {
      artifactPath,
      engineDir,
      reportPath,
    });
    ingestTelemetry(database.sqlite, {
      body: JSON.stringify(makeTracePayload()),
      contentEncoding: "identity",
      sizeBytes: 128,
    });

    const caller = appRouter.createCaller({
      database,
      live: createLiveEventStore(database.sqlite),
      liveUrl: "ws://127.0.0.1:8800",
    });
    const result = await caller.telemetry.factoryReset();

    expect(result.telemetry.traceCount).toBe(1);
    expect(result.telemetry.spanCount).toBe(2);
    expect(result.haloProviderCount).toBe(1);
    expect(result.haloRunCount).toBe(1);
    expect(result.langfuseConnectionCount).toBe(1);
    expect(result.phoenixConnectionCount).toBe(1);
    expect(result.fileImportJobCount).toBe(1);
    expect(result.appSettingCount).toBe(1);
    expect(result.deletedPathCount).toBeGreaterThanOrEqual(2);
    expect(result.failedPathCount).toBe(0);

    for (const table of [
      "spans",
      "trace_summaries",
      "span_search_fts",
      "ingest_batches",
      "live_events",
      "halo_run_events",
      "halo_run_artifacts",
      "halo_run_turns",
      "halo_runs",
      "halo_model_providers",
      "halo_engine_settings",
      "langfuse_import_jobs",
      "langfuse_connections",
      "phoenix_import_jobs",
      "phoenix_connections",
      "file_import_jobs",
      "app_settings",
    ]) {
      expect(tableCount(database.sqlite, table), table).toBe(0);
    }

    expect(await Bun.file(engineDir).exists()).toBe(false);
    expect(await Bun.file(reportPath).exists()).toBe(false);
    expect(await Bun.file(artifactPath).exists()).toBe(false);
  });
});

function seedNonTelemetryData(
  database: DatabaseHandle,
  paths: { artifactPath: string; engineDir: string; reportPath: string },
) {
  const sqlite = database.sqlite;
  const now = Date.now();
  sqlite
    .query(
      `INSERT INTO app_settings (key, value, updated_at)
       VALUES ('onboarding.completedAt', ?, ?)`,
    )
    .run(new Date(now).toISOString(), now);
  sqlite
    .query(
      `INSERT INTO halo_engine_settings (
        id, repo_url, install_path, status, status_detail, commit_sha,
        last_error, installed_at, updated_at
      ) VALUES ('default', ?, ?, 'installed', NULL, 'abc123', NULL, ?, ?)`,
    )
    .run("https://github.com/context-labs/HALO", paths.engineDir, now, now);
  sqlite
    .query(
      `INSERT INTO halo_model_providers (
        id, name, provider_type, base_url, api_key, model, headers_json,
        last_status, last_error, last_tested_at, created_at, updated_at
      ) VALUES ('provider-1', 'OpenAI', 'openai', 'https://api.openai.com/v1',
        'sk-test', 'gpt-4.1-mini', '{}', 'ok', NULL, ?, ?, ?)`,
    )
    .run(now, now, now);
  sqlite
    .query(
      `INSERT INTO langfuse_connections (
        id, name, base_url, public_key, secret_key, project_id, project_name,
        organization_id, organization_name, discovered_facets_json, last_status,
        last_error, last_connected_at, created_at, updated_at
      ) VALUES ('langfuse-1', 'Langfuse', 'https://cloud.langfuse.com',
        'pk', 'sk', NULL, NULL, NULL, NULL, '{}', 'ok', NULL, ?, ?, ?)`,
    )
    .run(now, now, now);
  sqlite
    .query(
      `INSERT INTO langfuse_import_jobs (
        id, connection_id, bunqueue_job_id, status, filters_json, progress,
        total_traces, imported_traces, total_observations, imported_observations,
        failed_traces, error_message, current_trace_id, current_trace_name,
        created_at, updated_at, started_at, finished_at
      ) VALUES ('langfuse-job-1', 'langfuse-1', NULL, 'completed', '{}', 100,
        1, 1, 1, 1, 0, NULL, NULL, NULL, ?, ?, ?, ?)`,
    )
    .run(now, now, now, now);
  sqlite
    .query(
      `INSERT INTO phoenix_connections (
        id, name, base_url, api_key, discovered_projects_json, last_status,
        last_error, last_connected_at, created_at, updated_at
      ) VALUES ('phoenix-1', 'Phoenix', 'https://app.phoenix.arize.com',
        'key', '[]', 'ok', NULL, ?, ?, ?)`,
    )
    .run(now, now, now);
  sqlite
    .query(
      `INSERT INTO phoenix_import_jobs (
        id, connection_id, bunqueue_job_id, status, filters_json, progress,
        total_traces, imported_traces, total_observations, imported_observations,
        failed_traces, error_message, current_trace_id, current_trace_name,
        created_at, updated_at, started_at, finished_at
      ) VALUES ('phoenix-job-1', 'phoenix-1', NULL, 'completed', '{}', 100,
        1, 1, 1, 1, 0, NULL, NULL, NULL, ?, ?, ?, ?)`,
    )
    .run(now, now, now, now);
  sqlite
    .query(
      `INSERT INTO file_import_jobs (
        id, bunqueue_job_id, status, file_name, file_path, file_size_bytes,
        progress, total_traces, imported_traces, total_observations,
        imported_observations, failed_traces, skipped_lines, error_message,
        current_trace_id, current_trace_name, created_at, updated_at,
        started_at, finished_at
      ) VALUES ('file-job-1', NULL, 'completed', 'traces.jsonl', '/tmp/traces.jsonl',
        10, 100, 1, 1, 1, 1, 0, 0, NULL, NULL, NULL, ?, ?, ?, ?)`,
    )
    .run(now, now, now, now);
  sqlite
    .query(
      `INSERT INTO halo_runs (
        id, bunqueue_job_id, title, status, target_type, filters_json,
        provider_id, provider_name, model, prompt, max_depth, max_turns,
        max_parallel, trace_count, session_count, span_count, progress,
        export_path, result_path, final_answer, final_answer_source,
        error_message, created_at, updated_at, started_at, finished_at
      ) VALUES ('run-1', NULL, 'Run 1', 'completed', 'traces', '{}',
        'provider-1', 'OpenAI', 'gpt-4.1-mini', 'Find problems', 1, 8,
        2, 1, 0, 2, 100, NULL, ?, 'Done', 'assistant', NULL, ?, ?, ?, ?)`,
    )
    .run(paths.reportPath, now, now, now, now);
  sqlite
    .query(
      `INSERT INTO halo_run_events (
        run_id, sequence, event_type, payload_json, created_at, turn_index
      ) VALUES ('run-1', 1, 'halo.run.completed', '{}', ?, NULL)`,
    )
    .run(now);
  sqlite
    .query(
      `INSERT INTO halo_run_turns (
        id, run_id, turn_index, role, content, status, error_message,
        created_at, finished_at
      ) VALUES ('turn-1', 'run-1', 0, 'assistant', 'Done', 'completed',
        NULL, ?, ?)`,
    )
    .run(now, now);
  sqlite
    .query(
      `INSERT INTO halo_run_artifacts (
        id, run_id, artifact_type, path, size_bytes, created_at
      ) VALUES ('artifact-1', 'run-1', 'json', ?, 2, ?)`,
    )
    .run(paths.artifactPath, now);
}

function tableCount(sqlite: DatabaseHandle["sqlite"], tableName: string) {
  const row = sqlite
    .query<{ count: number }, []>(`SELECT count(*) AS count FROM ${tableName}`)
    .get();
  return row?.count ?? 0;
}
