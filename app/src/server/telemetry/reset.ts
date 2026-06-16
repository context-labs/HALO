import { existsSync } from "node:fs";
import { rm } from "node:fs/promises";
import { dirname, resolve, sep } from "node:path";
import type { Database } from "bun:sqlite";
import type { DatabaseHandle } from "../db/client";
import { defaultHaloInstallPath } from "../halo/storage";
import { clearTelemetryData, type ClearTelemetryDataResult } from "./storage";

export type FactoryResetResult = {
  appSettingCount: number;
  deletedPathCount: number;
  engineSettingCount: number;
  failedPathCount: number;
  fileImportJobCount: number;
  haloProviderCount: number;
  haloRunArtifactCount: number;
  haloRunCount: number;
  haloRunEventCount: number;
  haloRunTurnCount: number;
  langfuseConnectionCount: number;
  langfuseImportJobCount: number;
  phoenixConnectionCount: number;
  phoenixImportJobCount: number;
  skippedPathCount: number;
  telemetry: ClearTelemetryDataResult;
};

const RESET_TABLES = [
  "halo_run_events",
  "halo_run_artifacts",
  "halo_run_turns",
  "halo_runs",
  "halo_model_providers",
  "halo_engine_settings",
  "file_import_jobs",
  "langfuse_import_jobs",
  "langfuse_connections",
  "phoenix_import_jobs",
  "phoenix_connections",
  "app_settings",
];

const AUTOINCREMENT_TABLES = [
  "halo_run_events",
  "ingest_batches",
  "live_events",
  "spans",
  "trace_summaries",
];

export async function factoryResetLocalData(
  database: DatabaseHandle,
): Promise<FactoryResetResult> {
  const sqlite = database.sqlite;
  const result: FactoryResetResult = {
    appSettingCount: tableCount(sqlite, "app_settings"),
    deletedPathCount: 0,
    engineSettingCount: tableCount(sqlite, "halo_engine_settings"),
    failedPathCount: 0,
    fileImportJobCount: tableCount(sqlite, "file_import_jobs"),
    haloProviderCount: tableCount(sqlite, "halo_model_providers"),
    haloRunArtifactCount: tableCount(sqlite, "halo_run_artifacts"),
    haloRunCount: tableCount(sqlite, "halo_runs"),
    haloRunEventCount: tableCount(sqlite, "halo_run_events"),
    haloRunTurnCount: tableCount(sqlite, "halo_run_turns"),
    langfuseConnectionCount: tableCount(sqlite, "langfuse_connections"),
    langfuseImportJobCount: tableCount(sqlite, "langfuse_import_jobs"),
    phoenixConnectionCount: tableCount(sqlite, "phoenix_connections"),
    phoenixImportJobCount: tableCount(sqlite, "phoenix_import_jobs"),
    skippedPathCount: 0,
    telemetry: clearTelemetryData(sqlite),
  };

  const cleanup = await removeAppOwnedPaths(database);
  result.deletedPathCount = cleanup.deleted;
  result.failedPathCount = cleanup.failed;
  result.skippedPathCount = cleanup.skipped;

  const transaction = sqlite.transaction(() => {
    for (const table of RESET_TABLES) {
      sqlite.run(`DELETE FROM ${table}`);
    }
    resetAutoincrement(sqlite, AUTOINCREMENT_TABLES);
  });
  transaction();

  return result;
}

async function removeAppOwnedPaths(database: DatabaseHandle) {
  const appDataDir = appDataDirForDatabase(database);
  if (!appDataDir) return { deleted: 0, failed: 0, skipped: 0 };

  const candidates = collectPathCandidates(database, appDataDir);
  let deleted = 0;
  let failed = 0;
  let skipped = 0;

  for (const candidate of candidates) {
    if (!isWithin(appDataDir, candidate)) {
      skipped += 1;
      continue;
    }
    if (!existsSync(candidate)) continue;
    try {
      await rm(candidate, { force: true, recursive: true });
      deleted += 1;
    } catch {
      failed += 1;
    }
  }

  return { deleted, failed, skipped };
}

function collectPathCandidates(database: DatabaseHandle, appDataDir: string) {
  const paths = new Set<string>();
  paths.add(defaultHaloInstallPath(database.path));
  paths.add(resolve(appDataDir, "halo-runs"));

  for (const row of database.sqlite
    .query<{ install_path: string | null }, []>(
      "SELECT install_path FROM halo_engine_settings",
    )
    .all()) {
    if (row.install_path) paths.add(resolve(row.install_path));
  }

  for (const row of database.sqlite
    .query<{ export_path: string | null; result_path: string | null }, []>(
      "SELECT export_path, result_path FROM halo_runs",
    )
    .all()) {
    if (row.export_path) paths.add(resolve(row.export_path));
    if (row.result_path) paths.add(resolve(row.result_path));
  }

  for (const row of database.sqlite
    .query<{ path: string | null }, []>("SELECT path FROM halo_run_artifacts")
    .all()) {
    if (row.path) paths.add(resolve(row.path));
  }

  return [...paths];
}

function appDataDirForDatabase(database: DatabaseHandle) {
  if (database.path === ":memory:") return null;
  return resolve(dirname(database.path));
}

function isWithin(parent: string, child: string) {
  const normalizedParent = resolve(parent);
  const normalizedChild = resolve(child);
  return (
    normalizedChild === normalizedParent ||
    normalizedChild.startsWith(`${normalizedParent}${sep}`)
  );
}

function tableCount(sqlite: Database, tableName: string) {
  const row = sqlite
    .query<{ count: number }, []>(`SELECT count(*) AS count FROM ${tableName}`)
    .get();
  return row?.count ?? 0;
}

function resetAutoincrement(sqlite: Database, tableNames: string[]) {
  try {
    const placeholders = tableNames.map(() => "?").join(", ");
    sqlite
      .query(`DELETE FROM sqlite_sequence WHERE name IN (${placeholders})`)
      .run(...tableNames);
  } catch {
    // sqlite_sequence is created lazily once AUTOINCREMENT tables receive rows.
  }
}
