import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { defineConfig } from "drizzle-kit";

const dbPath = process.env.HALO_DB_PATH ?? "data/halo-canvas.sqlite";

if (dbPath !== ":memory:") {
  mkdirSync(dirname(dbPath), { recursive: true });
}

export default defineConfig({
  dialect: "sqlite",
  schema: "./src/server/db/schema.ts",
  tablesFilter: [
    "spans",
    "trace_summaries",
    "ingest_batches",
    "live_events",
    "langfuse_connections",
    "langfuse_import_jobs",
    "halo_engine_settings",
    "halo_model_providers",
    "halo_runs",
    "halo_run_events",
    "halo_run_artifacts",
  ],
  dbCredentials: {
    url: dbPath,
  },
  strict: false,
  verbose: true,
});
