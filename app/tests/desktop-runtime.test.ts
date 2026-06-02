import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Database } from "bun:sqlite";
import { describe, expect, test } from "bun:test";
import {
  configureDesktopRuntimeEnv,
  defaultAppDataDir,
} from "../src/bun/desktopRuntime";

describe("desktop runtime paths", () => {
  test("uses macOS Application Support for production app data", () => {
    expect(defaultAppDataDir("darwin", "/Users/alice", {} as NodeJS.ProcessEnv)).toBe(
      "/Users/alice/Library/Application Support/net.inference.halo",
    );
  });

  test("uses XDG data home on Linux when available", () => {
    expect(
      defaultAppDataDir("linux", "/home/alice", {
        XDG_DATA_HOME: "/home/alice/.data",
      } as NodeJS.ProcessEnv),
    ).toBe("/home/alice/.data/net.inference.halo");
  });

  test("respects explicit app data and database paths", () => {
    const dir = mkdtempSync(join(tmpdir(), "halo-runtime-"));
    const appDataDir = join(dir, "support");
    const dbPath = join(dir, "custom.sqlite");
    const env = {
      HALO_APP_DATA_DIR: appDataDir,
      HALO_DB_PATH: dbPath,
    } as NodeJS.ProcessEnv;

    try {
      const paths = configureDesktopRuntimeEnv(env);

      expect(paths.appDataDir).toBe(appDataDir);
      expect(paths.dbPath).toBe(dbPath);
      expect(paths.migratedLegacyFiles).toEqual([]);
      expect(paths.legacyDataDirs.length).toBeGreaterThan(0);
      expect(existsSync(appDataDir)).toBe(true);
    } finally {
      rmSync(dir, { force: true, recursive: true });
    }
  });

  test("migrates legacy bundle data into app support", () => {
    const dir = mkdtempSync(join(tmpdir(), "halo-runtime-"));
    const appDataDir = join(dir, "support");
    const legacyDataDir = join(dir, "HALO.app", "Contents", "MacOS", "data");
    const dbPath = join(appDataDir, "halo-canvas.sqlite");
    const legacyDbPath = join(legacyDataDir, "halo-canvas.sqlite");

    mkdirSync(join(legacyDataDir, "halo-engine"), { recursive: true });
    mkdirSync(join(legacyDataDir, "halo-runs", "run-1"), { recursive: true });
    writeFileSync(join(legacyDataDir, "halo-engine", "README.md"), "engine");
    writeFileSync(join(legacyDataDir, "halo-runs", "run-1", "result.json"), "{}");

    const sqlite = new Database(legacyDbPath, { create: true, strict: true });
    sqlite.run(`
      CREATE TABLE halo_engine_settings (
        id TEXT PRIMARY KEY,
        install_path TEXT NOT NULL,
        updated_at INTEGER NOT NULL
      )
    `);
    sqlite.run(`
      CREATE TABLE halo_runs (
        id TEXT PRIMARY KEY,
        export_path TEXT,
        result_path TEXT
      )
    `);
    sqlite.run(`
      CREATE TABLE halo_run_artifacts (
        id TEXT PRIMARY KEY,
        path TEXT NOT NULL
      )
    `);
    sqlite
      .query("INSERT INTO halo_engine_settings VALUES ('default', ?, 1)")
      .run(join(legacyDataDir, "halo-engine"));
    sqlite
      .query("INSERT INTO halo_runs VALUES ('run-1', ?, ?)")
      .run(
        join(legacyDataDir, "halo-runs", "run-1", "traces.jsonl"),
        join(legacyDataDir, "halo-runs", "run-1", "result.json"),
      );
    sqlite
      .query("INSERT INTO halo_run_artifacts VALUES ('artifact-1', ?)")
      .run(join(legacyDataDir, "halo-runs", "run-1", "result.json"));
    sqlite.close(false);

    try {
      const paths = configureDesktopRuntimeEnv({
        HALO_APP_DATA_DIR: appDataDir,
        HALO_LEGACY_DATA_DIR: legacyDataDir,
      } as NodeJS.ProcessEnv);

      expect(paths.dbPath).toBe(dbPath);
      expect(existsSync(dbPath)).toBe(true);
      expect(
        readFileSync(join(appDataDir, "halo-engine", "README.md"), "utf8"),
      ).toBe("engine");
      expect(existsSync(join(appDataDir, "halo-runs", "run-1", "result.json"))).toBe(
        true,
      );

      const migrated = new Database(dbPath, { create: false, strict: true });
      expect(
        migrated
          .query<{ install_path: string }, []>(
            "SELECT install_path FROM halo_engine_settings WHERE id = 'default'",
          )
          .get()?.install_path,
      ).toBe(join(appDataDir, "halo-engine"));
      expect(
        migrated
          .query<{ result_path: string }, []>(
            "SELECT result_path FROM halo_runs WHERE id = 'run-1'",
          )
          .get()?.result_path,
      ).toBe(join(appDataDir, "halo-runs", "run-1", "result.json"));
      expect(
        migrated
          .query<{ path: string }, []>(
            "SELECT path FROM halo_run_artifacts WHERE id = 'artifact-1'",
          )
          .get()?.path,
      ).toBe(join(appDataDir, "halo-runs", "run-1", "result.json"));
      migrated.close(false);
    } finally {
      rmSync(dir, { force: true, recursive: true });
    }
  });
});
