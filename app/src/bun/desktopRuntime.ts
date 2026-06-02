import {
  copyFileSync,
  cpSync,
  existsSync,
  chmodSync,
  mkdirSync,
  readdirSync,
  statSync,
} from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { Database } from "bun:sqlite";

const APP_IDENTIFIER = "net.inference.halo";
const DB_FILE_NAME = "halo-canvas.sqlite";

export type DesktopRuntimePaths = {
  appDataDir: string;
  dbPath: string;
  legacyDataDir: string;
  legacyDataDirs: string[];
  migratedLegacyFiles: string[];
};

export function configureDesktopRuntimeEnv(
  env: NodeJS.ProcessEnv = process.env,
): DesktopRuntimePaths {
  const appDataDir = resolve(
    env.HALO_APP_DATA_DIR?.trim() || defaultAppDataDir(),
  );
  ensurePrivateDirectory(appDataDir);

  const dbPath = resolve(env.HALO_DB_PATH?.trim() || join(appDataDir, DB_FILE_NAME));
  mkdirSync(dirname(dbPath), { recursive: true });

  const legacyDataDirs = legacyDataDirCandidates(appDataDir, env);
  const legacyDataDir = legacyDataDirs[0] ?? resolve(process.cwd(), "data");
  const migratedLegacyFiles = env.HALO_DB_PATH
    ? []
    : migrateLegacyBundleData({
        appDataDir,
        destinationDbPath: dbPath,
        legacyDataDirs,
      });

  env.HALO_APP_DATA_DIR = appDataDir;
  env.HALO_DB_PATH = dbPath;
  env.HALO_LEGACY_DATA_DIR = legacyDataDir;

  return {
    appDataDir,
    dbPath,
    legacyDataDir,
    legacyDataDirs,
    migratedLegacyFiles,
  };
}

export function defaultAppDataDir(
  platform = process.platform,
  homeDir = homedir(),
  env: NodeJS.ProcessEnv = process.env,
) {
  if (platform === "darwin") {
    return join(homeDir, "Library", "Application Support", APP_IDENTIFIER);
  }

  if (platform === "win32") {
    return join(env.LOCALAPPDATA || join(homeDir, "AppData", "Local"), APP_IDENTIFIER);
  }

  return join(env.XDG_DATA_HOME || join(homeDir, ".local", "share"), APP_IDENTIFIER);
}

function ensurePrivateDirectory(path: string) {
  mkdirSync(path, { mode: 0o700, recursive: true });
  try {
    chmodSync(path, 0o700);
  } catch {
    // Best effort only: custom volumes can reject chmod while still being writable.
  }
}

function migrateLegacyBundleData(input: {
  appDataDir: string;
  destinationDbPath: string;
  legacyDataDirs: string[];
}) {
  const destinationDbDir = dirname(input.destinationDbPath);
  const destinationDbName = basename(input.destinationDbPath);
  const legacyDataDir = input.legacyDataDirs.find((candidate) =>
    existsSync(join(candidate, DB_FILE_NAME)),
  );

  mkdirSync(destinationDbDir, { recursive: true });
  if (!legacyDataDir) {
    return [];
  }

  const migrated: string[] = [];
  if (!destinationHasDatabase(input.destinationDbPath)) {
    for (const fileName of readdirSync(legacyDataDir)) {
      if (!fileName.startsWith(DB_FILE_NAME)) continue;

      const source = join(legacyDataDir, fileName);
      if (!statSync(source).isFile()) continue;

      const suffix = fileName.slice(DB_FILE_NAME.length);
      const destination = join(destinationDbDir, `${destinationDbName}${suffix}`);
      copyFileSync(source, destination);
      migrated.push(destination);
    }
  }

  for (const directoryName of ["halo-engine", "halo-runs"]) {
    const source = join(legacyDataDir, directoryName);
    const destination = join(input.appDataDir, directoryName);
    if (!existsSync(source) || existsSync(destination)) continue;
    if (!statSync(source).isDirectory()) continue;

    cpSync(source, destination, { recursive: true });
    migrated.push(destination);
  }

  rewriteLegacyPathsInDatabase({
    appDataDir: input.appDataDir,
    dbPath: input.destinationDbPath,
    legacyDataDir,
  });

  return migrated;
}

function destinationHasDatabase(dbPath: string) {
  if (!existsSync(dbPath)) return false;

  try {
    return statSync(dbPath).size > 0;
  } catch {
    return true;
  }
}

function legacyDataDirCandidates(appDataDir: string, env: NodeJS.ProcessEnv) {
  const home = homedir();
  return uniqueResolvedPaths([
    env.HALO_LEGACY_DATA_DIR,
    join(process.cwd(), "data"),
    join(dirname(process.argv0), "data"),
    process.platform === "darwin"
      ? "/Applications/HALO.app/Contents/MacOS/data"
      : undefined,
    process.platform === "darwin"
      ? join(home, "Applications", "HALO.app", "Contents", "MacOS", "data")
      : undefined,
  ]).filter((path) => path !== appDataDir);
}

function uniqueResolvedPaths(paths: Array<string | undefined>) {
  const seen = new Set<string>();
  const result: string[] = [];

  for (const path of paths) {
    if (!path?.trim()) continue;
    const resolved = resolve(path);
    if (seen.has(resolved)) continue;
    seen.add(resolved);
    result.push(resolved);
  }

  return result;
}

function rewriteLegacyPathsInDatabase(input: {
  appDataDir: string;
  dbPath: string;
  legacyDataDir: string;
}) {
  if (!existsSync(input.dbPath)) return;

  const sqlite = new Database(input.dbPath, { create: false, strict: true });
  try {
    const legacyPrefix = resolve(input.legacyDataDir);
    const appPrefix = resolve(input.appDataDir);
    const now = Date.now();

    sqlite
      .query(
        `UPDATE halo_engine_settings
         SET install_path = REPLACE(install_path, ?, ?),
             updated_at = ?
         WHERE install_path LIKE ?`,
      )
      .run(legacyPrefix, appPrefix, now, `${legacyPrefix}%`);
    sqlite
      .query(
        `UPDATE halo_runs
         SET export_path = CASE
               WHEN export_path IS NULL THEN NULL
               ELSE REPLACE(export_path, ?, ?)
             END,
             result_path = CASE
               WHEN result_path IS NULL THEN NULL
               ELSE REPLACE(result_path, ?, ?)
             END
         WHERE export_path LIKE ? OR result_path LIKE ?`,
      )
      .run(
        legacyPrefix,
        appPrefix,
        legacyPrefix,
        appPrefix,
        `${legacyPrefix}%`,
        `${legacyPrefix}%`,
      );
    sqlite
      .query(
        `UPDATE halo_run_artifacts
         SET path = REPLACE(path, ?, ?)
         WHERE path LIKE ?`,
      )
      .run(legacyPrefix, appPrefix, `${legacyPrefix}%`);
  } catch {
    // Older databases may not have HALO analysis tables yet.
  } finally {
    sqlite.close(false);
  }
}
