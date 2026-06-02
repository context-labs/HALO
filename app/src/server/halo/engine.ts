import { existsSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import type { DatabaseHandle } from "../db/client";
import {
  defaultHaloInstallPath,
  getHaloEngineSettings,
  saveHaloEngineSettings,
  updateHaloProviderTestStatus,
} from "./storage";
import { HALO_REPO_URL, type HaloEngineStatus, type StoredHaloModelProvider } from "./types";

const COMMAND_TIMEOUT_MS = 120_000;

export async function getHaloEngineStatus(
  database: DatabaseHandle,
): Promise<HaloEngineStatus> {
  const settings = getHaloEngineSettings(database.sqlite, database.path);
  const [git, uv, python, commit, importable] = await Promise.all([
    commandVersion(["git", "--version"]),
    commandVersion(["uv", "--version"]),
    commandVersion(["python3.12", "--version"]).then(async (value) =>
      value ?? (await commandVersion(["python3", "--version"])),
    ),
    gitCommit(settings.installPath),
    checkImportable(settings.installPath),
  ]);
  return {
    ...settings,
    checks: {
      git,
      importable,
      python,
      uv,
    },
    commitSha: commit ?? settings.commitSha,
    status:
      settings.status === "installing"
        ? "installing"
        : importable
          ? "installed"
          : settings.status === "error"
            ? "error"
            : "not_installed",
  };
}

export async function installOrUpdateHaloEngine(database: DatabaseHandle) {
  const current = getHaloEngineSettings(database.sqlite, database.path);
  const installPath = current.installPath || defaultHaloInstallPath(database.path);
  saveHaloEngineSettings(database.sqlite, {
    dbPath: database.path,
    installPath,
    repoUrl: current.repoUrl || HALO_REPO_URL,
    status: "installing",
  });

  try {
    mkdirSync(dirname(installPath), { recursive: true });
    if (existsSync(`${installPath}/.git`)) {
      await runCommand(["git", "-C", installPath, "pull", "--ff-only"]);
    } else {
      await runCommand(["git", "clone", current.repoUrl || HALO_REPO_URL, installPath]);
    }
    await runCommand(["uv", "sync"], { cwd: installPath, timeoutMs: 240_000 });
    await runCommand(
      [
        "uv",
        "run",
        "python",
        "-c",
        "from engine.main import stream_engine_async; print('halo import ok')",
      ],
      { cwd: installPath },
    );
    const commitSha = await gitCommit(installPath);
    saveHaloEngineSettings(database.sqlite, {
      commitSha,
      dbPath: database.path,
      error: null,
      installPath,
      repoUrl: current.repoUrl || HALO_REPO_URL,
      status: "installed",
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Could not install HALO.";
    saveHaloEngineSettings(database.sqlite, {
      dbPath: database.path,
      error: message,
      installPath,
      repoUrl: current.repoUrl || HALO_REPO_URL,
      status: "error",
    });
    throw error;
  }

  return getHaloEngineStatus(database);
}

export async function testHaloProvider(
  database: DatabaseHandle,
  provider: StoredHaloModelProvider,
) {
  try {
    const url = `${provider.baseUrl.replace(/\/+$/, "")}/models`;
    const response = await fetch(url, {
      headers: {
        ...provider.headers,
        Authorization: `Bearer ${provider.apiKey}`,
      },
    });
    if (!response.ok) {
      const body = await response.text().catch(() => "");
      throw new Error(
        `Provider returned ${response.status}${body ? `: ${body.slice(0, 300)}` : ""}`,
      );
    }
    return updateHaloProviderTestStatus(database.sqlite, provider.id, {
      status: "connected",
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Could not connect to provider.";
    const updated = updateHaloProviderTestStatus(database.sqlite, provider.id, {
      error: message,
      status: "error",
    });
    throw new Error(message, { cause: updated });
  }
}

export async function runCommand(
  command: string[],
  options: { cwd?: string; timeoutMs?: number } = {},
) {
  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(),
    options.timeoutMs ?? COMMAND_TIMEOUT_MS,
  );
  try {
    const proc = Bun.spawn(command, {
      cwd: options.cwd,
      stderr: "pipe",
      stdout: "pipe",
      signal: controller.signal,
    });
    const [exitCode, stdout, stderr] = await Promise.all([
      proc.exited,
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
    ]);
    if (exitCode !== 0) {
      throw new Error(
        `${command.join(" ")} failed with ${exitCode}: ${(stderr || stdout).trim()}`,
      );
    }
    return stdout.trim();
  } finally {
    clearTimeout(timer);
  }
}

async function commandVersion(command: string[]) {
  try {
    return await runCommand(command, { timeoutMs: 10_000 });
  } catch {
    return null;
  }
}

async function gitCommit(installPath: string) {
  if (!existsSync(`${installPath}/.git`)) return null;
  try {
    return await runCommand(["git", "-C", installPath, "rev-parse", "--short", "HEAD"]);
  } catch {
    return null;
  }
}

async function checkImportable(installPath: string) {
  if (!existsSync(`${installPath}/pyproject.toml`)) return false;
  try {
    await runCommand(
      [
        "uv",
        "run",
        "python",
        "-c",
        "from engine.main import stream_engine_async; print('ok')",
      ],
      { cwd: installPath, timeoutMs: 20_000 },
    );
    return true;
  } catch {
    return false;
  }
}
