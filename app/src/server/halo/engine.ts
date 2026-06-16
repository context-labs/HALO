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
  // While an install runs, status polls would spawn `uv run` checks into the
  // same directory `uv sync` is writing — skip the expensive probes and
  // report the stored phase instead.
  if (settings.status === "installing") {
    return { ...settings, status: "installing" };
  }
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
    status: importable
      ? "installed"
      : settings.status === "error"
        ? "error"
        : "not_installed",
  };
}

// Onboarding auto-starts the install while the Settings button stays
// clickable; share one in-flight install per database instead of racing two
// clones into the same directory.
const installsInFlight = new Map<string, Promise<HaloEngineStatus>>();

export function installOrUpdateHaloEngine(database: DatabaseHandle) {
  const inFlight = installsInFlight.get(database.path);
  if (inFlight) return inFlight;
  const install = runHaloEngineInstall(database).finally(() => {
    installsInFlight.delete(database.path);
  });
  installsInFlight.set(database.path, install);
  return install;
}

async function runHaloEngineInstall(database: DatabaseHandle) {
  const current = getHaloEngineSettings(database.sqlite, database.path);
  const installPath = current.installPath || defaultHaloInstallPath(database.path);
  const repoUrl = current.repoUrl || HALO_REPO_URL;
  const setPhase = (statusDetail: string) =>
    saveHaloEngineSettings(database.sqlite, {
      dbPath: database.path,
      installPath,
      repoUrl,
      status: "installing",
      statusDetail,
    });

  setPhase("Preparing…");

  try {
    mkdirSync(dirname(installPath), { recursive: true });
    if (existsSync(`${installPath}/.git`)) {
      setPhase("Updating the HALO repository…");
      await runCommand(["git", "-C", installPath, "pull", "--ff-only"]);
    } else {
      setPhase("Downloading the HALO repository…");
      await runCommand(["git", "clone", repoUrl, installPath]);
    }
    setPhase("Installing Python dependencies…");
    await runCommand(["uv", "sync"], { cwd: installPath, timeoutMs: 240_000 });
    setPhase("Verifying the engine…");
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
      repoUrl,
      status: "installed",
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Could not install HALO.";
    saveHaloEngineSettings(database.sqlite, {
      dbPath: database.path,
      error: message,
      installPath,
      repoUrl,
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
  const commandName = command[0];
  if (!commandName) {
    throw new Error("Cannot run an empty command.");
  }
  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(),
    options.timeoutMs ?? COMMAND_TIMEOUT_MS,
  );
  try {
    let proc: ReturnType<typeof Bun.spawn>;
    try {
      proc = Bun.spawn(command, {
        cwd: options.cwd,
        stderr: "pipe",
        stdout: "pipe",
        signal: controller.signal,
      });
    } catch (error) {
      if (isCommandNotFoundError(error)) {
        throw new Error(commandNotFoundMessage(commandName));
      }
      throw error;
    }
    const [exitCode, stdout, stderr] = await Promise.all([
      proc.exited,
      streamText(proc.stdout),
      streamText(proc.stderr),
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

async function streamText(
  stream: ReadableStream<Uint8Array<ArrayBuffer>> | number | undefined,
) {
  return stream instanceof ReadableStream ? new Response(stream).text() : "";
}

function isCommandNotFoundError(error: unknown) {
  const code =
    typeof error === "object" && error && "code" in error
      ? (error as { code?: unknown }).code
      : null;
  if (code === "ENOENT") return true;
  return error instanceof Error && /ENOENT|not found/i.test(error.message);
}

function commandNotFoundMessage(commandName: string) {
  if (commandName === "uv") {
    return "uv was not found. Install uv from https://docs.astral.sh/uv/getting-started/installation/ or make sure it is on PATH.";
  }
  if (commandName === "git") {
    return "git was not found. Install git from https://git-scm.com or make sure it is on PATH.";
  }
  if (commandName === "python3.12" || commandName === "python3" || commandName === "python") {
    return "Python 3.12 was not found. Install Python 3.12 or make sure it is on PATH.";
  }
  return `${commandName} was not found. Install it or make sure it is on PATH.`;
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
