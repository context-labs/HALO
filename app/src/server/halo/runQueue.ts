import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Bunqueue, type Job } from "bunqueue/client";
import type { DatabaseHandle } from "../db/client";
import type { LiveEventStore } from "../live/events";
import { exportHaloTraceJsonl, previewHaloRunExport } from "./exporter";
import { getHaloEngineStatus, installOrUpdateHaloEngine } from "./engine";
import {
  addHaloRunEvent,
  createHaloRun,
  getHaloProvider,
  getHaloRun,
  isHaloRunCancelled,
  listHaloRuns,
  markInterruptedHaloRuns,
  publishHaloRun,
  publishHaloRunEvent,
  updateHaloRun,
} from "./storage";
import type { HaloRun, StartHaloRunInput } from "./types";

type HaloJobData = {
  runId: string;
};

type HaloJobResult = {
  runId: string;
  cancelled?: boolean;
};

export type HaloRunService = ReturnType<typeof createHaloRunService>;

const HALO_QUEUE_NAME = "halo-runs";
const HALO_ROUTE = "halo.run";

export function createHaloRunService(options: {
  database: DatabaseHandle;
  live: LiveEventStore;
}) {
  const { database, live } = options;
  markInterruptedHaloRuns(database.sqlite);

  let queue: Bunqueue<HaloJobData, HaloJobResult>;
  queue = new Bunqueue<HaloJobData, HaloJobResult>(HALO_QUEUE_NAME, {
    concurrency: 1,
    dataPath: queueDataPath(database.path),
    defaultJobOptions: {
      durable: true,
      removeOnComplete: { count: 200 },
      removeOnFail: { count: 200 },
    },
    dlq: {
      autoRetry: false,
      maxEntries: 500,
    },
    embedded: true,
    retry: {
      delay: 1_000,
      maxAttempts: 1,
      strategy: "jitter",
    },
    routes: {
      [HALO_ROUTE]: async (job) =>
        processHaloRun({
          database,
          job,
          live,
          queue,
        }),
    },
  });

  queue.on("failed", (job, error) => {
    const run = getHaloRun(database.sqlite, job.data.runId);
    if (!run || run.status === "cancelled") return;
    const failed = updateHaloRun(database.sqlite, run.id, {
      errorMessage: error.message,
      finishedAt: Date.now(),
      progress: Math.max(run.progress, 95),
      status: "failed",
    });
    publishHaloRun(live, failed);
  });

  return {
    async cancel(runId: string) {
      const run = getHaloRun(database.sqlite, runId);
      if (!run) return null;
      const updated = updateHaloRun(database.sqlite, runId, {
        errorMessage: "HALO run cancelled by user.",
        finishedAt: Date.now(),
        status: "cancelled",
      });
      addAndPublishEvent(database, live, updated, "cancelled", {
        error: "HALO run cancelled by user.",
      });
      publishHaloRun(live, updated);
      if (run.bunqueueJobId) queue.cancel(run.bunqueueJobId);
      return updated;
    },

    close(force?: boolean) {
      return queue.close(force);
    },

    get(runId: string) {
      return getHaloRun(database.sqlite, runId);
    },

    list(limit?: number) {
      return listHaloRuns(database.sqlite, limit);
    },

    preview(input: Pick<StartHaloRunInput, "filters" | "targetType">) {
      return previewHaloRunExport(database.sqlite, input);
    },

    async retry(runId: string) {
      const run = getHaloRun(database.sqlite, runId);
      if (!run) return null;
      const queued = await queue.add(
        HALO_ROUTE,
        { runId },
        {
          durable: true,
          jobId: `${runId}:${Date.now()}`,
          priority: 5,
        },
      );
      const updated = updateHaloRun(database.sqlite, runId, {
        bunqueueJobId: queued.id,
        errorMessage: null,
        finishedAt: null,
        progress: 0,
        startedAt: null,
        status: "queued",
      });
      publishHaloRun(live, updated);
      return updated;
    },

    async start(input: StartHaloRunInput): Promise<HaloRun> {
      const provider = getHaloProvider(database.sqlite, input.providerId);
      if (!provider) throw new Error("HALO model provider not found.");
      const run = createHaloRun(database.sqlite, {
        ...input,
        model: input.model?.trim() || provider.model,
        providerName: provider.name,
        title:
          input.title?.trim() ||
          `${input.targetType === "session_group" ? "Session" : "Trace"} analysis`,
      });
      const queued = await queue.add(
        HALO_ROUTE,
        { runId: run.id },
        {
          durable: true,
          jobId: run.id,
          priority: 5,
        },
      );
      const updated = updateHaloRun(database.sqlite, run.id, {
        bunqueueJobId: queued.id,
        status: "queued",
      });
      addAndPublishEvent(database, live, updated, "queued", {
        targetType: updated.targetType,
      });
      publishHaloRun(live, updated);
      return updated;
    },
  };
}

async function processHaloRun(input: {
  database: DatabaseHandle;
  job: Job<HaloJobData>;
  live: LiveEventStore;
  queue: Bunqueue<HaloJobData, HaloJobResult>;
}): Promise<HaloJobResult> {
  const { database, job, live, queue } = input;
  const runId = job.data.runId;
  let run = getHaloRun(database.sqlite, runId);
  if (!run || !["queued", "running", "exporting"].includes(run.status)) {
    return { cancelled: true, runId };
  }
  const provider = run.providerId
    ? getHaloProvider(database.sqlite, run.providerId)
    : null;
  if (!provider) throw new Error("HALO model provider not found.");

  let engine = await getHaloEngineStatus(database);
  if (engine.status !== "installed" || !engine.checks.importable) {
    run = updateHaloRun(database.sqlite, runId, {
      progress: 10,
      status: "running",
    });
    publishHaloRun(live, run);
    addAndPublishEvent(database, live, run, "installing_engine", {
      installPath: engine.defaultInstallPath,
      previousStatus: engine.status,
    });

    try {
      engine = await installOrUpdateHaloEngine(database);
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Could not install the HALO engine automatically.";
      throw new Error(
        `HALO engine is not ready and automatic install failed: ${message}`,
      );
    }

    if (engine.status !== "installed" || !engine.checks.importable) {
      throw new Error(
        "HALO engine is not ready after automatic install. Check Settings for dependency status.",
      );
    }
  }

  const signal = queue.getSignal(job.id) ?? undefined;
  const outputDir = outputDirForRun(database.path, run.id);
  mkdirSync(outputDir, { recursive: true });

  run = updateHaloRun(database.sqlite, runId, {
    progress: 5,
    status: "exporting",
  });
  publishHaloRun(live, run);
  addAndPublishEvent(database, live, run, "exporting", {
    targetType: run.targetType,
  });

  if (isCancelled(database, runId, signal)) {
    await markCancelled(database, live, runId);
    return { cancelled: true, runId };
  }

  const exported = exportHaloTraceJsonl(database.sqlite, {
    filters: run.filters,
    outputDir,
    runId,
    targetType: run.targetType,
  });
  run = updateHaloRun(database.sqlite, runId, {
    exportPath: exported.path,
    progress: 18,
    sessionCount: exported.sessionCount,
    spanCount: exported.spanCount,
    traceCount: exported.traceCount,
  });
  publishHaloRun(live, run);
  addAndPublishEvent(database, live, run, "exported", {
    path: exported.path,
    sessionCount: exported.sessionCount,
    spanCount: exported.spanCount,
    traceCount: exported.traceCount,
    warnings: exported.warnings,
  });

  if (exported.spanCount === 0 || exported.traceCount === 0) {
    run = updateHaloRun(database.sqlite, runId, {
      errorMessage: "No traces matched the selected HALO filters.",
      finishedAt: Date.now(),
      progress: 100,
      status: "failed",
    });
    publishHaloRun(live, run);
    addAndPublishEvent(database, live, run, "failed", {
      error: run.errorMessage,
    });
    return { runId };
  }

  const configPath = join(outputDir, "runner-config.json");
  const resultPath = join(outputDir, "result.json");
  writeFileSync(
    configPath,
    JSON.stringify(
      {
        haloPath: engine.installPath,
        maxDepth: run.maxDepth,
        maxParallel: run.maxParallel,
        maxTurns: run.maxTurns,
        model: run.model || provider.model,
        prompt: run.prompt,
        provider: {
          apiKey: provider.apiKey,
          baseUrl: provider.baseUrl,
          headers: provider.headers,
        },
        runId,
        tracePath: exported.path,
      },
      null,
      2,
    ),
    "utf8",
  );

  run = updateHaloRun(database.sqlite, runId, {
    progress: 25,
    resultPath,
    startedAt: Date.now(),
    status: "running",
  });
  publishHaloRun(live, run);

  const terminal = await runPythonBridge({
    configPath,
    database,
    enginePath: engine.installPath,
    live,
    resultPath,
    run,
    signal,
  });
  if (terminal.cancelled) {
    await markCancelled(database, live, runId);
    return { cancelled: true, runId };
  }
  return { runId };
}

async function runPythonBridge(input: {
  configPath: string;
  database: DatabaseHandle;
  enginePath: string;
  live: LiveEventStore;
  resultPath: string;
  run: HaloRun;
  signal: AbortSignal | undefined;
}) {
  const runnerPath = resolveHaloRunnerPath();
  const proc = Bun.spawn(["uv", "run", "python", runnerPath, input.configPath], {
    cwd: input.enginePath,
    stderr: "pipe",
    stdout: "pipe",
  });
  const abort = () => proc.kill();
  input.signal?.addEventListener("abort", abort, { once: true });

  let terminalSeen = false;
  let currentRun = input.run;
  let stdoutError: Error | null = null;
  const stdoutPromise = readJsonLines(proc.stdout, async (event) => {
    if (isCancelled(input.database, input.run.id, input.signal)) {
      proc.kill();
      return;
    }
    const eventType = String(event.type ?? "log");
    currentRun = getHaloRun(input.database.sqlite, input.run.id) ?? currentRun;
    addAndPublishEvent(input.database, input.live, currentRun, eventType, event);

    if (eventType === "delta" || eventType === "agent_step") {
      const progress = Math.min(92, Math.max(currentRun.progress, eventType === "delta" ? 45 : 60));
      currentRun = updateHaloRun(input.database.sqlite, input.run.id, {
        progress,
      });
      publishHaloRun(input.live, currentRun);
      return;
    }

    if (eventType === "completed" || eventType === "incomplete") {
      terminalSeen = true;
      const finalAnswer =
        typeof event.finalAnswer === "string" ? event.finalAnswer : "";
      const finalAnswerSource =
        typeof event.finalAnswerSource === "string"
          ? event.finalAnswerSource
          : eventType;
      writeFileSync(
        input.resultPath,
        JSON.stringify({ event, finalAnswer, runId: input.run.id }, null, 2),
        "utf8",
      );
      currentRun = updateHaloRun(input.database.sqlite, input.run.id, {
        finalAnswer,
        finalAnswerSource,
        finishedAt: Date.now(),
        progress: 100,
        status: eventType === "completed" ? "completed" : "incomplete",
      });
      publishHaloRun(input.live, currentRun);
      return;
    }

    if (eventType === "failed") {
      terminalSeen = true;
      currentRun = updateHaloRun(input.database.sqlite, input.run.id, {
        errorMessage: typeof event.error === "string" ? event.error : "HALO run failed.",
        finishedAt: Date.now(),
        progress: 100,
        status: "failed",
      });
      publishHaloRun(input.live, currentRun);
    }
  }).catch((error) => {
    stdoutError = error instanceof Error ? error : new Error(String(error));
  });

  const stderrPromise = new Response(proc.stderr).text();
  const exitCode = await proc.exited;
  await stdoutPromise;
  input.signal?.removeEventListener("abort", abort);
  const stderr = await stderrPromise;

  if (isCancelled(input.database, input.run.id, input.signal)) {
    return { cancelled: true };
  }
  if (stdoutError) throw stdoutError;
  if (exitCode !== 0 && !terminalSeen) {
    const message = stderr.trim() || `HALO runner exited with ${exitCode}`;
    const failed = updateHaloRun(input.database.sqlite, input.run.id, {
      errorMessage: message,
      finishedAt: Date.now(),
      progress: 100,
      status: "failed",
    });
    publishHaloRun(input.live, failed);
    addAndPublishEvent(input.database, input.live, failed, "failed", {
      error: message,
    });
  }
  return { cancelled: false };
}

function resolveHaloRunnerPath() {
  const candidates = haloRunnerPathCandidates();
  const runnerPath = candidates.find((candidate) => existsSync(candidate));
  if (!runnerPath) {
    throw new Error(
      `HALO local runner script was not found. Checked: ${candidates.join(", ")}`,
    );
  }
  return runnerPath;
}

export function haloRunnerPathCandidates(input: {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  importMetaUrl?: string;
} = {}) {
  const env = input.env ?? process.env;
  const cwd = input.cwd ?? process.cwd();
  const importMetaUrl = input.importMetaUrl ?? import.meta.url;

  return [
    env.HALO_RUNNER_PATH,
    env.HALO_PROJECT_ROOT
      ? resolve(env.HALO_PROJECT_ROOT, "scripts/halo-local-runner.py")
      : undefined,
    fileURLToPath(new URL("../scripts/halo-local-runner.py", importMetaUrl)),
    fileURLToPath(new URL("./app/scripts/halo-local-runner.py", importMetaUrl)),
    fileURLToPath(new URL("./scripts/halo-local-runner.py", importMetaUrl)),
    fileURLToPath(new URL("../../../scripts/halo-local-runner.py", importMetaUrl)),
    resolve(cwd, "scripts/halo-local-runner.py"),
  ].filter(Boolean) as string[];
}

async function readJsonLines(
  stream: ReadableStream<Uint8Array> | null,
  onLine: (event: Record<string, unknown>) => Promise<void>,
) {
  if (!stream) return;
  const decoder = new TextDecoder();
  const reader = stream.getReader();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline = buffer.indexOf("\n");
    while (newline !== -1) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) await onLine(JSON.parse(line) as Record<string, unknown>);
      newline = buffer.indexOf("\n");
    }
  }
  const trailing = buffer.trim();
  if (trailing) await onLine(JSON.parse(trailing) as Record<string, unknown>);
}

function addAndPublishEvent(
  database: DatabaseHandle,
  live: LiveEventStore,
  run: HaloRun,
  eventType: string,
  payload: Record<string, unknown>,
) {
  const event = addHaloRunEvent(database.sqlite, {
    eventType,
    payload,
    runId: run.id,
  });
  publishHaloRunEvent(live, run, event);
  return event;
}

async function markCancelled(
  database: DatabaseHandle,
  live: LiveEventStore,
  runId: string,
) {
  const cancelled = updateHaloRun(database.sqlite, runId, {
    errorMessage: "HALO run cancelled by user.",
    finishedAt: Date.now(),
    status: "cancelled",
  });
  publishHaloRun(live, cancelled);
  addAndPublishEvent(database, live, cancelled, "cancelled", {
    error: "HALO run cancelled by user.",
  });
}

function isCancelled(
  database: DatabaseHandle,
  runId: string,
  signal: AbortSignal | undefined,
) {
  return signal?.aborted || isHaloRunCancelled(database.sqlite, runId);
}

function outputDirForRun(databasePath: string, runId: string) {
  if (databasePath === ":memory:") return resolve("data/halo-runs", runId);
  return resolve(dirname(databasePath), "halo-runs", runId);
}

function queueDataPath(databasePath: string) {
  return databasePath === ":memory:" ? ":memory:" : `${databasePath}.halo.bunqueue.sqlite`;
}
