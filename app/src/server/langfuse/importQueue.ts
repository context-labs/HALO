import { Bunqueue, type Job } from "bunqueue/client";
import type { DatabaseHandle } from "../db/client";
import type { LiveEventStore } from "../live/events";
import { ingestTelemetry } from "../telemetry/storage";
import { LangfuseApiClient, LangfuseApiError } from "./client";
import { langfuseTraceToOtlp } from "./mapper";
import {
  createLangfuseImportJob,
  getLangfuseConnection,
  getLangfuseImportJob,
  isLangfuseImportCancelled,
  listLangfuseImportJobs,
  markInterruptedLangfuseImports,
  publishLangfuseImportJob,
  updateLangfuseImportJob,
} from "./storage";
import type { LangfuseImportJob, LangfuseTraceFilters } from "./types";

type ImportJobData = {
  appJobId: string;
};

type ImportJobResult = {
  appJobId: string;
  cancelled?: boolean;
  importedTraces?: number;
};

type LangfuseImportServiceOptions = {
  database: DatabaseHandle;
  live: LiveEventStore;
};

export type LangfuseImportService = ReturnType<typeof createLangfuseImportService>;

const IMPORT_QUEUE_NAME = "langfuse-imports";
const IMPORT_ROUTE = "langfuse.import";
const PAGE_LIMIT = 50;

export function createLangfuseImportService(options: LangfuseImportServiceOptions) {
  const { database, live } = options;
  markInterruptedLangfuseImports(database.sqlite);

  let queue: Bunqueue<ImportJobData, ImportJobResult>;
  queue = new Bunqueue<ImportJobData, ImportJobResult>(IMPORT_QUEUE_NAME, {
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
      delay: 750,
      maxAttempts: 3,
      retryIf: (error) => isTransientImportError(error),
      strategy: "jitter",
    },
    routes: {
      [IMPORT_ROUTE]: async (job) =>
        processImportJob({
          database,
          job,
          live,
          queue,
        }),
    },
  });

  queue.on("failed", (job, error) => {
    const appJobId = job.data.appJobId;
    const current = getLangfuseImportJob(database.sqlite, appJobId);
    if (!current || current.status === "cancelled") return;
    const updated = updateLangfuseImportJob(database.sqlite, appJobId, {
      errorMessage: error.message,
      finishedAt: Date.now(),
      status: "failed",
    });
    publishLangfuseImportJob(live, updated);
  });

  return {
    async cancel(jobId: string) {
      const job = getLangfuseImportJob(database.sqlite, jobId);
      if (!job) return null;
      const updated = updateLangfuseImportJob(database.sqlite, jobId, {
        errorMessage: "Import cancelled by user.",
        finishedAt: Date.now(),
        status: "cancelled",
      });
      publishLangfuseImportJob(live, updated);
      if (job.bunqueueJobId) queue.cancel(job.bunqueueJobId);
      return updated;
    },

    close(force?: boolean) {
      return queue.close(force);
    },

    get(jobId: string) {
      return getLangfuseImportJob(database.sqlite, jobId);
    },

    list(limit?: number) {
      return listLangfuseImportJobs(database.sqlite, limit);
    },

    async start(input: {
      connectionId: string;
      filters: LangfuseTraceFilters;
    }): Promise<LangfuseImportJob> {
      const connection = getLangfuseConnection(database.sqlite, input.connectionId);
      if (!connection) throw new Error("Langfuse connection not found");

      const appJob = createLangfuseImportJob(database.sqlite, input);
      const queued = await queue.add(
        IMPORT_ROUTE,
        { appJobId: appJob.id },
        {
          durable: true,
          jobId: appJob.id,
          priority: 5,
        },
      );
      const updated = updateLangfuseImportJob(database.sqlite, appJob.id, {
        bunqueueJobId: queued.id,
        status: "queued",
      });
      publishLangfuseImportJob(live, updated);
      return updated;
    },
  };
}

async function processImportJob(input: {
  database: DatabaseHandle;
  job: Job<ImportJobData>;
  live: LiveEventStore;
  queue: Bunqueue<ImportJobData, ImportJobResult>;
}): Promise<ImportJobResult> {
  const { database, job, live, queue } = input;
  const appJobId = job.data.appJobId;
  const appJob = getLangfuseImportJob(database.sqlite, appJobId);
  if (!appJob || !["queued", "running"].includes(appJob.status)) {
    return { appJobId, cancelled: true };
  }

  const connection = getLangfuseConnection(database.sqlite, appJob.connectionId);
  if (!connection) {
    throw new Error("Langfuse connection not found");
  }

  const client = new LangfuseApiClient(connection);
  const signal = queue.getSignal(job.id) ?? undefined;
  let importedTraces = appJob.importedTraces;
  let importedObservations = appJob.importedObservations;
  let failedTraces = appJob.failedTraces;
  let processedTraces = importedTraces + failedTraces;
  let totalTraces = appJob.totalTraces;
  let totalObservations = appJob.totalObservations;

  await updateProgress({
    database,
    job,
    live,
    patch: {
      errorMessage: null,
      progress: 1,
      startedAt: Date.now(),
      status: "running",
    },
  });

  try {
    let page = 1;
    while (true) {
      if (isCancelled(database, appJobId, signal)) {
        await markCancelled({ database, job, live });
        return { appJobId, cancelled: true };
      }

      const list = await client.listTraces(
        {
          filters: appJob.filters,
          limit: PAGE_LIMIT,
          orderBy: "timestamp.asc",
          page,
        },
        signal,
      );
      const traces = list.data ?? [];
      totalTraces = list.meta?.totalItems ?? totalTraces;

      await updateProgress({
        database,
        job,
        live,
        patch: {
          progress: progressFor(processedTraces, totalTraces),
          totalTraces,
        },
      });

      if (traces.length === 0) break;

      for (const listedTrace of traces) {
        if (isCancelled(database, appJobId, signal)) {
          await markCancelled({ database, job, live });
          return { appJobId, cancelled: true };
        }

        await updateProgress({
          database,
          job,
          live,
          patch: {
            currentTraceId: listedTrace.id,
            currentTraceName: listedTrace.name ?? null,
            progress: progressFor(processedTraces, totalTraces),
          },
        });

        try {
          const trace = await client.getTrace(listedTrace.id, signal);
          const body = JSON.stringify(
            langfuseTraceToOtlp(trace, {
              baseUrl: connection.baseUrl,
              connectionId: connection.id,
              connectionName: connection.name,
              importedAt: Date.now(),
              importJobId: appJobId,
            }),
          );
          const result = ingestTelemetry(
            database.sqlite,
            {
              body,
              contentEncoding: "langfuse-import",
              sizeBytes: Buffer.byteLength(body),
            },
            live,
          );
          importedTraces += 1;
          importedObservations += result.acceptedSpanCount;
          totalObservations += Math.max(trace.observations.length, result.acceptedSpanCount);
        } catch (error) {
          if (isCancelled(database, appJobId, signal) || isAbortError(error)) {
            await markCancelled({ database, job, live });
            return { appJobId, cancelled: true };
          }
          failedTraces += 1;
        } finally {
          processedTraces += 1;
          await updateProgress({
            database,
            job,
            live,
            patch: {
              failedTraces,
              importedObservations,
              importedTraces,
              progress: progressFor(processedTraces, totalTraces),
              totalObservations,
              totalTraces,
            },
          });
        }
      }

      const totalPages = list.meta?.totalPages;
      if (totalPages != null && page >= totalPages) break;
      if (traces.length < PAGE_LIMIT) break;
      page += 1;
    }

    const complete = updateLangfuseImportJob(database.sqlite, appJobId, {
      currentTraceId: null,
      currentTraceName: null,
      finishedAt: Date.now(),
      progress: 100,
      status: "completed",
    });
    await job.updateProgress(100, "Import complete");
    publishLangfuseImportJob(live, complete);
    return { appJobId, importedTraces };
  } catch (error) {
    if (isCancelled(database, appJobId, signal) || isAbortError(error)) {
      await markCancelled({ database, job, live });
      return { appJobId, cancelled: true };
    }
    const message = error instanceof Error ? error.message : "Import failed";
    const failed = updateLangfuseImportJob(database.sqlite, appJobId, {
      errorMessage: message,
      finishedAt: Date.now(),
      status: "failed",
    });
    publishLangfuseImportJob(live, failed);
    throw error;
  }
}

async function updateProgress(input: {
  database: DatabaseHandle;
  job: Job<ImportJobData>;
  live: LiveEventStore;
  patch: Parameters<typeof updateLangfuseImportJob>[2];
}) {
  const updated = updateLangfuseImportJob(
    input.database.sqlite,
    input.job.data.appJobId,
    input.patch,
  );
  if (input.patch.progress != null) {
    await input.job.updateProgress(
      input.patch.progress,
      updated.currentTraceName ?? updated.status,
    );
  }
  publishLangfuseImportJob(input.live, updated);
}

async function markCancelled(input: {
  database: DatabaseHandle;
  job: Job<ImportJobData>;
  live: LiveEventStore;
}) {
  const updated = updateLangfuseImportJob(
    input.database.sqlite,
    input.job.data.appJobId,
    {
      errorMessage: "Import cancelled by user.",
      finishedAt: Date.now(),
      status: "cancelled",
    },
  );
  await input.job.updateProgress(updated.progress, "Import cancelled");
  publishLangfuseImportJob(input.live, updated);
}

function isCancelled(
  database: DatabaseHandle,
  appJobId: string,
  signal: AbortSignal | undefined,
) {
  return signal?.aborted || isLangfuseImportCancelled(database.sqlite, appJobId);
}

function progressFor(processedTraces: number, totalTraces: number) {
  if (totalTraces <= 0) return processedTraces > 0 ? 95 : 5;
  return Math.min(99, Math.max(5, Math.floor((processedTraces / totalTraces) * 100)));
}

function isTransientImportError(error: Error) {
  if (error instanceof LangfuseApiError) {
    return error.status === 429 || (error.status != null && error.status >= 500);
  }
  return true;
}

function isAbortError(error: unknown) {
  return error instanceof Error && error.name === "AbortError";
}

function queueDataPath(databasePath: string) {
  return databasePath === ":memory:" ? ":memory:" : `${databasePath}.bunqueue.sqlite`;
}
