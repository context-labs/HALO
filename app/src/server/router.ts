import { TRPCError, initTRPC, tracked } from "@trpc/server";
import { z } from "zod";
import type { DatabaseHandle } from "./db/client";
import { getHaloEngineStatus, installOrUpdateHaloEngine, testHaloProvider } from "./halo/engine";
import type { HaloRunService } from "./halo/runQueue";
import {
  deleteHaloProvider,
  getHaloProvider,
  listHaloProviders,
  listHaloRunEvents,
  saveHaloProvider,
} from "./halo/storage";
import { HALO_PROVIDER_TYPES, HALO_RUN_TARGET_TYPES } from "./halo/types";
import { discoverLangfuse } from "./langfuse/client";
import type { LangfuseImportService } from "./langfuse/importQueue";
import {
  deleteLangfuseConnection,
  getLangfuseConnection,
  getLangfuseImportJob,
  listLangfuseConnections,
  listLangfuseImportJobs,
  markLangfuseConnectionError,
  saveLangfuseConnection,
} from "./langfuse/storage";
import type { LiveEvent, LiveEventFilter, LiveEventStore } from "./live/events";
import {
  buildSpanTree,
  clearTelemetryData,
  getSpan,
  getSession,
  getSessionFacets,
  getSpansForSession,
  getSpanFacets,
  getSpansForTrace,
  getTelemetryInfo,
  getTrace,
  getTraceFacets,
  getTracesForSession,
  listSpans,
  listSessions,
  listTraces,
  searchSessions,
  searchTraces,
} from "./telemetry/storage";
import {
  OBSERVATION_KINDS,
  TRACE_SOURCES,
  type FacetId,
  type SessionSortKey,
  type SpanSortKey,
  type TraceSortKey,
} from "./telemetry/types";

export type TRPCContext = {
  database: DatabaseHandle;
  haloRuns?: HaloRunService;
  langfuseImports?: LangfuseImportService;
  live: LiveEventStore;
  liveUrl: string;
};

const t = initTRPC.context<TRPCContext>().create();

const observationKindSchema = z.enum(OBSERVATION_KINDS);
const traceSourceSchema = z.enum(TRACE_SOURCES);
const sortOrderSchema = z.enum(["asc", "desc"]);

const filtersSchema = z.object({
  startDate: z.coerce.date().optional(),
  endDate: z.coerce.date().optional(),
  observationKinds: z.array(observationKindSchema).optional(),
  llmProviders: z.array(z.string()).optional(),
  llmModelNames: z.array(z.string()).optional(),
  serviceNames: z.array(z.string()).optional(),
  deploymentEnvironments: z.array(z.string()).optional(),
  userIds: z.array(z.string()).optional(),
  sessionIds: z.array(z.string()).optional(),
  agents: z.array(z.string()).optional(),
  sources: z.array(traceSourceSchema).optional(),
  status: z.enum(["error", "ok"]).optional(),
  freeText: z.string().optional(),
  traceId: z.string().optional(),
  scope: z.enum(["all", "root", "entrypoint"]).optional(),
});

const traceSortKeySchema = z.enum([
  "start_time",
  "duration",
  "total_cost",
  "total_tokens",
  "span_count",
  "llm_span_count",
] satisfies TraceSortKey[]);

const spanSortKeySchema = z.enum([
  "start_time",
  "duration_ns",
  "cost_total",
  "total_tokens",
] satisfies SpanSortKey[]);

const sessionSortKeySchema = z.enum([
  "last_activity",
  "start_time",
  "duration",
  "total_cost",
  "total_tokens",
  "trace_count",
  "span_count",
  "llm_span_count",
] satisfies SessionSortKey[]);

const facetIdSchema = z.enum([
  "observation_kind",
  "status",
  "service_name",
  "deployment_environment",
  "agent_name",
  "agent_id",
  "llm_provider",
  "llm_model_name",
  "user_id",
  "session_id",
  "source",
  "duration_ns",
  "total_tokens",
  "input_tokens",
  "cache_read_tokens",
  "output_tokens",
  "cost_total",
  "span_count",
  "llm_span_count",
  "span_attributes",
  "resource_attributes",
] satisfies FacetId[]);

const pageInput = {
  cursor: z.string().nullable().optional(),
  limit: z.number().int().min(1).max(1000).optional(),
};

const lastEventInput = {
  lastEventId: z.string().nullable().optional(),
};

const langfuseConnectionInputSchema = z.object({
  baseUrl: z.string().min(1).optional(),
  id: z.string().min(1).optional(),
  name: z.string().min(1).max(120).optional(),
  publicKey: z.string().min(1).optional(),
  secretKey: z.string().min(1).optional(),
});

const langfuseTraceFiltersSchema = z.object({
  environment: z.string().optional(),
  fromTimestamp: z.string().optional(),
  release: z.string().optional(),
  sessionId: z.string().optional(),
  tag: z.string().optional(),
  toTimestamp: z.string().optional(),
  traceName: z.string().optional(),
  userId: z.string().optional(),
  version: z.string().optional(),
});

const haloProviderTypeSchema = z.enum(HALO_PROVIDER_TYPES);
const haloRunTargetTypeSchema = z.enum(HALO_RUN_TARGET_TYPES);
const haloProviderInputSchema = z.object({
  apiKey: z.string().min(1),
  baseUrl: z.string().min(1),
  headers: z.record(z.string(), z.string()).optional(),
  id: z.string().min(1).optional(),
  model: z.string().min(1),
  name: z.string().min(1).max(120),
  providerType: haloProviderTypeSchema,
});
const haloRunStartSchema = z.object({
  filters: filtersSchema.optional(),
  maxDepth: z.number().int().min(0).max(8).default(1),
  maxParallel: z.number().int().min(1).max(8).default(2),
  maxTurns: z.number().int().min(1).max(50).default(8),
  model: z.string().min(1).optional(),
  prompt: z.string().min(1).max(20000),
  providerId: z.string().min(1),
  targetType: haloRunTargetTypeSchema,
  title: z.string().max(160).optional(),
});

export const appRouter = t.router({
  telemetry: t.router({
    info: t.procedure.query(({ ctx }) =>
      getTelemetryInfo(ctx.database.sqlite, ctx.database.path, ctx.liveUrl),
    ),
    clearData: t.procedure.mutation(({ ctx }) =>
      clearTelemetryData(ctx.database.sqlite),
    ),
  }),

  live: t.router({
    workspace: t.procedure
      .input(z.object(lastEventInput).optional())
      .subscription(({ ctx, input, signal }) =>
        streamLiveEvents(ctx, {}, signal, input?.lastEventId),
      ),

    trace: t.procedure
      .input(
        z.object({
          ...lastEventInput,
          traceId: z.string().min(1),
        }),
      )
      .subscription(({ ctx, input, signal }) =>
        streamLiveEvents(
          ctx,
          { traceId: input.traceId },
          signal,
          input.lastEventId,
        ),
      ),

    importJob: t.procedure
      .input(
        z.object({
          ...lastEventInput,
          jobId: z.string().min(1),
        }),
      )
      .subscription(({ ctx, input, signal }) =>
        streamLiveEvents(
          ctx,
          { importJobId: input.jobId },
          signal,
          input.lastEventId,
        ),
      ),

    haloRun: t.procedure
      .input(
        z.object({
          ...lastEventInput,
          runId: z.string().min(1),
        }),
      )
      .subscription(({ ctx, input, signal }) =>
        streamLiveEvents(
          ctx,
          { haloRunId: input.runId },
          signal,
          input.lastEventId,
        ),
      ),
  }),

  halo: t.router({
    engine: t.router({
      installOrUpdate: t.procedure.mutation(async ({ ctx }) =>
        installOrUpdateHaloEngine(ctx.database),
      ),
      status: t.procedure.query(({ ctx }) => getHaloEngineStatus(ctx.database)),
    }),

    providers: t.router({
      delete: t.procedure
        .input(z.object({ id: z.string().min(1) }))
        .mutation(({ ctx, input }) => {
          deleteHaloProvider(ctx.database.sqlite, input.id);
          return { ok: true };
        }),
      list: t.procedure.query(({ ctx }) =>
        listHaloProviders(ctx.database.sqlite),
      ),
      save: t.procedure.input(haloProviderInputSchema).mutation(({ ctx, input }) =>
        saveHaloProvider(ctx.database.sqlite, input),
      ),
      test: t.procedure
        .input(z.object({ id: z.string().min(1) }))
        .mutation(async ({ ctx, input }) => {
          const provider = getHaloProvider(ctx.database.sqlite, input.id);
          if (!provider) {
            throw new TRPCError({
              code: "NOT_FOUND",
              message: "HALO model provider not found.",
            });
          }
          try {
            return await testHaloProvider(ctx.database, provider);
          } catch (error) {
            throw new TRPCError({
              code: "BAD_REQUEST",
              message:
                error instanceof Error
                  ? error.message
                  : "Could not connect to provider.",
            });
          }
        }),
    }),

    runs: t.router({
      cancel: t.procedure
        .input(z.object({ runId: z.string().min(1) }))
        .mutation(async ({ ctx, input }) => {
          const service = requireHaloRunService(ctx);
          const run = await service.cancel(input.runId);
          if (!run) {
            throw new TRPCError({ code: "NOT_FOUND", message: "HALO run not found" });
          }
          return run;
        }),
      get: t.procedure
        .input(z.object({ runId: z.string().min(1) }))
        .query(({ ctx, input }) => {
          const service = requireHaloRunService(ctx);
          const run = service.get(input.runId);
          if (!run) {
            throw new TRPCError({ code: "NOT_FOUND", message: "HALO run not found" });
          }
          return run;
        }),
      getEvents: t.procedure
        .input(
          z.object({
            limit: z.number().int().min(1).max(1000).optional(),
            runId: z.string().min(1),
          }),
        )
        .query(({ ctx, input }) =>
          listHaloRunEvents(ctx.database.sqlite, input.runId, input.limit),
        ),
      list: t.procedure
        .input(z.object({ limit: z.number().int().min(1).max(100).optional() }).optional())
        .query(({ ctx, input }) => {
          const service = requireHaloRunService(ctx);
          return service.list(input?.limit);
        }),
      preview: t.procedure
        .input(
          z.object({
            filters: filtersSchema.optional(),
            targetType: haloRunTargetTypeSchema,
          }),
        )
        .query(({ ctx, input }) => {
          const service = requireHaloRunService(ctx);
          return service.preview({
            filters: input.filters ?? {},
            targetType: input.targetType,
          });
        }),
      retry: t.procedure
        .input(z.object({ runId: z.string().min(1) }))
        .mutation(async ({ ctx, input }) => {
          const service = requireHaloRunService(ctx);
          const run = await service.retry(input.runId);
          if (!run) {
            throw new TRPCError({ code: "NOT_FOUND", message: "HALO run not found" });
          }
          return run;
        }),
      start: t.procedure.input(haloRunStartSchema).mutation(async ({ ctx, input }) => {
        const service = requireHaloRunService(ctx);
        try {
          return await service.start({
            ...input,
            filters: input.filters ?? {},
          });
        } catch (error) {
          throw new TRPCError({
            code: "BAD_REQUEST",
            message:
              error instanceof Error ? error.message : "Could not start HALO run.",
          });
        }
      }),
    }),
  }),

  langfuse: t.router({
    connections: t.router({
      list: t.procedure.query(({ ctx }) =>
        listLangfuseConnections(ctx.database.sqlite),
      ),

      saveAndDiscover: t.procedure
        .input(langfuseConnectionInputSchema)
        .mutation(async ({ ctx, input }) => {
          const existing = input.id
            ? getLangfuseConnection(ctx.database.sqlite, input.id)
            : null;
          const baseUrl = input.baseUrl ?? existing?.baseUrl;
          const publicKey = input.publicKey ?? existing?.publicKey;
          const secretKey = input.secretKey ?? existing?.secretKey;
          const name =
            input.name ??
            existing?.name ??
            (baseUrl ? safeUrlHost(baseUrl) : "Langfuse");

          if (!baseUrl || !publicKey || !secretKey) {
            throw new TRPCError({
              code: "BAD_REQUEST",
              message: "Langfuse URL, public key, and secret key are required.",
            });
          }

          let discovery: Awaited<ReturnType<typeof discoverLangfuse>>;
          try {
            discovery = await discoverLangfuse({
              baseUrl,
              publicKey,
              secretKey,
            });
          } catch (error) {
            const message =
              error instanceof Error
                ? error.message
                : "Could not connect to Langfuse.";
            try {
              markLangfuseConnectionError(ctx.database.sqlite, {
                baseUrl,
                error: message,
                id: input.id,
                publicKey,
              });
            } catch {
              // Preserve the actionable Langfuse connection error even if the
              // local database is currently unavailable.
            }
            throw new TRPCError({
              code: "BAD_REQUEST",
              message,
            });
          }

          try {
            const connection = saveLangfuseConnection(ctx.database.sqlite, {
              baseUrl: discovery.baseUrl,
              discovery,
              id: input.id,
              name,
              publicKey,
              secretKey,
            });
            return { connection, discovery };
          } catch (error) {
            const detail =
              error instanceof Error ? error.message : "Unknown SQLite error.";
            throw new TRPCError({
              code: "INTERNAL_SERVER_ERROR",
              message: `Connected to Langfuse, but could not save the connection locally: ${detail}`,
            });
          }
        }),

      delete: t.procedure
        .input(z.object({ id: z.string().min(1) }))
        .mutation(({ ctx, input }) => {
          deleteLangfuseConnection(ctx.database.sqlite, input.id);
          return { ok: true };
        }),
    }),

    imports: t.router({
      cancel: t.procedure
        .input(z.object({ jobId: z.string().min(1) }))
        .mutation(async ({ ctx, input }) => {
          const service = requireLangfuseImportService(ctx);
          const job = await service.cancel(input.jobId);
          if (!job) {
            throw new TRPCError({
              code: "NOT_FOUND",
              message: "Import job not found",
            });
          }
          return job;
        }),

      get: t.procedure
        .input(z.object({ jobId: z.string().min(1) }))
        .query(({ ctx, input }) => {
          const job = getLangfuseImportJob(ctx.database.sqlite, input.jobId);
          if (!job) {
            throw new TRPCError({
              code: "NOT_FOUND",
              message: "Import job not found",
            });
          }
          return job;
        }),

      list: t.procedure
        .input(z.object({ limit: z.number().int().min(1).max(100).optional() }).optional())
        .query(({ ctx, input }) =>
          listLangfuseImportJobs(ctx.database.sqlite, input?.limit ?? 20),
        ),

      start: t.procedure
        .input(
          z.object({
            connectionId: z.string().min(1),
            filters: langfuseTraceFiltersSchema.optional(),
          }),
        )
        .mutation(async ({ ctx, input }) => {
          const service = requireLangfuseImportService(ctx);
          return service.start({
            connectionId: input.connectionId,
            filters: input.filters ?? {},
          });
        }),
    }),
  }),

  traces: t.router({
    list: t.procedure
      .input(
        z.object({
          ...pageInput,
          filters: filtersSchema.optional(),
          sortBy: traceSortKeySchema.optional(),
          sortOrder: sortOrderSchema.optional(),
        }),
      )
      .query(({ ctx, input }) =>
        listTraces(ctx.database.sqlite, {
          cursor: input.cursor ?? null,
          filters: input.filters,
          limit: input.limit,
          sortBy: input.sortBy,
          sortOrder: input.sortOrder,
        }),
      ),

    search: t.procedure
      .input(
        z.object({
          ...pageInput,
          filters: filtersSchema.optional(),
          query: z.string().max(1000),
        }),
      )
      .query(({ ctx, input }) =>
        searchTraces(ctx.database.sqlite, {
          cursor: input.cursor ?? null,
          filters: input.filters,
          limit: input.limit,
          query: input.query,
        }),
      ),

    facets: t.procedure
      .input(
        z.object({
          facetIds: z.array(facetIdSchema).min(1).max(20),
        }),
      )
      .query(({ ctx, input }) =>
        getTraceFacets(ctx.database.sqlite, input.facetIds),
      ),

    get: t.procedure
      .input(z.object({ traceId: z.string().min(1) }))
      .query(({ ctx, input }) => {
        const trace = getTrace(ctx.database.sqlite, input.traceId);
        if (!trace) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Trace not found" });
        }
        return trace;
      }),

    getSpans: t.procedure
      .input(
        z.object({
          ...pageInput,
          traceId: z.string().min(1),
        }),
      )
      .query(({ ctx, input }) => {
        const result = getSpansForTrace(ctx.database.sqlite, {
          cursor: input.cursor ?? null,
          limit: input.limit,
          traceId: input.traceId,
        });
        return {
          ...result,
          tree: buildSpanTree(result.spans),
        };
      }),
  }),

  sessions: t.router({
    list: t.procedure
      .input(
        z.object({
          ...pageInput,
          filters: filtersSchema.optional(),
          sortBy: sessionSortKeySchema.optional(),
          sortOrder: sortOrderSchema.optional(),
        }),
      )
      .query(({ ctx, input }) =>
        listSessions(ctx.database.sqlite, {
          cursor: input.cursor ?? null,
          filters: input.filters,
          limit: input.limit,
          sortBy: input.sortBy,
          sortOrder: input.sortOrder,
        }),
      ),

    search: t.procedure
      .input(
        z.object({
          ...pageInput,
          filters: filtersSchema.optional(),
          query: z.string().max(1000),
        }),
      )
      .query(({ ctx, input }) =>
        searchSessions(ctx.database.sqlite, {
          cursor: input.cursor ?? null,
          filters: input.filters,
          limit: input.limit,
          query: input.query,
        }),
      ),

    facets: t.procedure
      .input(
        z.object({
          facetIds: z.array(facetIdSchema).min(1).max(20),
        }),
      )
      .query(({ ctx, input }) =>
        getSessionFacets(ctx.database.sqlite, input.facetIds),
      ),

    get: t.procedure
      .input(z.object({ sessionId: z.string().min(1) }))
      .query(({ ctx, input }) => {
        const session = getSession(ctx.database.sqlite, input.sessionId);
        if (!session) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Session not found" });
        }
        return session;
      }),

    getTraces: t.procedure
      .input(
        z.object({
          ...pageInput,
          sessionId: z.string().min(1),
        }),
      )
      .query(({ ctx, input }) =>
        getTracesForSession(ctx.database.sqlite, {
          cursor: input.cursor ?? null,
          limit: input.limit,
          sessionId: input.sessionId,
        }),
      ),

    getSpans: t.procedure
      .input(
        z.object({
          ...pageInput,
          sessionId: z.string().min(1),
        }),
      )
      .query(({ ctx, input }) => {
        const result = getSpansForSession(ctx.database.sqlite, {
          cursor: input.cursor ?? null,
          limit: input.limit,
          sessionId: input.sessionId,
        });
        return {
          ...result,
          tree: buildSpanTree(result.spans),
        };
      }),
  }),

  spans: t.router({
    list: t.procedure
      .input(
        z.object({
          ...pageInput,
          filters: filtersSchema.optional(),
          sortBy: spanSortKeySchema.optional(),
          sortOrder: sortOrderSchema.optional(),
        }),
      )
      .query(({ ctx, input }) =>
        listSpans(ctx.database.sqlite, {
          cursor: input.cursor ?? null,
          filters: input.filters,
          limit: input.limit,
          sortBy: input.sortBy,
          sortOrder: input.sortOrder,
        }),
      ),

    facets: t.procedure
      .input(
        z.object({
          facetIds: z.array(facetIdSchema).min(1).max(20),
        }),
      )
      .query(({ ctx, input }) => getSpanFacets(ctx.database.sqlite, input.facetIds)),

    get: t.procedure
      .input(
        z.object({
          spanId: z.string().min(1),
          traceId: z.string().min(1),
        }),
      )
      .query(({ ctx, input }) => {
        const span = getSpan(ctx.database.sqlite, input);
        if (!span) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Span not found" });
        }
        return span;
      }),
  }),
});

export type AppRouter = typeof appRouter;

function requireLangfuseImportService(ctx: TRPCContext): LangfuseImportService {
  if (!ctx.langfuseImports) {
    throw new TRPCError({
      code: "INTERNAL_SERVER_ERROR",
      message: "Langfuse import queue is not available.",
    });
  }
  return ctx.langfuseImports;
}

function requireHaloRunService(ctx: TRPCContext): HaloRunService {
  if (!ctx.haloRuns) {
    throw new TRPCError({
      code: "INTERNAL_SERVER_ERROR",
      message: "HALO run queue is not available.",
    });
  }
  return ctx.haloRuns;
}

function safeUrlHost(value: string) {
  try {
    return new URL(value).host || "Langfuse";
  } catch {
    return "Langfuse";
  }
}

async function* streamLiveEvents(
  ctx: TRPCContext,
  filter: LiveEventFilter,
  signal: AbortSignal | undefined,
  lastEventId: string | null | undefined,
) {
  const queue: LiveEvent[] = [];
  const seenIds = new Set<number>();
  let notify: (() => void) | null = null;
  const unsubscribe = ctx.live.subscribe((event) => {
    queue.push(event);
    notify?.();
    notify = null;
  }, filter);

  try {
    for (const event of ctx.live.replay(parseLastEventId(lastEventId), filter)) {
      seenIds.add(event.id);
      yield tracked(String(event.id), event);
    }

    while (!signal?.aborted) {
      const event = queue.shift();
      if (!event) {
        await waitForLiveEvent(signal, (nextNotify) => {
          notify = nextNotify;
        });
        continue;
      }
      if (seenIds.has(event.id)) continue;
      seenIds.add(event.id);
      yield tracked(String(event.id), event);
    }
  } finally {
    unsubscribe();
  }
}

function parseLastEventId(lastEventId: string | null | undefined) {
  if (!lastEventId) return null;
  const parsed = Number(lastEventId);
  return Number.isFinite(parsed) ? parsed : null;
}

function waitForLiveEvent(
  signal: AbortSignal | undefined,
  setNotify: (notify: () => void) => void,
) {
  if (signal?.aborted) return Promise.resolve();
  return new Promise<void>((resolve) => {
    const cleanup = () => {
      signal?.removeEventListener("abort", handleAbort);
    };
    const handleAbort = () => {
      cleanup();
      resolve();
    };
    setNotify(() => {
      cleanup();
      resolve();
    });
    signal?.addEventListener("abort", handleAbort, { once: true });
  });
}
