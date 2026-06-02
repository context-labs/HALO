import { afterEach, describe, expect, test } from "bun:test";
import { Hono } from "hono";
import {
  basicAuthHeader,
  buildTraceListPath,
  discoverLangfuse,
  normalizeLangfuseBaseUrl,
} from "../src/server/langfuse/client";
import { createLangfuseImportService } from "../src/server/langfuse/importQueue";
import { langfuseTraceToOtlp, toOtelSpanId } from "../src/server/langfuse/mapper";
import {
  getLangfuseImportJob,
  saveLangfuseConnection,
} from "../src/server/langfuse/storage";
import { createDatabase, ensureSchema } from "../src/server/db/client";
import { createLiveEventStore } from "../src/server/live/events";
import { getSpansForTrace, getTrace } from "../src/server/telemetry/storage";

const PUBLIC_KEY = "lf_pk_test";
const SECRET_KEY = "lf_sk_test";

let servers: Bun.Server<undefined>[] = [];

afterEach(() => {
  for (const server of servers) server.stop(true);
  servers = [];
});

describe("Langfuse API helpers", () => {
  test("normalizes hosts and creates Basic Auth", () => {
    expect(normalizeLangfuseBaseUrl(" http://localhost:3001/ ")).toBe(
      "http://localhost:3001",
    );
    expect(basicAuthHeader(PUBLIC_KEY, SECRET_KEY)).toBe(
      `Basic ${Buffer.from(`${PUBLIC_KEY}:${SECRET_KEY}`).toString("base64")}`,
    );
  });

  test("builds trace list query filters", () => {
    const path = buildTraceListPath({
      filters: {
        environment: "production",
        fromTimestamp: "2026-05-01T00:00:00.000Z",
        tag: "agent",
        traceName: "agent.run",
      },
      limit: 25,
      page: 2,
    });
    expect(path).toContain("page=2");
    expect(path).toContain("limit=25");
    expect(path).toContain("environment=production");
    expect(path).toContain("tags=agent");
    expect(path).toContain("name=agent.run");
  });
});

describe("Langfuse mapping", () => {
  test("maps traces and observations into OTLP spans", () => {
    const otlp = langfuseTraceToOtlp(makeLangfuseTrace());
    const resourceAttrs = otlp.resourceSpans?.[0]?.resource?.attributes ?? [];
    const spans = otlp.resourceSpans?.[0]?.scopeSpans?.[0]?.spans ?? [];

    expect(
      resourceAttrs.some(
        (attribute) =>
          attribute.key === "service.name" &&
          attribute.value?.stringValue === "gator-agent",
      ),
    ).toBe(true);
    expect(spans).toHaveLength(3);
    expect(spans[0]?.name).toBe("agent.run");
    expect(spans[0]?.spanId).toBe("aaaaaaaaaaaaaaaa");
    expect(spans[0]?.parentSpanId).toBeUndefined();
    expect(
      spans.some(
        (span) => span.spanId === toOtelSpanId(`halo-root:${makeLangfuseTrace().id}`),
      ),
    ).toBe(false);
    expect(
      spans[0]?.attributes?.some(
        (attribute) =>
          attribute.key === "agent.name" &&
          attribute.value?.stringValue === "Gator Flue Agent",
      ),
    ).toBe(true);
    expect(spans[1]?.parentSpanId).toBe(spans[0]?.spanId);
    expect(
      spans[1]?.attributes?.some(
        (attribute) =>
          attribute.key === "llm.model_name" &&
          attribute.value?.stringValue === "anthropic/claude-3-5-sonnet",
      ),
    ).toBe(true);
    expect(
      spans[2]?.attributes?.some(
        (attribute) =>
          attribute.key === "tool.name" &&
          attribute.value?.stringValue === "contacts.list",
      ),
    ).toBe(true);
    expect(
      spans[0]?.attributes?.some(
        (attribute) =>
          attribute.key === "halo.source" &&
          attribute.value?.stringValue === "langfuse",
      ),
    ).toBe(true);
    expect(
      spans[0]?.attributes?.some(
        (attribute) =>
          attribute.key === "halo.source.trace_id" &&
          attribute.value?.stringValue === makeLangfuseTrace().id,
      ),
    ).toBe(true);
  });

  test("creates a fallback root only when a Langfuse trace has no observations", () => {
    const trace = { ...makeLangfuseTrace(), observations: [] };
    const otlp = langfuseTraceToOtlp(trace);
    const spans = otlp.resourceSpans?.[0]?.scopeSpans?.[0]?.spans ?? [];

    expect(spans).toHaveLength(1);
    expect(spans[0]?.spanId).toBe(toOtelSpanId(`halo-root:${trace.id}`));
    expect(spans[0]?.name).toBe("agent.run");
    expect(
      spans[0]?.attributes?.some(
        (attribute) =>
          attribute.key === "input.value" &&
          attribute.value?.stringValue?.includes("List contacts"),
      ),
    ).toBe(true);
  });
});

describe("Langfuse import queue", () => {
  test("imports paginated Langfuse traces through bunqueue into SQLite", async () => {
    const langfuse = startFakeLangfuse();
    const database = createDatabase(":memory:");
    ensureSchema(database.sqlite);
    const live = createLiveEventStore(database.sqlite);
    const service = createLangfuseImportService({ database, live });

    try {
      const discovery = await discoverLangfuse({
        baseUrl: langfuse.baseUrl,
        publicKey: PUBLIC_KEY,
        secretKey: SECRET_KEY,
      });
      const connection = saveLangfuseConnection(database.sqlite, {
        baseUrl: discovery.baseUrl,
        discovery,
        name: "Fake Langfuse",
        publicKey: PUBLIC_KEY,
        secretKey: SECRET_KEY,
      });

      const job = await service.start({
        connectionId: connection.id,
        filters: { fromTimestamp: "2026-05-01T00:00:00.000Z" },
      });
      const completed = await waitForImportJob(database.sqlite, job.id, "completed");

      expect(completed.importedTraces).toBe(1);
      expect(completed.importedObservations).toBe(3);
      const trace = getTrace(database.sqlite, makeLangfuseTrace().id);
      expect(trace?.rootSpanName).toBe("agent.run");
      expect(trace?.serviceName).toBe("gator-agent");
      expect(trace?.agentName).toBe("Gator Flue Agent");
      expect(trace?.llmSpanCount).toBe(1);
      expect(trace?.source).toBe("langfuse");
      expect(trace?.sourceConnectionName).toBe("Fake Langfuse");
      expect(trace?.sourceTraceId).toBe(makeLangfuseTrace().id);
      expect(trace?.sourceTags).toEqual(["agent", "local"]);
      const spans = getSpansForTrace(database.sqlite, {
        traceId: makeLangfuseTrace().id,
      });
      expect(spans.spans).toHaveLength(makeLangfuseTrace().observations.length);
      expect(
        spans.spans.some(
          (span) => span.spanId === toOtelSpanId(`halo-root:${makeLangfuseTrace().id}`),
        ),
      ).toBe(false);
      expect(spans.spans.some((span) => span.observationKind === "TOOL")).toBe(
        true,
      );
    } finally {
      await service.close(true);
      database.sqlite.close(false);
    }
  });
});

function startFakeLangfuse() {
  const app = new Hono();
  app.get("/api/public/health", (c) => c.json({ status: "OK" }));
  app.use("/api/public/*", async (c, next) => {
    if (c.req.header("authorization") !== basicAuthHeader(PUBLIC_KEY, SECRET_KEY)) {
      return c.json({ message: "unauthorized" }, 401);
    }
    await next();
  });
  app.get("/api/public/projects", (c) =>
    c.json({
      data: [
        {
          id: "project-1",
          name: "Project 1",
          organization: { id: "org-1", name: "Org 1" },
        },
      ],
    }),
  );
  app.get("/api/public/traces", (c) =>
    c.json({
      data: [makeLangfuseTraceListItem()],
      meta: { limit: 50, page: 1, totalItems: 1, totalPages: 1 },
    }),
  );
  app.get("/api/public/traces/:traceId", (c) => c.json(makeLangfuseTrace()));

  const server = Bun.serve({
    fetch: app.fetch,
    hostname: "127.0.0.1",
    port: 0,
  });
  servers.push(server);
  return { baseUrl: `http://127.0.0.1:${server.port}` };
}

async function waitForImportJob(
  sqlite: ReturnType<typeof createDatabase>["sqlite"],
  jobId: string,
  status: string,
) {
  const timeoutAt = Date.now() + 4_000;
  while (Date.now() < timeoutAt) {
    const job = getLangfuseImportJob(sqlite, jobId);
    if (job?.status === status) return job;
    if (job?.status === "failed") {
      throw new Error(job.errorMessage ?? "Import failed");
    }
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  throw new Error(`Timed out waiting for import job ${jobId}`);
}

function makeLangfuseTraceListItem() {
  return {
    environment: "production",
    id: "0123456789abcdef0123456789abcdef",
    name: "agent.run",
    release: "2026.05",
    sessionId: "session-1",
    tags: ["agent", "local"],
    timestamp: "2026-05-22T10:00:00.000Z",
    userId: "user-1",
    version: "1",
  };
}

function makeLangfuseTrace() {
  return {
    ...makeLangfuseTraceListItem(),
    htmlPath: "/project/project-1/traces/0123456789abcdef0123456789abcdef",
    input: { prompt: "List contacts" },
    metadata: {
      attributes: {
        "agent.id": "gator-flue-agent",
        "agent.name": "Gator Flue Agent",
      },
      resourceAttributes: {
        "service.name": "gator-agent",
        "service.version": "0.0.1",
      },
    },
    observations: [
      {
        costDetails: { total: 0.002 },
        endTime: "2026-05-22T10:00:03.000Z",
        id: "aaaaaaaaaaaaaaaa",
        input: { prompt: "List contacts" },
        level: "DEFAULT",
        name: "agent.run",
        output: { response: "Here are contacts" },
        startTime: "2026-05-22T10:00:00.000Z",
        traceId: "0123456789abcdef0123456789abcdef",
        type: "AGENT",
        usageDetails: { input: 12, output: 24, total: 36 },
      },
      {
        costDetails: { total: 0.001 },
        endTime: "2026-05-22T10:00:02.000Z",
        id: "1111111111111111",
        input: [{ role: "user", content: "List contacts" }],
        level: "DEFAULT",
        model: "anthropic/claude-3-5-sonnet",
        name: "anthropic.chat",
        output: [{ role: "assistant", content: "Here are contacts" }],
        parentObservationId: "aaaaaaaaaaaaaaaa",
        startTime: "2026-05-22T10:00:01.000Z",
        traceId: "0123456789abcdef0123456789abcdef",
        type: "GENERATION",
        usageDetails: { input: 10, output: 20, total: 30 },
      },
      {
        endTime: "2026-05-22T10:00:03.000Z",
        id: "2222222222222222",
        input: { limit: 5 },
        metadata: { function_name: "contacts.list" },
        name: "tool_call_contacts.list",
        output: { count: 5 },
        parentObservationId: "aaaaaaaaaaaaaaaa",
        startTime: "2026-05-22T10:00:02.100Z",
        traceId: "0123456789abcdef0123456789abcdef",
        type: "SPAN",
      },
    ],
    output: { response: "Here are contacts" },
  };
}
