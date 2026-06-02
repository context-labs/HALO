import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  createDatabase,
  ensureSchema,
  type DatabaseHandle,
} from "../src/server/db/client";
import { createLiveEventStore } from "../src/server/live/events";
import { appRouter } from "../src/server/router";
import { ingestTelemetry } from "../src/server/telemetry/storage";
import { TRACE_ID, makeTracePayload } from "./support/otlp-fixtures";

let database: DatabaseHandle;
let caller: ReturnType<typeof appRouter.createCaller>;

beforeEach(() => {
  database = createDatabase(":memory:");
  ensureSchema(database.sqlite);
  caller = appRouter.createCaller({
    database,
    live: createLiveEventStore(database.sqlite),
    liveUrl: "ws://127.0.0.1:8800",
  });
});

afterEach(() => {
  database.sqlite.close(false);
});

describe("telemetry router", () => {
  test("queries ingested traces, spans, search, and facets", async () => {
    const payload = makeTracePayload();
    payload.resourceSpans?.[0]?.scopeSpans?.[0]?.spans?.[0]?.attributes?.push({
      key: "langfuse.trace.id",
      value: { stringValue: TRACE_ID },
    });
    const result = ingestTelemetry(database.sqlite, {
      body: JSON.stringify(payload),
      contentEncoding: "identity",
      sizeBytes: 128,
    });

    expect(result.acceptedSpanCount).toBe(2);
    expect(result.traceCount).toBe(1);

    const info = await caller.telemetry.info();
    expect(info.ingestUrl).toBe("http://127.0.0.1:8799/v1/traces");
    expect(info.traceCount).toBe(1);
    expect(info.spanCount).toBe(2);

    const listed = await caller.traces.list({
      filters: { agents: ["Local agent"] },
      limit: 10,
      sortBy: "start_time",
      sortOrder: "desc",
    });
    expect(listed.totalCount).toBe(1);
    expect(listed.traces[0]?.traceId).toBe(TRACE_ID);
    expect(listed.traces[0]?.llmSpanCount).toBe(1);
    expect(listed.traces[0]?.totalTokens).toBe(30);
    expect(listed.traces[0]?.source).toBe("local");

    const trace = await caller.traces.get({ traceId: TRACE_ID });
    expect(trace.rootSpanName).toBe("agent.run");
    expect(trace.hasError).toBe(false);

    const spans = await caller.traces.getSpans({ traceId: TRACE_ID });
    expect(spans.spans).toHaveLength(2);
    expect(spans.tree).toHaveLength(1);
    expect(spans.tree[0]?.children).toHaveLength(1);
    expect(spans.spans.some((span) => span.llmModelName === "gpt-5-mini")).toBe(
      true,
    );

    const search = await caller.traces.search({ query: "gpt-5", limit: 10 });
    expect(search.results[0]?.trace.traceId).toBe(TRACE_ID);

    const sourceFiltered = await caller.traces.list({
      filters: { sources: ["local"] },
      limit: 10,
    });
    expect(sourceFiltered.totalCount).toBe(1);

    const facets = await caller.traces.facets({
      facetIds: ["agent_name", "llm_model_name", "service_name", "source"],
    });
    expect(facets.categorical.agent_name?.[0]?.value).toBe("Local agent");
    expect(facets.categorical.llm_model_name?.[0]?.value).toBe("gpt-5-mini");
    expect(facets.categorical.source?.[0]?.value).toBe("local");

    const listedSpans = await caller.spans.list({
      filters: { observationKinds: ["LLM"], sources: ["local"], traceId: TRACE_ID },
      limit: 10,
    });
    expect(listedSpans.spans).toHaveLength(1);
    expect(listedSpans.spans[0]?.spanName).toBe("openai.chat.completions");

    const spanFacets = await caller.spans.facets({ facetIds: ["source"] });
    expect(spanFacets.categorical.source?.[0]?.value).toBe("local");

    ingestTelemetry(database.sqlite, {
      body: JSON.stringify(makeImportedRootPayload()),
      contentEncoding: "langfuse-import",
      sizeBytes: 128,
    });

    const mixedTrace = await caller.traces.get({ traceId: TRACE_ID });
    expect(mixedTrace.source).toBe("local");
    expect(mixedTrace.sourceTraceId).toBeNull();
    expect(mixedTrace.rootSpanName).toBe("agent.run");

    const cleared = await caller.telemetry.clearData();
    expect(cleared.traceCount).toBe(1);
    expect(cleared.spanCount).toBe(3);

    const emptyInfo = await caller.telemetry.info();
    expect(emptyInfo.traceCount).toBe(0);
    expect(emptyInfo.spanCount).toBe(0);
    expect(emptyInfo.lastBatch).toBeNull();

    const emptyList = await caller.traces.list({ limit: 10 });
    expect(emptyList.totalCount).toBe(0);
  });

  test("groups traces into sessions and returns full session detail", async () => {
    ingestTelemetry(database.sqlite, {
      body: JSON.stringify(
        makeSessionPayload({
          childHasSession: false,
          serviceName: "service-a",
          sessionId: "session-1",
          traceId: "11111111111111111111111111111111",
        }),
      ),
      contentEncoding: "identity",
      sizeBytes: 256,
    });
    ingestTelemetry(database.sqlite, {
      body: JSON.stringify(
        makeSessionPayload({
          error: true,
          serviceName: "service-b",
          sessionId: "session-1",
          source: "langfuse",
          traceId: "22222222222222222222222222222222",
        }),
      ),
      contentEncoding: "langfuse-import",
      sizeBytes: 256,
    });
    ingestTelemetry(database.sqlite, {
      body: JSON.stringify(
        makeSessionPayload({
          serviceName: "unsessioned-service",
          sessionId: null,
          traceId: "33333333333333333333333333333333",
        }),
      ),
      contentEncoding: "identity",
      sizeBytes: 256,
    });

    const sessions = await caller.sessions.list({
      limit: 10,
      sortBy: "last_activity",
      sortOrder: "desc",
    });
    expect(sessions.totalCount).toBe(1);
    expect(sessions.sessions[0]?.sessionId).toBe("session-1");
    expect(sessions.sessions[0]?.traceCount).toBe(2);
    expect(sessions.sessions[0]?.spanCount).toBe(4);
    expect(sessions.sessions[0]?.llmSpanCount).toBe(2);
    expect(sessions.sessions[0]?.hasError).toBe(true);
    expect(sessions.sessions[0]?.serviceNames.sort()).toEqual([
      "service-a",
      "service-b",
    ]);
    expect(sessions.sessions[0]?.sources.sort()).toEqual(["langfuse", "local"]);

    const filtered = await caller.sessions.list({
      filters: { serviceNames: ["service-a"] },
      limit: 10,
    });
    expect(filtered.totalCount).toBe(1);

    const noMatch = await caller.sessions.list({
      filters: { serviceNames: ["service-c"] },
      limit: 10,
    });
    expect(noMatch.totalCount).toBe(0);

    const search = await caller.sessions.search({
      filters: { serviceNames: ["service-a"] },
      limit: 10,
      query: "service-b",
    });
    expect(search.totalCount).toBe(1);

    const session = await caller.sessions.get({ sessionId: "session-1" });
    expect(session.serviceNames.length).toBe(2);

    const traces = await caller.sessions.getTraces({
      limit: 10,
      sessionId: "session-1",
    });
    expect(traces.traces).toHaveLength(2);

    const spans = await caller.sessions.getSpans({
      limit: 10,
      sessionId: "session-1",
    });
    expect(spans.spans).toHaveLength(4);
    expect(spans.spans.some((span) => span.sessionId == null)).toBe(true);
  });

  test("accepts empty OTLP payloads", async () => {
    ingestTelemetry(database.sqlite, {
      body: "",
      contentEncoding: "identity",
      sizeBytes: 0,
    });

    const info = await caller.telemetry.info();
    expect(info.traceCount).toBe(0);
    expect(info.spanCount).toBe(0);
    expect(info.lastBatch?.acceptedSpanCount).toBe(0);
  });
});

function makeSessionPayload(input: {
  childHasSession?: boolean;
  error?: boolean;
  serviceName: string;
  sessionId: string | null;
  source?: "local" | "langfuse";
  traceId: string;
}) {
  const rootSpanId = `${input.traceId.slice(0, 15)}a`;
  const childSpanId = `${input.traceId.slice(16, 31)}b`;
  const rootAttributes = [
    { key: "openinference.span.kind", value: { stringValue: "AGENT" } },
    { key: "agent.name", value: { stringValue: "Session agent" } },
    ...(input.sessionId
      ? [{ key: "session.id", value: { stringValue: input.sessionId } }]
      : []),
    ...(input.source === "langfuse"
      ? [
          { key: "halo.source", value: { stringValue: "langfuse" } },
          {
            key: "halo.source.connection_name",
            value: { stringValue: "Imported Langfuse" },
          },
          {
            key: "halo.source.trace_id",
            value: { stringValue: input.traceId },
          },
        ]
      : []),
  ];
  return {
    resourceSpans: [
      {
        resource: {
          attributes: [
            { key: "service.name", value: { stringValue: input.serviceName } },
          ],
        },
        scopeSpans: [
          {
            scope: { name: "@inference/tracing", version: "1.0.0" },
            spans: [
              {
                attributes: rootAttributes,
                endTimeUnixNano: "1710000000800000000",
                kind: 2,
                name: `${input.serviceName}.run`,
                spanId: rootSpanId,
                startTimeUnixNano: "1710000000000000000",
                status: {
                  code: input.error ? "STATUS_CODE_ERROR" : "STATUS_CODE_OK",
                },
                traceId: input.traceId,
              },
              {
                attributes: [
                  { key: "openinference.span.kind", value: { stringValue: "LLM" } },
                  { key: "llm.model_name", value: { stringValue: "gpt-session" } },
                  { key: "llm.token_count.total", value: { intValue: 12 } },
                  ...(input.childHasSession !== false && input.sessionId
                    ? [{ key: "session.id", value: { stringValue: input.sessionId } }]
                    : []),
                ],
                endTimeUnixNano: "1710000000700000000",
                kind: 3,
                name: `${input.serviceName}.llm`,
                parentSpanId: rootSpanId,
                spanId: childSpanId,
                startTimeUnixNano: "1710000000100000000",
                status: { code: "STATUS_CODE_OK" },
                traceId: input.traceId,
              },
            ],
          },
        ],
      },
    ],
  };
}

function makeImportedRootPayload() {
  return {
    resourceSpans: [
      {
        resource: {
          attributes: [
            { key: "service.name", value: { stringValue: "langfuse-import" } },
          ],
        },
        scopeSpans: [
          {
            scope: { name: "langfuse-import", version: "1" },
            spans: [
              {
                attributes: [
                  { key: "halo.source", value: { stringValue: "langfuse" } },
                  {
                    key: "halo.source.connection_name",
                    value: { stringValue: "Local Langfuse" },
                  },
                  {
                    key: "halo.source.import_job_id",
                    value: { stringValue: "import-job-1" },
                  },
                  {
                    key: "halo.source.trace_id",
                    value: { stringValue: TRACE_ID },
                  },
                  {
                    key: "openinference.span.kind",
                    value: { stringValue: "AGENT" },
                  },
                ],
                endTimeUnixNano: "1710000000800000000",
                kind: "SPAN_KIND_INTERNAL",
                name: "agent.run imported from Langfuse",
                spanId: "dddddddddddddddd",
                startTimeUnixNano: "1710000000000000000",
                status: { code: "STATUS_CODE_OK" },
                traceId: TRACE_ID,
              },
            ],
          },
        ],
      },
    ],
  };
}
