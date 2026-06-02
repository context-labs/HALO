import { createTRPCProxyClient, httpBatchLink } from "@trpc/client";
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  buildTestSpans,
  fireTestSpans,
  randomDurationMs,
} from "../scripts/fire-test-spans";
import { createServerApp } from "../src/server/app";
import {
  createDatabase,
  ensureSchema,
  type DatabaseHandle,
} from "../src/server/db/client";
import type { AppRouter } from "../src/server/router";

let database: DatabaseHandle;
let server: Bun.Server<undefined>;
let client: ReturnType<typeof createTRPCProxyClient<AppRouter>>;
let baseUrl: string;

beforeEach(() => {
  database = createDatabase(":memory:");
  ensureSchema(database.sqlite);
  const app = createServerApp(database);

  server = Bun.serve({
    fetch: app.fetch,
    hostname: "127.0.0.1",
    port: 0,
  });
  baseUrl = `http://127.0.0.1:${server.port}`;

  client = createTRPCProxyClient<AppRouter>({
    links: [httpBatchLink({ url: `${baseUrl}/trpc` })],
  });
});

afterEach(() => {
  server.stop(true);
  database.sqlite.close(false);
});

describe("fire test spans script", () => {
  test("generates durations between 1 and 3 seconds", () => {
    expect(randomDurationMs(() => 0)).toBe(1_000);
    expect(randomDurationMs(() => 0.999_999_999)).toBe(3_000);

    const values = [0, 0.1, 0.35, 0.7, 0.999_999_999];
    let index = 0;
    const spans = buildTestSpans({
      rng: () => values[index++ % values.length] ?? 0,
      spanCount: 10,
      traceId: "11111111111111111111111111111111",
    });

    expect(spans).toHaveLength(10);
    expect(spans.every((span) => span.durationMs >= 1_000)).toBe(true);
    expect(spans.every((span) => span.durationMs <= 3_000)).toBe(true);
  });

  test("fires 10 OTLP spans into the local ingest endpoint", async () => {
    const values = [0, 0.2, 0.4, 0.6, 0.8];
    let index = 0;
    const result = await fireTestSpans({
      delayMs: 0,
      endpoint: `${baseUrl}/v1/traces`,
      rng: () => values[index++ % values.length] ?? 0,
      spanCount: 10,
    });

    expect(result.spans).toHaveLength(10);
    expect(result.spans.every((span) => span.durationMs >= 1_000)).toBe(true);
    expect(result.spans.every((span) => span.durationMs <= 3_000)).toBe(true);

    const info = await client.telemetry.info.query();
    expect(info.traceCount).toBe(1);
    expect(info.spanCount).toBe(10);

    const trace = await client.traces.get.query({ traceId: result.traceId });
    expect(trace.spanCount).toBe(10);
    expect(trace.rootSpanName).toBe("test.random_duration_span.1");

    const spans = await client.traces.getSpans.query({
      limit: 20,
      traceId: result.traceId,
    });
    expect(spans.spans).toHaveLength(10);
    expect(spans.spans.map((span) => Math.round(span.durationMs))).toEqual(
      result.spans.map((span) => span.durationMs),
    );
  });
});
