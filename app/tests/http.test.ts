import { createTRPCProxyClient, httpBatchLink } from "@trpc/client";
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { createServerApp } from "../src/server/app";
import {
  createDatabase,
  ensureSchema,
  type DatabaseHandle,
} from "../src/server/db/client";
import type { AppRouter } from "../src/server/router";
import { TRACE_ID, makeTracePayload } from "./support/otlp-fixtures";

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
    links: [
      httpBatchLink({
        url: `${baseUrl}/trpc`,
      }),
    ],
  });
});

afterEach(() => {
  server.stop(true);
  database.sqlite.close(false);
});

describe("telemetry HTTP server", () => {
  test("accepts OTLP JSON and serves trace queries over tRPC", async () => {
    const response = await fetch(`${baseUrl}/v1/traces`, {
      body: JSON.stringify(makeTracePayload()),
      headers: { "content-type": "application/json" },
      method: "POST",
    });

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({});

    const info = await client.telemetry.info.query();
    expect(info.traceCount).toBe(1);
    expect(info.spanCount).toBe(2);

    const traces = await client.traces.list.query({ limit: 10 });
    expect(traces.traces[0]?.traceId).toBe(TRACE_ID);
  });

  test("accepts gzip-compressed OTLP JSON", async () => {
    const compressed = Bun.gzipSync(JSON.stringify(makeTracePayload()));
    const response = await fetch(`${baseUrl}/v1/traces`, {
      body: compressed,
      headers: {
        "content-encoding": "gzip",
        "content-type": "application/json",
      },
      method: "POST",
    });

    expect(response.status).toBe(200);
    const info = await client.telemetry.info.query();
    expect(info.traceCount).toBe(1);
    expect(info.spanCount).toBe(2);
  });

  test("rejects unsupported protobuf ingest", async () => {
    const response = await fetch(`${baseUrl}/v1/traces`, {
      body: new Uint8Array([0, 1, 2, 3]),
      headers: { "content-type": "application/x-protobuf" },
      method: "POST",
    });

    expect(response.status).toBe(415);
  });

  test("rejects oversized payloads", async () => {
    const response = await fetch(`${baseUrl}/v1/traces`, {
      body: "x".repeat(4 * 1024 * 1024 + 1),
      headers: { "content-type": "application/json" },
      method: "POST",
    });

    expect(response.status).toBe(413);
  });
});
