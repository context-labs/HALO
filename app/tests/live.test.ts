import {
  createTRPCProxyClient,
  createWSClient,
  httpBatchLink,
  splitLink,
  wsLink,
} from "@trpc/client";
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { startTelemetryServer } from "../src/server/start";
import type { AppRouter } from "../src/server/router";
import type { LiveEvent } from "../src/server/live/events";
import { TRACE_ID, makeTracePayload } from "./support/otlp-fixtures";

let api: ReturnType<typeof startTelemetryServer>;
let client: ReturnType<typeof createTRPCProxyClient<AppRouter>>;
let wsClient: ReturnType<typeof createWSClient>;
let baseUrl: string;

beforeEach(() => {
  api = startTelemetryServer({
    dbPath: ":memory:",
    port: 0,
    wsPort: 0,
  });
  baseUrl = `http://127.0.0.1:${api.port}`;
  wsClient = createWSClient({ url: api.liveUrl });
  client = createTRPCProxyClient<AppRouter>({
    links: [
      splitLink({
        condition: (op) => op.type === "subscription",
        false: httpBatchLink({ url: `${baseUrl}/trpc` }),
        true: wsLink({ client: wsClient }),
      }),
    ],
  });
});

afterEach(() => {
  wsClient.close();
  api.liveServer?.stop();
  api.server.stop(true);
  api.database.sqlite.close(false);
});

describe("live telemetry subscriptions", () => {
  test("streams ingest, trace, and span events over tRPC WebSockets", async () => {
    const events: LiveEvent[] = [];
    let subscription: { unsubscribe(): void } | null = null;
    const received = new Promise<LiveEvent[]>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error("timed out waiting for live events")),
        2_000,
      );
      subscription = client.live.workspace.subscribe(undefined, {
        onData(event) {
          events.push(event.data);
          if (events.some((item) => item.eventType === "telemetry.changed")) {
            clearTimeout(timeout);
            subscription?.unsubscribe();
            resolve(events);
          }
        },
        onError(error) {
          clearTimeout(timeout);
          reject(error);
        },
      });
    });

    await new Promise((resolve) => setTimeout(resolve, 100));
    const response = await fetch(`${baseUrl}/v1/traces`, {
      body: JSON.stringify(makeTracePayload()),
      headers: { "content-type": "application/json" },
      method: "POST",
    });

    expect(response.status).toBe(200);
    const liveEvents = await received;

    expect(
      liveEvents.filter((event) => event.eventType === "span.upserted"),
    ).toHaveLength(2);
    expect(
      liveEvents.some(
        (event) =>
          event.eventType === "trace.upserted" && event.traceId === TRACE_ID,
      ),
    ).toBe(true);
    expect(liveEvents.at(-1)?.eventType).toBe("telemetry.changed");
  });
});
