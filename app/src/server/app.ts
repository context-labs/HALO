import { trpcServer } from "@hono/trpc-server";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { HTTPException } from "hono/http-exception";
import type { DatabaseHandle } from "./db/client";
import type { HaloRunService } from "./halo/runQueue";
import type { LangfuseImportService } from "./langfuse/importQueue";
import { createLiveEventStore, type LiveEventStore } from "./live/events";
import { appRouter } from "./router";
import { ingestTelemetry } from "./telemetry/storage";
import { LIVE_WS_URL } from "./telemetry/types";

const MAX_BODY_BYTES = 4 * 1024 * 1024;

export function createServerApp(
  database: DatabaseHandle,
  live: LiveEventStore = createLiveEventStore(database.sqlite),
  liveUrl = LIVE_WS_URL,
  langfuseImports?: LangfuseImportService,
  haloRuns?: HaloRunService,
) {
  const app = new Hono();

  app.use(
    "*",
    cors({
      origin: "*",
      allowHeaders: ["authorization", "content-encoding", "content-type"],
      allowMethods: ["GET", "POST", "OPTIONS"],
    }),
  );

  app.get("/health", (c) =>
    c.json({
      dbPath: database.path,
      ok: true,
      service: "halo-canvas-telemetry",
    }),
  );

  app.post("/v1/traces", async (c) => {
    const contentType = (c.req.header("content-type") ?? "").toLowerCase();
    if (!contentType.includes("application/json")) {
      throw new HTTPException(415, {
        message:
          "unsupported content-type: HALO currently accepts OTLP/JSON only",
      });
    }

    const { body, contentEncoding, sizeBytes } = await readDecompressedBody(c);
    ingestTelemetry(
      database.sqlite,
      {
        body,
        contentEncoding,
        sizeBytes,
      },
      live,
    );

    c.status(200);
    return c.json({});
  });

  app.use(
    "/trpc/*",
    trpcServer({
      router: appRouter,
      createContext: () => ({
        database,
        haloRuns,
        langfuseImports,
        live,
        liveUrl,
      }),
    }),
  );

  return app;
}

async function readDecompressedBody(c: {
  req: {
    arrayBuffer(): Promise<ArrayBuffer>;
    header(name: string): string | undefined;
  };
}) {
  const contentEncoding = (c.req.header("content-encoding") ?? "identity").toLowerCase();
  const declaredLength = Number(c.req.header("content-length") ?? "0");
  if (Number.isFinite(declaredLength) && declaredLength > MAX_BODY_BYTES) {
    throw new HTTPException(413, {
      message: `payload too large: Content-Length ${declaredLength} exceeds ${MAX_BODY_BYTES}`,
    });
  }

  const raw = await c.req.arrayBuffer();
  if (raw.byteLength > MAX_BODY_BYTES) {
    throw new HTTPException(413, {
      message: `payload too large: ${raw.byteLength} bytes exceeds ${MAX_BODY_BYTES}`,
    });
  }

  if (contentEncoding === "identity" || contentEncoding === "") {
    return {
      body: new TextDecoder().decode(raw),
      contentEncoding: "identity",
      sizeBytes: raw.byteLength,
    };
  }

  if (contentEncoding !== "gzip") {
    throw new HTTPException(415, {
      message: `unsupported Content-Encoding: ${contentEncoding}`,
    });
  }

  const stream = new Response(raw).body?.pipeThrough(
    new DecompressionStream("gzip"),
  );
  if (!stream) {
    throw new HTTPException(400, { message: "could not read gzip body" });
  }
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > MAX_BODY_BYTES) {
      await reader.cancel();
      throw new HTTPException(413, {
        message: `decompressed payload too large: ${total} bytes exceeds ${MAX_BODY_BYTES}`,
      });
    }
    chunks.push(value);
  }
  const merged = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return {
    body: new TextDecoder().decode(merged),
    contentEncoding,
    sizeBytes: raw.byteLength,
  };
}
