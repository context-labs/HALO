import { resolveServerPorts } from "./config";
import { createServerApp } from "./app";
import { createDatabase, ensureSchema } from "./db/client";
import { createHaloRunService } from "./halo/runQueue";
import { createLangfuseImportService } from "./langfuse/importQueue";
import { createPhoenixImportService } from "./phoenix/importQueue";
import { createLiveEventStore } from "./live/events";
import { startLiveWebSocketServer } from "./live/server";
import { appRouter } from "./router";
import { backfillTracePreviews } from "./telemetry/storage";
import { INGEST_HOSTNAME, TRACE_INGEST_PATH } from "./telemetry/types";

type StartTelemetryServerOptions = {
  dbPath?: string;
  enableLiveServer?: boolean;
  hostname?: string;
  port?: number;
  wsPort?: number;
};

export function startTelemetryServer(options: StartTelemetryServerOptions = {}) {
  const envPorts = resolveServerPorts();
  const hostname = options.hostname ?? INGEST_HOSTNAME;
  const port = options.port ?? envPorts.ingestPort;
  const database = createDatabase(options.dbPath);

  ensureSchema(database.sqlite);
  try {
    const backfilled = backfillTracePreviews(database.sqlite);
    if (backfilled > 0) {
      console.log(`[telemetry] backfilled previews for ${backfilled} traces`);
    }
  } catch (error) {
    console.error("[telemetry] preview backfill failed", error);
  }

  const live = createLiveEventStore(database.sqlite);
  const langfuseImports = createLangfuseImportService({ database, live });
  const phoenixImports = createPhoenixImportService({ database, live });
  const haloRuns = createHaloRunService({ database, live });
  const requestedWsPort = options.wsPort ?? envPorts.liveWsPort;
  const configuredLiveUrl = `ws://${hostname}:${requestedWsPort}`;
  const ingestUrl = `http://${hostname}:${port}${TRACE_INGEST_PATH}`;
  const liveServer =
    options.enableLiveServer === false
      ? null
      : guardPortInUse(requestedWsPort, "HALO_LIVE_WS_PORT", () =>
          startLiveWebSocketServer({
            createContext: () => ({
              database,
              haloRuns,
              ingestUrl,
              langfuseImports,
              live,
              liveUrl: configuredLiveUrl,
              phoenixImports,
            }),
            hostname,
            port: requestedWsPort,
            router: appRouter,
          }),
        );
  const liveUrl = liveServer?.url ?? configuredLiveUrl;
  const app = createServerApp(
    database,
    live,
    liveUrl,
    langfuseImports,
    haloRuns,
    ingestUrl,
    phoenixImports,
  );
  const server = guardPortInUse(port, "HALO_INGEST_PORT", () =>
    Bun.serve({
      hostname,
      port,
      fetch: app.fetch,
    }),
  );

  return {
    app,
    database,
    hostname,
    haloRuns,
    ingestUrl,
    live,
    langfuseImports,
    phoenixImports,
    liveServer,
    liveUrl,
    port: server.port,
    server,
  };
}

function guardPortInUse<T>(port: number, envVar: string, start: () => T): T {
  try {
    return start();
  } catch (error) {
    if (isAddressInUse(error)) {
      throw new Error(
        `Port ${port} is already in use (another HALO instance or a local dev server may hold it). ` +
          `Stop that process or set ${envVar} to a different port and restart HALO.`,
        { cause: error },
      );
    }
    throw error;
  }
}

function isAddressInUse(error: unknown) {
  return (
    error instanceof Error &&
    ("code" in error && (error as { code?: string }).code === "EADDRINUSE")
  );
}
