import { createServerApp } from "./app";
import { createDatabase, ensureSchema } from "./db/client";
import { createHaloRunService } from "./halo/runQueue";
import { createLangfuseImportService } from "./langfuse/importQueue";
import { createLiveEventStore } from "./live/events";
import { startLiveWebSocketServer } from "./live/server";
import { appRouter } from "./router";
import { INGEST_HOSTNAME, INGEST_PORT, LIVE_WS_PORT } from "./telemetry/types";

type StartTelemetryServerOptions = {
  dbPath?: string;
  enableLiveServer?: boolean;
  hostname?: string;
  port?: number;
  wsPort?: number;
};

export function startTelemetryServer(options: StartTelemetryServerOptions = {}) {
  const hostname = options.hostname ?? INGEST_HOSTNAME;
  const port = options.port ?? INGEST_PORT;
  const database = createDatabase(options.dbPath);

  ensureSchema(database.sqlite);

  const live = createLiveEventStore(database.sqlite);
  const langfuseImports = createLangfuseImportService({ database, live });
  const haloRuns = createHaloRunService({ database, live });
  const requestedWsPort = options.wsPort ?? LIVE_WS_PORT;
  const configuredLiveUrl = `ws://${hostname}:${requestedWsPort}`;
  const liveServer =
    options.enableLiveServer === false
      ? null
      : startLiveWebSocketServer({
          createContext: () => ({
            database,
            haloRuns,
            live,
            liveUrl: configuredLiveUrl,
          }),
          hostname,
          port: requestedWsPort,
          router: appRouter,
        });
  const liveUrl = liveServer?.url ?? configuredLiveUrl;
  const app = createServerApp(database, live, liveUrl, langfuseImports, haloRuns);
  const server = Bun.serve({
    hostname,
    port,
    fetch: app.fetch,
  });

  return {
    app,
    database,
    hostname,
    haloRuns,
    live,
    langfuseImports,
    liveServer,
    liveUrl,
    port: server.port,
    server,
  };
}
