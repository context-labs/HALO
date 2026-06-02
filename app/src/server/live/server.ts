import { applyWSSHandler } from "@trpc/server/adapters/ws";
import { WebSocketServer } from "ws";
import type { AppRouter, TRPCContext } from "../router";

type StartLiveWebSocketServerOptions = {
  createContext: () => TRPCContext | Promise<TRPCContext>;
  hostname: string;
  port: number;
  router: AppRouter;
};

export function startLiveWebSocketServer(
  options: StartLiveWebSocketServerOptions,
) {
  const server = new WebSocketServer({
    host: options.hostname,
    port: options.port,
  });
  const handler = applyWSSHandler({
    createContext: options.createContext,
    keepAlive: {
      enabled: true,
      pingMs: 30_000,
      pongWaitMs: 5_000,
    },
    router: options.router,
    wss: server,
  });

  const address = server.address();
  const port =
    typeof address === "object" && address !== null ? address.port : options.port;

  return {
    handler,
    hostname: options.hostname,
    port,
    server,
    stop() {
      handler.broadcastReconnectNotification();
      server.close();
    },
    url: `ws://${options.hostname}:${port}`,
  };
}
