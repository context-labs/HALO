import Electrobun, {
  BrowserView,
  BrowserWindow,
  ContextMenu,
  Updater,
  Utils,
} from "electrobun/bun";
import { configureDesktopRuntimeEnv } from "./desktopRuntime";
import { installApplicationMenu } from "./appMenu";
import { loadWindowFrame, persistWindowFrame } from "./windowState";
import {
  APP_BUNDLE_ID,
  APP_NAME,
  APP_RELEASE_URL,
  type DesktopAppMetadata,
  type DesktopCommand,
  type DesktopNativeStatus,
  type HaloDesktopRPCSchema,
} from "../desktop/commands";
import { startTelemetryServer } from "../server/start";

const runtimePaths = configureDesktopRuntimeEnv();
const api = startTelemetryServer({
  dbPath: runtimePaths.dbPath,
  hostname: "127.0.0.1",
  port: 8799,
});

const desktopRpc = BrowserView.defineRPC<HaloDesktopRPCSchema>({
  maxRequestTime: 60_000,
  handlers: {
    requests: {
      checkForUpdates,
      getAppMetadata,
      openAppDataFolder: () => {
        const ok = Utils.openPath(runtimePaths.appDataDir);
        return { ok };
      },
      openExternal: ({ url }) => ({ ok: Utils.openExternal(url) }),
      revealDatabaseFile: () => {
        Utils.showItemInFolder(runtimePaths.dbPath);
        return { ok: true };
      },
      showNotification: ({ body, title }) => {
        Utils.showNotification({ body, silent: true, title });
        return { ok: true };
      },
      showRowContextMenu: (input) => {
        const label = input.kind === "trace" ? "Trace" : "Session";
        ContextMenu.showContextMenu([
          {
            label: `Copy ${label} ID`,
            action: "copy-context-value",
            data: {
              message: `${label} ID copied`,
              value: input.id,
            },
          },
          {
            label: "Copy Local Link",
            action: "copy-context-value",
            data: {
              message: `${label} link copied`,
              value:
                input.kind === "trace"
                  ? `#/traces?traceId=${encodeURIComponent(input.id)}`
                  : `#/sessions?sessionId=${encodeURIComponent(input.id)}`,
            },
          },
          ...(input.sourceUrl
            ? [
                { type: "separator" as const },
                {
                  label: "Open Langfuse Source",
                  action: "open-context-url",
                  data: {
                    url: input.sourceUrl,
                  },
                },
              ]
            : []),
        ]);
        return { ok: true };
      },
    },
    messages: {},
  },
});

function sendDesktopCommand(command: DesktopCommand) {
  try {
    desktopRpc.send.desktopCommand(command);
  } catch {
    // The renderer may not have finished wiring RPC during early startup.
  }
}

function sendNativeStatus(status: DesktopNativeStatus) {
  try {
    desktopRpc.send.nativeStatus(status);
  } catch {
    // Menu actions should stay harmless even if the window is still loading.
  }
}

installApplicationMenu({
  checkForUpdates: async () => {
    sendNativeStatus(await checkForUpdates());
  },
  openAppDataFolder: () => {
    const ok = Utils.openPath(runtimePaths.appDataDir);
    sendNativeStatus({
      status: ok ? "success" : "error",
      title: ok ? "Opened app data folder" : "Could not open app data folder",
      message: runtimePaths.appDataDir,
    });
  },
  openDocs: (url) => {
    const ok = Utils.openExternal(url);
    if (!ok) {
      sendNativeStatus({
        status: "error",
        title: "Could not open HALO docs",
        message: url,
      });
    }
  },
  quit: () => Utils.quit(),
  revealDatabaseFile: () => {
    Utils.showItemInFolder(runtimePaths.dbPath);
    sendNativeStatus({
      status: "info",
      title: "Revealed database file",
      message: runtimePaths.dbPath,
    });
  },
  sendCommand: sendDesktopCommand,
});

ContextMenu.on("context-menu-clicked", (event) => {
  const action = menuActionFromEvent(event);
  const data = menuDataFromEvent(event);
  if (action === "copy-context-value") {
    const value = typeof data.value === "string" ? data.value : "";
    if (!value) return;
    Utils.clipboardWriteText(value);
    sendNativeStatus({
      status: "success",
      title: typeof data.message === "string" ? data.message : "Copied",
      message: value,
    });
  }
  if (action === "open-context-url") {
    const url = typeof data.url === "string" ? data.url : "";
    if (!url) return;
    const ok = Utils.openExternal(url);
    if (!ok) {
      sendNativeStatus({
        status: "error",
        title: "Could not open source",
        message: url,
      });
    }
  }
});

Updater.onStatusChange((entry) => {
  sendNativeStatus({
    status: "info",
    title: "Updater",
    message: entry.message,
  });
});

Electrobun.events.on("before-quit", () => {
  windowState.stop();
  void api.langfuseImports.close(true);
  void api.haloRuns.close(true);
  api.liveServer?.stop();
  api.server.stop(true);
  api.database.sqlite.close(false);
});

const viewUrl = process.env.HALO_VIEW_URL ?? "views://mainview/_shell.html";
const defaultFrame = {
  x: 0,
  y: 0,
  width: 1040,
  height: 760,
};

const mainWindow = new BrowserWindow({
  title: APP_NAME,
  url: viewUrl,
  frame: loadWindowFrame(runtimePaths.appDataDir, defaultFrame),
  rpc: desktopRpc,
  titleBarStyle: "hiddenInset",
  trafficLightOffset: {
    x: 18,
    y: 16,
  },
});

const windowState = persistWindowFrame(runtimePaths.appDataDir, mainWindow);

console.log(`Trace ingest listening at http://${api.hostname}:${api.port}/v1/traces`);
console.log(`Trace API listening at http://${api.hostname}:${api.port}/trpc`);
console.log(`Trace live updates listening at ${api.liveUrl}`);
console.log(`Trace monitor view loaded from ${viewUrl}`);
console.log(`HALO app data stored at ${runtimePaths.appDataDir}`);
console.log(`HALO database stored at ${runtimePaths.dbPath}`);
if (runtimePaths.migratedLegacyFiles.length > 0) {
  console.log(
    `Migrated legacy bundle data from ${runtimePaths.legacyDataDir} to ${runtimePaths.appDataDir}`,
  );
}

async function getAppMetadata(): Promise<DesktopAppMetadata> {
  const fallback = {
    baseUrl: APP_RELEASE_URL,
    channel: process.env.HALO_RELEASE_CHANNEL ?? "dev",
    version: process.env.npm_package_version ?? "dev",
  };

  try {
    const localInfo = await Updater.getLocalInfo();
    return {
      appDataDir: runtimePaths.appDataDir,
      bundleId: APP_BUNDLE_ID,
      channel: localInfo.channel || fallback.channel,
      dbPath: runtimePaths.dbPath,
      ingestUrl: `http://${api.hostname}:${api.port}/v1/traces`,
      liveUrl: api.liveUrl,
      releaseUrl: localInfo.baseUrl || fallback.baseUrl,
      version: localInfo.version || fallback.version,
    };
  } catch {
    return {
      appDataDir: runtimePaths.appDataDir,
      bundleId: APP_BUNDLE_ID,
      channel: fallback.channel,
      dbPath: runtimePaths.dbPath,
      ingestUrl: `http://${api.hostname}:${api.port}/v1/traces`,
      liveUrl: api.liveUrl,
      releaseUrl: fallback.baseUrl,
      version: fallback.version,
    };
  }
}

async function checkForUpdates(): Promise<DesktopNativeStatus> {
  try {
    const update = await Updater.checkForUpdate();
    return {
      status: "update",
      title: update.updateAvailable ? "Update available" : "HALO is up to date",
      message: update.updateAvailable
        ? `Version ${update.version || "latest"} is available.`
        : "No newer release was found for this channel.",
      updateAvailable: update.updateAvailable,
      version: update.version,
    };
  } catch (error) {
    return {
      status: "error",
      title: "Could not check for updates",
      message: error instanceof Error ? error.message : String(error),
    };
  }
}

function menuActionFromEvent(event: unknown) {
  const data = menuEventData(event);
  const action = data.action;
  return typeof action === "string" ? action : undefined;
}

function menuDataFromEvent(event: unknown) {
  const root = menuEventData(event);
  const data = root.data;
  return data && typeof data === "object" ? (data as Record<string, unknown>) : {};
}

function menuEventData(event: unknown) {
  if (!event || typeof event !== "object") return {};
  const data = (event as { data?: unknown }).data;
  if (data && typeof data === "object") return data as Record<string, unknown>;
  return event as Record<string, unknown>;
}
