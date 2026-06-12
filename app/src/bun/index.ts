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
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import {
  APP_BUNDLE_ID,
  APP_NAME,
  APP_RELEASE_URL,
  type CodingToolAvailability,
  type DesktopAppMetadata,
  type DesktopCommand,
  type DesktopNativeStatus,
  type DesktopUpdateFlowStatus,
  type DesktopUpdatePrompt,
  type HaloDesktopRPCSchema,
} from "../desktop/commands";
import { startTelemetryServer } from "../server/start";

// A desktop app must not die because a background queue hiccupped. Anything
// that escapes to the process level gets logged; the queue failure handlers
// and startup interrupted-run sweeps own the actual recovery.
process.on("unhandledRejection", (error) => {
  console.error(
    "[halo] unhandled rejection:",
    error instanceof Error ? (error.stack ?? error.message) : error,
  );
});
process.on("uncaughtException", (error) => {
  console.error(
    "[halo] uncaught exception:",
    error instanceof Error ? (error.stack ?? error.message) : error,
  );
});

const runtimePaths = configureDesktopRuntimeEnv();
const api = (() => {
  try {
    return startTelemetryServer({
      dbPath: runtimePaths.dbPath,
      hostname: "127.0.0.1",
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    try {
      Utils.showNotification({
        body: message,
        silent: false,
        title: "HALO could not start its local server",
      });
    } catch {
      // Notifications are best-effort during early startup.
    }
    throw error;
  }
})();

const desktopRpc = BrowserView.defineRPC<HaloDesktopRPCSchema>({
  maxRequestTime: 60_000,
  handlers: {
    requests: {
      applyUpdate: () => startUpdateFlow(),
      checkForUpdates,
      detectCodingTools,
      getAppMetadata,
      snoozeUpdatePrompt: () => {
        snoozeUpdatePrompt();
        return { ok: true };
      },
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
                  : `#/traces?view=sessions&sessionId=${encodeURIComponent(input.id)}`,
            },
          },
          ...(input.sourceUrl
            ? [
                { type: "separator" as const },
                {
                  label: `Open ${input.sourceName || "Imported"} Source`,
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

// Downloads emit many progress entries — only failures surface as toasts;
// the update dialog owns the happy-path progress display.
Updater.onStatusChange((entry) => {
  if (entry.status !== "error" && entry.status !== "patch-failed") return;
  sendNativeStatus({
    status: "error",
    title: "Updater",
    message: entry.message,
  });
});

// ── automatic update prompts ──────────────────────────────────────────────
const UPDATE_CHECK_INITIAL_DELAY_MS = 30_000;
const UPDATE_CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000;
const UPDATE_PROMPT_SNOOZE_MS = 6 * 60 * 60 * 1000;

// In-memory on purpose: declining waits six hours OR an app restart,
// whichever comes first — a fresh process naturally clears the snooze.
let updatePromptSnoozedUntil = 0;
let updateFlowActive = false;

function snoozeUpdatePrompt() {
  updatePromptSnoozedUntil = Date.now() + UPDATE_PROMPT_SNOOZE_MS;
}

async function maybePromptForUpdate() {
  if (updateFlowActive) return;
  if (Date.now() < updatePromptSnoozedUntil) return;
  try {
    const update = await Updater.checkForUpdate();
    if (!update.updateAvailable) return;
    sendUpdatePrompt({ version: update.version || "latest" });
  } catch {
    // Dev builds and offline checks fail quietly; the next tick retries.
  }
}

/**
 * Kick off download + install. Responds immediately (downloads can outlive
 * the RPC timeout); progress and failures flow back as updateFlowStatus
 * messages, and success ends with the app relaunching itself.
 */
function startUpdateFlow(): { message?: string; ok: boolean } {
  if (updateFlowActive) {
    return { message: "An update is already in progress.", ok: false };
  }
  updateFlowActive = true;
  void (async () => {
    try {
      sendUpdateFlowStatus({ status: "downloading" });
      await Updater.downloadUpdate();
      sendUpdateFlowStatus({ status: "installing" });
      await Updater.applyUpdate();
      // applyUpdate quits and relaunches on success; reaching here without a
      // restart means the platform path declined to apply.
    } catch (error) {
      updateFlowActive = false;
      sendUpdateFlowStatus({
        message: error instanceof Error ? error.message : String(error),
        status: "failed",
      });
    }
  })();
  return { ok: true };
}

function sendUpdatePrompt(prompt: DesktopUpdatePrompt) {
  try {
    desktopRpc.send.updatePrompt(prompt);
  } catch {
    // The renderer may not have finished wiring RPC yet; the next check
    // interval will prompt again.
  }
}

function sendUpdateFlowStatus(status: DesktopUpdateFlowStatus) {
  try {
    desktopRpc.send.updateFlowStatus(status);
  } catch {
    // Best effort — the dialog also hears terminal failures via toasts.
  }
}

setTimeout(() => void maybePromptForUpdate(), UPDATE_CHECK_INITIAL_DELAY_MS);
setInterval(() => void maybePromptForUpdate(), UPDATE_CHECK_INTERVAL_MS);

Electrobun.events.on("before-quit", () => {
  windowState.stop();
  void api.langfuseImports.close(true);
  void api.phoenixImports.close(true);
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
  // The dots own the empty 56px header strip above the sidebar brand: x
  // aligns the first dot with the wordmark's left edge, y centers the ~12px
  // buttons in the strip. macOS fixes their size — only position is ours.
  trafficLightOffset: {
    x: 24,
    y: 22,
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

/**
 * Best-effort detection of installed coding tools. App-bundle checks cover
 * the URL-scheme handlers (which is what the deep links need); CLI binaries
 * on PATH are a softer signal that still lets "Copy prompt" make sense.
 */
function detectCodingTools(): CodingToolAvailability {
  const home = homedir();
  return {
    "claude-code": Boolean(
      existsSync(join(home, "Applications", "Claude Code URL Handler.app")) ||
        Bun.which("claude"),
    ),
    codex: Boolean(
      existsSync("/Applications/Codex.app") ||
        existsSync(join(home, "Applications", "Codex.app")) ||
        Bun.which("codex"),
    ),
    cursor: Boolean(
      existsSync("/Applications/Cursor.app") ||
        existsSync(join(home, "Applications", "Cursor.app")) ||
        Bun.which("cursor"),
    ),
  };
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
