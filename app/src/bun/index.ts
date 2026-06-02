import Electrobun, { ApplicationMenu, BrowserWindow } from "electrobun/bun";
import { configureDesktopRuntimeEnv } from "./desktopRuntime";
import { startTelemetryServer } from "../server/start";

const runtimePaths = configureDesktopRuntimeEnv();
const api = startTelemetryServer({
  dbPath: runtimePaths.dbPath,
  hostname: "127.0.0.1",
  port: 8799,
});

ApplicationMenu.setApplicationMenu([
  {
    submenu: [{ role: "quit" }],
  },
  {
    label: "Edit",
    submenu: [
      { role: "undo" },
      { role: "redo" },
      { type: "separator" },
      { role: "cut" },
      { role: "copy" },
      { role: "paste" },
      { role: "pasteAndMatchStyle" },
      { role: "delete" },
      { role: "selectAll" },
    ],
  },
]);

Electrobun.events.on("before-quit", () => {
  void api.langfuseImports.close(true);
  void api.haloRuns.close(true);
  api.liveServer?.stop();
  api.server.stop(true);
  api.database.sqlite.close(false);
});

const viewUrl = process.env.HALO_VIEW_URL ?? "views://mainview/_shell.html";

new BrowserWindow({
  title: "HALO",
  url: viewUrl,
  frame: {
    x: 0,
    y: 0,
    width: 1040,
    height: 760,
  },
  titleBarStyle: "hiddenInset",
  trafficLightOffset: {
    x: 18,
    y: 16,
  },
});

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
