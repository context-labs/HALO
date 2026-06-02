import { ApplicationMenu, type ApplicationMenuItemConfig } from "electrobun/bun";
import {
  APP_DOCS_URL,
  APP_NAME,
  isDesktopCommandName,
  type DesktopCommand,
} from "../desktop/commands";

type AppMenuHandlers = {
  checkForUpdates: () => void | Promise<void>;
  openAppDataFolder: () => void | Promise<void>;
  openDocs: (url: string) => void | Promise<void>;
  quit: () => void;
  revealDatabaseFile: () => void | Promise<void>;
  sendCommand: (command: DesktopCommand) => void;
};

export function installApplicationMenu(handlers: AppMenuHandlers) {
  ApplicationMenu.setApplicationMenu(buildApplicationMenu());
  ApplicationMenu.on("application-menu-clicked", (event) => {
    const action = menuActionFromEvent(event);
    if (!action) return;

    if (action === "open-docs") {
      void handlers.openDocs(APP_DOCS_URL);
      return;
    }

    if (action === "open-app-data") {
      void handlers.openAppDataFolder();
      return;
    }

    if (action === "reveal-database") {
      void handlers.revealDatabaseFile();
      return;
    }

    if (action === "check-updates") {
      void handlers.checkForUpdates();
      return;
    }

    if (action === "quit-app") {
      handlers.quit();
      return;
    }

    if (isDesktopCommandName(action)) {
      handlers.sendCommand({ name: action, source: "menu" });
    }
  });
}

export function buildApplicationMenu(): ApplicationMenuItemConfig[] {
  return [
    {
      label: APP_NAME,
      submenu: [
        { label: `About ${APP_NAME}`, action: "about" },
        { label: "Check for Updates...", action: "check-updates" },
        { type: "separator" },
        {
          accelerator: "CommandOrControl+,",
          label: "Preferences...",
          action: "preferences",
        },
        { type: "separator" },
        { role: "hide" },
        { role: "hideOthers" },
        { role: "showAll" },
        { type: "separator" },
        {
          accelerator: "CommandOrControl+Q",
          label: `Quit ${APP_NAME}`,
          action: "quit-app",
        },
      ],
    },
    {
      label: "File",
      submenu: [
        {
          accelerator: "CommandOrControl+Shift+C",
          label: "Copy Ingest URL",
          action: "copy-ingest-url",
        },
        {
          accelerator: "CommandOrControl+Shift+I",
          label: "Import Langfuse Data...",
          action: "import-data",
        },
        { label: "Clear Telemetry Data...", action: "clear-data" },
        { type: "separator" },
        { label: "Close Window", role: "close" },
      ],
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
        { type: "separator" },
        { role: "selectAll" },
      ],
    },
    {
      label: "View",
      submenu: [
        {
          accelerator: "CommandOrControl+1",
          label: "Traces",
          action: "navigate-traces",
        },
        {
          accelerator: "CommandOrControl+2",
          label: "Sessions",
          action: "navigate-sessions",
        },
        {
          accelerator: "CommandOrControl+3",
          label: "Analysis",
          action: "navigate-analysis",
        },
        {
          accelerator: "CommandOrControl+4",
          label: "Settings",
          action: "navigate-settings",
        },
        { type: "separator" },
        {
          accelerator: "CommandOrControl+K",
          label: "Command Palette...",
          action: "command-palette",
        },
        {
          accelerator: "CommandOrControl+R",
          label: "Refresh",
          action: "refresh",
        },
        {
          accelerator: "CommandOrControl+Shift+L",
          label: "Follow Latest",
          action: "toggle-follow-latest",
        },
        { type: "separator" },
        { role: "toggleFullScreen" },
      ],
    },
    {
      label: "Window",
      submenu: [
        { role: "minimize" },
        { role: "zoom" },
        { type: "separator" },
        { role: "bringAllToFront" },
      ],
    },
    {
      label: "Help",
      submenu: [
        { label: "HALO Docs", action: "open-docs" },
        { type: "separator" },
        { label: "Open App Data Folder", action: "open-app-data" },
        { label: "Reveal Database File", action: "reveal-database" },
        { type: "separator" },
        { label: "Copy Diagnostics", action: "copy-diagnostics" },
      ],
    },
  ];
}

function menuActionFromEvent(event: unknown) {
  const data = menuEventData(event);
  const action = data.action;
  return typeof action === "string" ? action : undefined;
}

function menuEventData(event: unknown) {
  if (!event || typeof event !== "object") return {};
  const data = (event as { data?: unknown }).data;
  if (data && typeof data === "object") return data as Record<string, unknown>;
  return event as Record<string, unknown>;
}
