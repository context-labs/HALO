export const APP_NAME = "HALO";
export const APP_BUNDLE_ID = "net.inference.halo";
export const APP_DOCS_URL = "https://github.com/context-labs/halo";
export const APP_RELEASE_URL = "https://inference.net/halo/releases";
export const DEFAULT_INGEST_URL = "http://127.0.0.1:8799/v1/traces";

export type WorkspaceRoute = "traces" | "analysis" | "settings";

export type DesktopCommandName =
  | "about"
  | "check-updates"
  | "clear-data"
  | "command-palette"
  | "copy-diagnostics"
  | "copy-ingest-url"
  | "import-data"
  | "navigate-analysis"
  | "navigate-sessions"
  | "navigate-settings"
  | "navigate-traces"
  | "open-app-data"
  | "open-docs"
  | "preferences"
  | "refresh"
  | "reveal-database"
  | "toggle-follow-latest";

export type DesktopCommand = {
  name: DesktopCommandName;
  source?: "menu" | "keyboard" | "palette" | "native";
};

export type DesktopNativeStatus =
  | {
      message: string;
      status: "error" | "info" | "success";
      title: string;
    }
  | {
      status: "update";
      title: string;
      message: string;
      updateAvailable: boolean;
      version?: string;
    };

/** Bun→view nudge that a newer release is ready to install. */
export type DesktopUpdatePrompt = {
  version: string;
};

/** Progress of an in-flight update install, for the prompt dialog. */
export type DesktopUpdateFlowStatus = {
  message?: string;
  status: "downloading" | "failed" | "installing";
};

export type DesktopAppMetadata = {
  appDataDir: string;
  bundleId: string;
  channel: string;
  dbPath: string;
  ingestUrl: string;
  liveUrl: string;
  releaseUrl: string;
  version: string;
};

export type DesktopRowContextMenuInput = {
  id: string;
  kind: "session" | "trace";
  /** Display name of the import source ("Langfuse", "Phoenix") for sourceUrl. */
  sourceName?: string | null;
  sourceUrl?: string | null;
};

export type CodingTool = "claude-code" | "codex" | "cursor";

export type CodingToolAvailability = Record<CodingTool, boolean>;

/**
 * Short prompt sent into the coding tool. Reports routinely exceed every
 * tool's URL length limit (Cursor 8k, Claude Code 5k), so the prompt
 * references the markdown report on disk instead of inlining it.
 */
export function buildCodingToolPrompt(reportPath: string) {
  return (
    `Read the HALO trace-analysis report at ${reportPath} and act on its findings ` +
    `in this codebase: fix the identified failures and implement the recommended improvements.`
  );
}

/**
 * Deep links verified against vendor docs:
 * - Cursor: cursor://anysphere.cursor-deeplink/prompt?text=… (prefills chat)
 * - Claude Code: claude-cli://open?q=… (opens a terminal with the prompt prefilled)
 * - Codex: codex://new?prompt=… (prefills the Codex app composer)
 * None of them auto-run; the user confirms inside the tool.
 */
export function buildCodingToolDeepLink(tool: CodingTool, reportPath: string) {
  const prompt = encodeURIComponent(buildCodingToolPrompt(reportPath));
  switch (tool) {
    case "cursor":
      return `cursor://anysphere.cursor-deeplink/prompt?text=${prompt}`;
    case "claude-code":
      return `claude-cli://open?q=${prompt}`;
    case "codex":
      return `codex://new?prompt=${prompt}`;
  }
}

export type HaloDesktopRPCSchema = {
  bun: {
    requests: {
      applyUpdate: {
        params: undefined;
        response: { message?: string; ok: boolean };
      };
      checkForUpdates: {
        params: undefined;
        response: DesktopNativeStatus;
      };
      snoozeUpdatePrompt: {
        params: undefined;
        response: { ok: boolean };
      };
      detectCodingTools: {
        params: undefined;
        response: CodingToolAvailability;
      };
      getAppMetadata: {
        params: undefined;
        response: DesktopAppMetadata;
      };
      openAppDataFolder: {
        params: undefined;
        response: { ok: boolean };
      };
      openExternal: {
        params: { url: string };
        response: { ok: boolean };
      };
      revealDatabaseFile: {
        params: undefined;
        response: { ok: boolean };
      };
      showNotification: {
        params: { body?: string; title: string };
        response: { ok: boolean };
      };
      showRowContextMenu: {
        params: DesktopRowContextMenuInput;
        response: { ok: boolean };
      };
    };
    messages: Record<never, never>;
  };
  webview: {
    requests: Record<never, never>;
    messages: {
      desktopCommand: DesktopCommand;
      nativeStatus: DesktopNativeStatus;
      updateFlowStatus: DesktopUpdateFlowStatus;
      updatePrompt: DesktopUpdatePrompt;
    };
  };
};

export type CommandPaletteItem = {
  command: DesktopCommandName;
  description: string;
  group: "Navigation" | "Data" | "App";
  keywords: string[];
  label: string;
  shortcut?: string;
};

export const commandPaletteItems: CommandPaletteItem[] = [
  {
    command: "navigate-traces",
    description: "Open the trace monitor.",
    group: "Navigation",
    keywords: ["trace", "monitor", "spans"],
    label: "Go to Traces",
    shortcut: "⌘1",
  },
  {
    command: "navigate-sessions",
    description: "Open grouped conversation sessions.",
    group: "Navigation",
    keywords: ["sessions", "conversation"],
    label: "Go to Sessions",
    shortcut: "⌘2",
  },
  {
    command: "navigate-analysis",
    description: "Open HALO analysis runs.",
    group: "Navigation",
    keywords: ["analysis", "halo", "runs"],
    label: "Go to Analysis",
    shortcut: "⌘3",
  },
  {
    command: "preferences",
    description: "Open local settings and model providers.",
    group: "Navigation",
    keywords: ["settings", "preferences", "providers"],
    label: "Open Settings",
    shortcut: "⌘,",
  },
  {
    command: "refresh",
    description: "Refresh the current workspace data.",
    group: "Data",
    keywords: ["reload", "refresh", "sync"],
    label: "Refresh Current View",
    shortcut: "⌘R",
  },
  {
    command: "copy-ingest-url",
    description: "Copy the local OTLP endpoint.",
    group: "Data",
    keywords: ["copy", "otlp", "endpoint", "ingest"],
    label: "Copy Ingest URL",
    shortcut: "⇧⌘C",
  },
  {
    command: "import-data",
    description: "Import historical traces from Langfuse or Phoenix.",
    group: "Data",
    keywords: ["langfuse", "phoenix", "arize", "import", "data"],
    label: "Import Data",
    shortcut: "⇧⌘I",
  },
  {
    command: "clear-data",
    description: "Open the telemetry clear confirmation.",
    group: "Data",
    keywords: ["clear", "delete", "telemetry"],
    label: "Clear Telemetry Data",
  },
  {
    command: "toggle-follow-latest",
    description: "Follow the newest trace as it arrives.",
    group: "Data",
    keywords: ["follow", "latest", "live"],
    label: "Toggle Follow Latest",
    shortcut: "⇧⌘L",
  },
  {
    command: "check-updates",
    description: "Manually check for a newer HALO build.",
    group: "App",
    keywords: ["update", "release"],
    label: "Check for Updates",
  },
  {
    command: "about",
    description: "Show version, paths, and diagnostics.",
    group: "App",
    keywords: ["about", "version", "diagnostics"],
    label: "About HALO",
  },
  {
    command: "copy-diagnostics",
    description: "Copy app paths and runtime details.",
    group: "App",
    keywords: ["diagnostics", "support", "debug"],
    label: "Copy Diagnostics",
  },
  {
    command: "open-app-data",
    description: "Open HALO's local application data folder.",
    group: "App",
    keywords: ["folder", "data", "support"],
    label: "Open App Data Folder",
  },
  {
    command: "reveal-database",
    description: "Reveal the local SQLite database file.",
    group: "App",
    keywords: ["database", "sqlite", "file"],
    label: "Reveal Database File",
  },
];

const commandNames = new Set(commandPaletteItems.map((item) => item.command));
commandNames.add("navigate-settings");
commandNames.add("open-docs");
commandNames.add("command-palette");

export function isDesktopCommandName(value: unknown): value is DesktopCommandName {
  return typeof value === "string" && commandNames.has(value as DesktopCommandName);
}

export function routeForCommand(
  command: DesktopCommandName,
): WorkspaceRoute | undefined {
  switch (command) {
    case "navigate-traces":
      return "traces";
    case "navigate-sessions":
      return "traces";
    case "navigate-analysis":
      return "analysis";
    case "navigate-settings":
    case "preferences":
      return "settings";
    default:
      return undefined;
  }
}

export function commandLabel(command: DesktopCommandName) {
  return (
    commandPaletteItems.find((item) => item.command === command)?.label ??
    command.replaceAll("-", " ")
  );
}

export function filterCommandPaletteItems(query: string) {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return commandPaletteItems;

  return commandPaletteItems.filter((item) => {
    const haystack = [
      item.label,
      item.description,
      item.group,
      ...item.keywords,
    ]
      .join(" ")
      .toLowerCase();
    return haystack.includes(normalized);
  });
}

export function routePath(route: WorkspaceRoute) {
  return `/${route}` as const;
}

export function desktopCommandForShortcut(
  key: string,
  shiftKey = false,
): DesktopCommandName | undefined {
  const normalizedKey = key.toLowerCase();
  if (normalizedKey === "k" && !shiftKey) return "command-palette";
  if (normalizedKey === "," && !shiftKey) return "preferences";
  if (normalizedKey === "1" && !shiftKey) return "navigate-traces";
  if (normalizedKey === "2" && !shiftKey) return "navigate-sessions";
  if (normalizedKey === "3" && !shiftKey) return "navigate-analysis";
  if (normalizedKey === "4" && !shiftKey) return "navigate-settings";
  if (normalizedKey === "r" && !shiftKey) return "refresh";
  if (normalizedKey === "c" && shiftKey) return "copy-ingest-url";
  if (normalizedKey === "i" && shiftKey) return "import-data";
  if (normalizedKey === "l" && shiftKey) return "toggle-follow-latest";
  return undefined;
}
