import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { useQueryClient } from "@tanstack/react-query";
import {
  Clipboard,
  Command,
  Database,
  ExternalLink,
  FolderOpen,
  Info,
  RefreshCcw,
  Search,
  Sparkles,
} from "lucide-react";

import {
  Badge,
  Button,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogRoot,
  DialogTitle,
  InferenceIcon,
  Input,
  Separator,
  cn,
  toast,
} from "~/lib/ui";
import { trpc } from "~/trpc";
import {
  APP_BUNDLE_ID,
  APP_DOCS_URL,
  APP_NAME,
  APP_RELEASE_URL,
  DEFAULT_INGEST_URL,
  commandLabel,
  desktopCommandForShortcut,
  filterCommandPaletteItems,
  routeForCommand,
  routePath,
  type CommandPaletteItem,
  type DesktopAppMetadata,
  type DesktopCommand,
  type DesktopCommandName,
  type DesktopNativeStatus,
} from "../../desktop/commands";
import {
  DESKTOP_COMMAND_EVENT,
  DESKTOP_NATIVE_STATUS_EVENT,
  dispatchTracePageCommand,
  getDesktopRpc,
  initializeDesktopBridge,
  type TracePageCommand,
} from "./desktopBridge";

type NavigateOptions = Parameters<ReturnType<typeof useNavigate>>[0];

export function DesktopCommandCenter() {
  const navigate = useNavigate();
  const routerState = useRouterState();
  const queryClient = useQueryClient();
  const infoQuery = trpc.telemetry.info.useQuery();
  const [aboutOpen, setAboutOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [paletteQuery, setPaletteQuery] = useState("");
  const [metadata, setMetadata] = useState<DesktopAppMetadata | null>(null);
  const notifiedLiveEvents = useRef(new Set<number>());

  const path = routerState.location.pathname;
  const search = routerState.location.search as Record<string, unknown>;
  const fallbackMetadata = useMemo<DesktopAppMetadata>(
    () => ({
      appDataDir: "Unavailable in browser preview",
      bundleId: APP_BUNDLE_ID,
      channel: import.meta.env.DEV ? "dev" : "stable",
      dbPath: infoQuery.data?.dbPath ?? "Unavailable",
      ingestUrl: infoQuery.data?.ingestUrl ?? DEFAULT_INGEST_URL,
      liveUrl: infoQuery.data?.liveUrl ?? "Unavailable",
      releaseUrl: APP_RELEASE_URL,
      version: import.meta.env.DEV ? "dev" : "unknown",
    }),
    [infoQuery.data?.dbPath, infoQuery.data?.ingestUrl, infoQuery.data?.liveUrl],
  );
  const appMetadata = metadata ?? fallbackMetadata;

  const loadMetadata = useCallback(async () => {
    const rpc = await getDesktopRpc();
    if (!rpc) {
      setMetadata(null);
      return;
    }
    try {
      setMetadata(await rpc.request.getAppMetadata());
    } catch {
      setMetadata(null);
    }
  }, []);

  useEffect(() => {
    void initializeDesktopBridge();
    void loadMetadata();
  }, [loadMetadata]);

  const navigateTo = useCallback(
    (command: DesktopCommandName) => {
      const route = routeForCommand(command);
      if (!route) return;
      void navigate({ to: routePath(route) } as NavigateOptions);
    },
    [navigate],
  );

  const dispatchTraceCommand = useCallback(
    (command: TracePageCommand) => {
      if (path === "/traces" || path === "/sessions") {
        dispatchTracePageCommand(command);
        return;
      }

      void navigate({ to: "/traces" } as NavigateOptions).then(() => {
        window.setTimeout(() => dispatchTracePageCommand(command), 80);
      });
    },
    [navigate, path],
  );

  const copyIngestUrl = useCallback(async () => {
    const ingestUrl = infoQuery.data?.ingestUrl ?? appMetadata.ingestUrl;
    await navigator.clipboard.writeText(ingestUrl);
    toast.success({
      title: "Ingest URL copied",
      description: "Paste it into your local agent telemetry config.",
    });
  }, [appMetadata.ingestUrl, infoQuery.data?.ingestUrl]);

  const copyDiagnostics = useCallback(async () => {
    const diagnostics = diagnosticsText(appMetadata);
    await navigator.clipboard.writeText(diagnostics);
    toast.success({
      title: "Diagnostics copied",
      description: "Version, paths, and runtime URLs are on the clipboard.",
    });
  }, [appMetadata]);

  const checkForUpdates = useCallback(async () => {
    const rpc = await getDesktopRpc();
    if (!rpc) {
      toast.info({
        title: "Updater unavailable",
        description: "Update checks run from the desktop app.",
      });
      return;
    }
    showNativeStatus(await rpc.request.checkForUpdates());
  }, []);

  const executeCommand = useCallback(
    async (command: DesktopCommand) => {
      switch (command.name) {
        case "about":
          void loadMetadata();
          setAboutOpen(true);
          break;
        case "check-updates":
          await checkForUpdates();
          break;
        case "clear-data":
          dispatchTraceCommand({ type: "open-clear-data" });
          break;
        case "command-palette":
          setPaletteOpen(true);
          break;
        case "copy-diagnostics":
          await copyDiagnostics();
          break;
        case "copy-ingest-url":
          await copyIngestUrl();
          break;
        case "import-data":
          dispatchTraceCommand({ type: "open-import" });
          break;
        case "navigate-analysis":
        case "navigate-sessions":
        case "navigate-settings":
        case "navigate-traces":
        case "preferences":
          navigateTo(command.name);
          break;
        case "open-app-data": {
          const rpc = await getDesktopRpc();
          const ok = rpc ? (await rpc.request.openAppDataFolder()).ok : false;
          toast[ok ? "success" : "info"]({
            title: ok ? "Opened app data folder" : "App data folder",
            description: appMetadata.appDataDir,
          });
          break;
        }
        case "open-docs": {
          const rpc = await getDesktopRpc();
          if (rpc) {
            await rpc.request.openExternal({ url: APP_DOCS_URL });
          } else {
            window.open(APP_DOCS_URL, "_blank", "noopener,noreferrer");
          }
          break;
        }
        case "refresh":
          if (path === "/traces" || path === "/sessions") {
            dispatchTraceCommand({ type: "refresh" });
          } else {
            void queryClient.invalidateQueries();
            toast.info({ title: "Refreshing workspace" });
          }
          break;
        case "reveal-database": {
          const rpc = await getDesktopRpc();
          const ok = rpc ? (await rpc.request.revealDatabaseFile()).ok : false;
          toast[ok ? "success" : "info"]({
            title: ok ? "Revealed database file" : "Database file",
            description: appMetadata.dbPath,
          });
          break;
        }
        case "toggle-follow-latest": {
          if (path === "/traces") {
            dispatchTraceCommand({ type: "toggle-follow-latest" });
          } else {
            void navigate({
              to: "/traces",
              search: {
                followLatest: 1,
                traceId:
                  typeof search.traceId === "string" ? search.traceId : undefined,
              },
            } as unknown as NavigateOptions);
          }
          break;
        }
        default:
          break;
      }
    },
    [
      appMetadata,
      checkForUpdates,
      copyDiagnostics,
      copyIngestUrl,
      dispatchTraceCommand,
      loadMetadata,
      navigate,
      navigateTo,
      path,
      queryClient,
      search.traceId,
    ],
  );

  useEffect(() => {
    const onCommand = (event: WindowEventMap[typeof DESKTOP_COMMAND_EVENT]) => {
      void executeCommand(event.detail);
    };
    const onNativeStatus = (
      event: WindowEventMap[typeof DESKTOP_NATIVE_STATUS_EVENT],
    ) => showNativeStatus(event.detail);
    window.addEventListener(DESKTOP_COMMAND_EVENT, onCommand);
    window.addEventListener(DESKTOP_NATIVE_STATUS_EVENT, onNativeStatus);
    return () => {
      window.removeEventListener(DESKTOP_COMMAND_EVENT, onCommand);
      window.removeEventListener(DESKTOP_NATIVE_STATUS_EVENT, onNativeStatus);
    };
  }, [executeCommand]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const modifier = event.metaKey || event.ctrlKey;
      if (!modifier || event.altKey) return;

      const key = event.key.toLowerCase();
      const isTextEntry = isTextInputTarget(event.target);
      const command = desktopCommandForShortcut(key, event.shiftKey);
      if (!command) return;
      if (isTextEntry && command !== "command-palette") return;

      event.preventDefault();
      void executeCommand({ name: command, source: "keyboard" });
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [executeCommand]);

  trpc.live.workspace.useSubscription(undefined, {
    onData(eventEnvelope) {
      const event = eventEnvelope.data;
      if (notifiedLiveEvents.current.has(event.id)) return;
      notifiedLiveEvents.current.add(event.id);
      if (notifiedLiveEvents.current.size > 200) {
        const first = notifiedLiveEvents.current.values().next().value;
        if (typeof first === "number") {
          notifiedLiveEvents.current.delete(first);
        }
      }

      const notification = notificationForLiveEvent(event);
      if (!notification || document.hasFocus()) return;
      void getDesktopRpc().then((rpc) => {
        if (!rpc) return;
        return rpc.request.showNotification(notification);
      });
    },
  });

  const paletteItems = filterCommandPaletteItems(paletteQuery);

  return (
    <>
      <CommandPaletteDialog
        items={paletteItems}
        onExecute={(command) => {
          setPaletteOpen(false);
          setPaletteQuery("");
          void executeCommand({ name: command, source: "palette" });
        }}
        onOpenChange={setPaletteOpen}
        open={paletteOpen}
        query={paletteQuery}
        setQuery={setPaletteQuery}
      />
      <AboutHaloDialog
        metadata={appMetadata}
        onCheckUpdates={() => void checkForUpdates()}
        onCopyDiagnostics={() => void copyDiagnostics()}
        onOpenAppData={() => void executeCommand({ name: "open-app-data" })}
        onOpenDocs={() => void executeCommand({ name: "open-docs" })}
        onOpenChange={setAboutOpen}
        onRevealDatabase={() => void executeCommand({ name: "reveal-database" })}
        open={aboutOpen}
      />
    </>
  );
}

function CommandPaletteDialog({
  items,
  onExecute,
  onOpenChange,
  open,
  query,
  setQuery,
}: {
  items: CommandPaletteItem[];
  onExecute: (command: DesktopCommandName) => void;
  onOpenChange: (open: boolean) => void;
  open: boolean;
  query: string;
  setQuery: (query: string) => void;
}) {
  const groupedItems = useMemo(() => groupCommandItems(items), [items]);

  return (
    <DialogRoot onOpenChange={onOpenChange} open={open}>
      <DialogContent className="sm:max-w-2xl overflow-hidden p-0">
        <div className="flex items-center gap-3 border-b px-4 py-3">
          <Search className="h-4 w-4 text-muted-foreground" />
          <Input
            autoFocus
            className="h-10 border-0 bg-transparent px-0 text-sm shadow-none focus-visible:ring-0"
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search commands..."
            value={query}
          />
        </div>
        <div className="max-h-[480px] overflow-y-auto p-2">
          {groupedItems.length === 0 ? (
            <div className="px-3 py-10 text-center text-sm text-muted-foreground">
              No commands found.
            </div>
          ) : (
            groupedItems.map(([group, groupItems]) => (
              <div className="py-1" key={group}>
                <div className="px-3 py-2 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                  {group}
                </div>
                {groupItems.map((item) => (
                  <button
                    className={cn(
                      "flex w-full items-center gap-3 rounded-md px-3 py-2 text-left text-sm transition-colors",
                      "hover:bg-muted focus:bg-muted focus:outline-none",
                    )}
                    key={item.command}
                    onClick={() => onExecute(item.command)}
                    type="button"
                  >
                    <Command className="h-4 w-4 text-muted-foreground" />
                    <span className="min-w-0 flex-1">
                      <span className="block font-medium text-foreground">
                        {item.label}
                      </span>
                      <span className="block truncate text-xs text-muted-foreground">
                        {item.description}
                      </span>
                    </span>
                    {item.shortcut ? (
                      <span className="rounded border bg-background px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground">
                        {item.shortcut}
                      </span>
                    ) : null}
                  </button>
                ))}
              </div>
            ))
          )}
        </div>
      </DialogContent>
    </DialogRoot>
  );
}

function AboutHaloDialog({
  metadata,
  onCheckUpdates,
  onCopyDiagnostics,
  onOpenAppData,
  onOpenChange,
  onOpenDocs,
  onRevealDatabase,
  open,
}: {
  metadata: DesktopAppMetadata;
  onCheckUpdates: () => void;
  onCopyDiagnostics: () => void;
  onOpenAppData: () => void;
  onOpenChange: (open: boolean) => void;
  onOpenDocs: () => void;
  onRevealDatabase: () => void;
  open: boolean;
}) {
  return (
    <DialogRoot onOpenChange={onOpenChange} open={open}>
      <DialogContent className="sm:max-w-2xl overflow-hidden p-0">
        <DialogHeader className="p-6">
          <div className="flex items-start gap-4">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg border bg-muted">
              <InferenceIcon height={22} width={32} />
            </div>
            <div className="min-w-0">
              <DialogTitle className="text-2xl">{APP_NAME}</DialogTitle>
              <DialogDescription>
                Local trace monitoring and analysis for AI agent development.
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>
        <div className="space-y-5 p-6">
          <div className="grid grid-cols-2 gap-3">
            <AboutMetric label="Version" value={metadata.version} />
            <AboutMetric label="Channel" value={metadata.channel} />
            <AboutMetric label="Bundle ID" value={metadata.bundleId} />
            <AboutMetric label="Release URL" value={metadata.releaseUrl} />
          </div>
          <Separator />
          <div className="space-y-2">
            <AboutPathRow icon={Clipboard} label="Ingest URL" value={metadata.ingestUrl} />
            <AboutPathRow icon={Sparkles} label="Live socket" value={metadata.liveUrl} />
            <AboutPathRow icon={FolderOpen} label="App data" value={metadata.appDataDir} />
            <AboutPathRow icon={Database} label="Database" value={metadata.dbPath} />
          </div>
          <div className="flex flex-wrap justify-end gap-2">
            <Button onClick={onOpenDocs} size="sm" variant="outline">
              <ExternalLink className="mr-2 h-4 w-4" />
              Docs
            </Button>
            <Button onClick={onOpenAppData} size="sm" variant="outline">
              <FolderOpen className="mr-2 h-4 w-4" />
              App Data
            </Button>
            <Button onClick={onRevealDatabase} size="sm" variant="outline">
              <Database className="mr-2 h-4 w-4" />
              Database
            </Button>
            <Button onClick={onCopyDiagnostics} size="sm" variant="outline">
              <Clipboard className="mr-2 h-4 w-4" />
              Copy Diagnostics
            </Button>
            <Button onClick={onCheckUpdates} size="sm">
              <RefreshCcw className="mr-2 h-4 w-4" />
              Check Updates
            </Button>
          </div>
        </div>
      </DialogContent>
    </DialogRoot>
  );
}

function AboutMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border bg-muted/20 p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 truncate font-mono text-sm text-foreground">{value}</div>
    </div>
  );
}

function AboutPathRow({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Info;
  label: string;
  value: string;
}) {
  return (
    <div className="grid grid-cols-[120px_1fr] items-center gap-3 rounded-md border bg-muted/10 px-3 py-2">
      <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
        <Icon className="h-3.5 w-3.5" />
        {label}
      </div>
      <div className="truncate font-mono text-xs text-foreground">{value}</div>
    </div>
  );
}

function groupCommandItems(items: CommandPaletteItem[]) {
  const groups = new Map<CommandPaletteItem["group"], CommandPaletteItem[]>();
  for (const item of items) {
    const groupItems = groups.get(item.group) ?? [];
    groupItems.push(item);
    groups.set(item.group, groupItems);
  }
  return Array.from(groups.entries());
}

function isTextInputTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  return Boolean(target.closest("input, textarea, select, [contenteditable='true']"));
}

function showNativeStatus(status: DesktopNativeStatus) {
  if (status.status === "update") {
    toast[status.updateAvailable ? "success" : "info"]({
      title: status.title,
      description: status.message,
    });
    return;
  }

  toast[status.status]({
    title: status.title,
    description: status.message,
  });
}

function diagnosticsText(metadata: DesktopAppMetadata) {
  return JSON.stringify(
    {
      app: APP_NAME,
      generatedAt: new Date().toISOString(),
      location: typeof window === "undefined" ? null : window.location.href,
      metadata,
      userAgent: typeof navigator === "undefined" ? null : navigator.userAgent,
    },
    null,
    2,
  );
}

function notificationForLiveEvent(event: { id: number; payload: unknown }) {
  const payload = event.payload;
  if (!payload || typeof payload !== "object") return null;
  const type = (payload as { type?: unknown }).type;

  if (type === "halo.run.completed") {
    const run = (payload as { run?: { title?: string | null } }).run;
    return { title: "HALO run completed", body: run?.title ?? "Analysis run" };
  }

  if (type === "halo.run.failed") {
    const run = (payload as { run?: { title?: string | null } }).run;
    return { title: "HALO run failed", body: run?.title ?? "Analysis run" };
  }

  if (type === "import.job.updated") {
    const job = (payload as {
      job?: { connectionName?: string | null; status?: string | null };
    }).job;
    if (job?.status !== "completed" && job?.status !== "failed") return null;
    return {
      title: job.status === "completed"
        ? "Langfuse import completed"
        : "Langfuse import failed",
      body: job.connectionName ?? "Langfuse",
    };
  }

  return null;
}

export function commandPaletteLabelForTest(command: DesktopCommandName) {
  return commandLabel(command);
}
