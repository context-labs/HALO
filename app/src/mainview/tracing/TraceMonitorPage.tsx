import {
  useDeferredValue,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { Link } from "@tanstack/react-router";
import {
  Activity,
  AlertCircle,
  Boxes,
  Braces,
  CalendarClock,
  CheckCircle2,
  ChevronRight,
  CircleDollarSign,
  Clipboard,
  Code2,
  Copy,
  DownloadCloud,
  Filter,
  Layers3,
  ListTree,
  Loader2,
  MessageSquare,
  RefreshCcw,
  Search,
  Server,
  Terminal,
  Trash2,
  XCircle,
  Zap,
} from "lucide-react";

import {
  Badge,
  Button,
  Dialog,
  Input,
  Separator,
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  buttonVariants,
  cn,
  toast,
} from "~/lib/ui";
import { trpc } from "~/trpc";
import { WorkspaceNav } from "~/workspace/WorkspaceNav";
import {
  TRACE_PAGE_COMMAND_EVENT,
  showDesktopRowContextMenu,
} from "~/desktop/desktopBridge";
import { LangfuseImportDialog } from "./LangfuseImportDialog";
import type {
  FacetOption,
  SessionSortKey,
  SessionSummary,
  Span,
  SpanNode,
  Trace,
  TraceSortKey,
} from "../../server/telemetry/types";
import {
  nextFollowLatestTraceId,
  traceIdsForLiveEvent,
} from "./followLatest";
import {
  buildClientSpanTree,
  buildSessionSpanTree,
  findFirstInspectableSpan,
  flattenSpanTree,
  isSessionTraceGroupSpan,
  isSyntheticSpan,
} from "./spanTree";

const DEFAULT_INGEST_URL = "http://127.0.0.1:8799/v1/traces";
const CATALYST_ENV_LINE =
  "CATALYST_OTLP_ENDPOINT=http://127.0.0.1:8799/v1/traces";
const EMPTY_SPANS: Span[] = [];

type DateRange = "1h" | "24h" | "7d" | "all";
type DetailTab = "tree" | "timeline" | "span" | "raw";
type LiveStatus = "connecting" | "live" | "reconnecting" | "offline";
type StatusFilter = "all" | "ok" | "error";
type ScopeFilter = "all" | "root" | "entrypoint";
type SourceFilter = "all" | "local" | "langfuse";

export function TraceMonitorPage({
  followLatest,
  onFollowLatestChange,
  onSelectLatestTrace,
  onSelectTrace,
  selectedTraceId,
}: {
  followLatest: boolean;
  onFollowLatestChange: (enabled: boolean) => void;
  onSelectLatestTrace: (traceId: string) => void;
  onSelectTrace: (traceId: string | null) => void;
  selectedTraceId?: string;
}) {
  const [searchText, setSearchText] = useState("");
  const [dateRange, setDateRange] = useState<DateRange>("24h");
  const [status, setStatus] = useState<StatusFilter>("all");
  const [scope, setScope] = useState<ScopeFilter>("all");
  const [serviceName, setServiceName] = useState("all");
  const [agentName, setAgentName] = useState("all");
  const [modelName, setModelName] = useState("all");
  const [source, setSource] = useState<SourceFilter>("all");
  const [sortBy, setSortBy] = useState<TraceSortKey>("start_time");
  const [clearDialogOpen, setClearDialogOpen] = useState(false);
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [liveStatus, setLiveStatus] = useState<LiveStatus>("connecting");
  const [recentTraceIds, setRecentTraceIds] = useState<Set<string>>(() => new Set());
  const recentTraceTimers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const workspaceInvalidateTimer = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );
  const followLatestRef = useRef(followLatest);
  const selectedTraceIdRef = useRef<string | undefined>(selectedTraceId);
  const utils = trpc.useUtils();

  const clearDataMutation = trpc.telemetry.clearData.useMutation({
    onError(error) {
      toast.error({
        title: "Could not clear telemetry data",
        description: error.message,
      });
    },
    async onSuccess(result) {
      setClearDialogOpen(false);
      setRecentTraceIds(new Set());
      setSearchText("");
      onFollowLatestChange(false);
      onSelectTrace(null);
      await Promise.all([
        utils.telemetry.info.invalidate(),
        utils.traces.facets.invalidate(),
        utils.traces.list.invalidate(),
        utils.traces.search.invalidate(),
        utils.traces.get.invalidate(),
        utils.traces.getSpans.invalidate(),
        utils.spans.list.invalidate(),
        utils.spans.facets.invalidate(),
        utils.sessions.facets.invalidate(),
        utils.sessions.list.invalidate(),
        utils.sessions.search.invalidate(),
        utils.sessions.get.invalidate(),
        utils.sessions.getSpans.invalidate(),
        utils.sessions.getTraces.invalidate(),
      ]);
      toast.success({
        title: "Telemetry data cleared",
        description: `${result.traceCount} traces and ${result.spanCount} spans removed.`,
      });
    },
  });

  useEffect(() => {
    followLatestRef.current = followLatest;
    selectedTraceIdRef.current = selectedTraceId;
  }, [followLatest, selectedTraceId]);

  const markRecentTraceIds = useCallback((traceIds: string[]) => {
    if (traceIds.length === 0) return;
    setRecentTraceIds((current) => {
      const next = new Set(current);
      traceIds.forEach((traceId) => next.add(traceId));
      return next;
    });
    for (const traceId of traceIds) {
      const existing = recentTraceTimers.current.get(traceId);
      if (existing) clearTimeout(existing);
      const timer = setTimeout(() => {
        recentTraceTimers.current.delete(traceId);
        setRecentTraceIds((current) => {
          const next = new Set(current);
          next.delete(traceId);
          return next;
        });
      }, 1_800);
      recentTraceTimers.current.set(traceId, timer);
    }
  }, []);

  const invalidateWorkspace = useCallback(() => {
    if (workspaceInvalidateTimer.current) return;
    workspaceInvalidateTimer.current = setTimeout(() => {
      workspaceInvalidateTimer.current = null;
      void utils.telemetry.info.invalidate();
      void utils.traces.facets.invalidate();
      void utils.traces.list.invalidate();
      void utils.traces.search.invalidate();
      void utils.sessions.facets.invalidate();
      void utils.sessions.list.invalidate();
      void utils.sessions.search.invalidate();
    }, 80);
  }, [utils]);

  trpc.live.workspace.useSubscription(undefined, {
    onComplete() {
      setLiveStatus("offline");
    },
    onData(eventEnvelope) {
      const event = eventEnvelope.data;
      setLiveStatus("live");
      markRecentTraceIds(traceIdsForLiveEvent(event));
      const latestTraceId = nextFollowLatestTraceId({
        currentTraceId: selectedTraceIdRef.current,
        event,
        followLatest: followLatestRef.current,
      });
      if (latestTraceId) {
        selectedTraceIdRef.current = latestTraceId;
        onSelectLatestTrace(latestTraceId);
      }
      if (event.payload.type === "trace.upserted") {
        utils.traces.get.setData(
          { traceId: event.payload.trace.traceId },
          event.payload.trace,
        );
      }
      invalidateWorkspace();
    },
    onError() {
      setLiveStatus("reconnecting");
    },
    onStarted() {
      setLiveStatus("live");
    },
  });

  useEffect(
    () => () => {
      if (workspaceInvalidateTimer.current) {
        clearTimeout(workspaceInvalidateTimer.current);
      }
      for (const timer of recentTraceTimers.current.values()) {
        clearTimeout(timer);
      }
    },
    [],
  );

  const activeSearch = useDeferredValue(searchText.trim());
  const filters = useMemo(() => {
    const startDate = startDateForRange(dateRange);
    return {
      agents: agentName === "all" ? undefined : [agentName],
      llmModelNames: modelName === "all" ? undefined : [modelName],
      scope: scope === "all" ? undefined : scope,
      serviceNames: serviceName === "all" ? undefined : [serviceName],
      sources: source === "all" ? undefined : [source],
      startDate,
      status: status === "all" ? undefined : status,
    };
  }, [agentName, dateRange, modelName, scope, serviceName, source, status]);

  const infoQuery = trpc.telemetry.info.useQuery();
  const facetsQuery = trpc.traces.facets.useQuery(
    {
      facetIds: [
        "agent_name",
        "llm_model_name",
        "observation_kind",
        "service_name",
        "source",
        "status",
      ],
    },
    {},
  );
  const listQuery = trpc.traces.list.useQuery(
    {
      filters,
      limit: 75,
      sortBy,
      sortOrder: "desc",
    },
    {
      enabled: activeSearch.length === 0,
    },
  );
  const searchQuery = trpc.traces.search.useQuery(
    {
      filters,
      limit: 75,
      query: activeSearch,
    },
    {
      enabled: activeSearch.length > 0,
    },
  );

  const traces = useMemo(
    () =>
      activeSearch
        ? (searchQuery.data?.results.map((result) => result.trace) ?? [])
        : (listQuery.data?.traces ?? []),
    [activeSearch, listQuery.data?.traces, searchQuery.data?.results],
  );
  const totalCount = activeSearch
    ? (searchQuery.data?.totalCount ?? 0)
    : (listQuery.data?.totalCount ?? 0);
  const isLoading =
    infoQuery.isLoading ||
    (activeSearch ? searchQuery.isLoading : listQuery.isLoading);
  const isRefreshing =
    infoQuery.isFetching ||
    facetsQuery.isFetching ||
    listQuery.isFetching ||
    searchQuery.isFetching;

  const visibleMetrics = useMemo(() => summarizeVisibleTraces(traces), [traces]);
  const ingestUrl = infoQuery.data?.ingestUrl ?? DEFAULT_INGEST_URL;

  const copyIngestUrl = useCallback(async () => {
    await navigator.clipboard.writeText(ingestUrl);
    toast.success({
      title: "Ingest URL copied",
      description: "Paste it into your local agent telemetry config.",
    });
  }, [ingestUrl]);

  const refresh = useCallback(() => {
    void infoQuery.refetch();
    void facetsQuery.refetch();
    void (activeSearch ? searchQuery.refetch() : listQuery.refetch());
  }, [activeSearch, facetsQuery, infoQuery, listQuery, searchQuery]);

  useEffect(() => {
    const onPageCommand = (
      event: WindowEventMap[typeof TRACE_PAGE_COMMAND_EVENT],
    ) => {
      switch (event.detail.type) {
        case "copy-ingest-url":
          void copyIngestUrl();
          break;
        case "open-clear-data":
          setClearDialogOpen(true);
          break;
        case "open-import":
          setImportDialogOpen(true);
          break;
        case "refresh":
          refresh();
          break;
        case "toggle-follow-latest":
          onFollowLatestChange(!followLatestRef.current);
          break;
        default:
          break;
      }
    };

    window.addEventListener(TRACE_PAGE_COMMAND_EVENT, onPageCommand);
    return () =>
      window.removeEventListener(TRACE_PAGE_COMMAND_EVENT, onPageCommand);
  }, [copyIngestUrl, onFollowLatestChange, refresh]);

  return (
    <main className="min-h-screen bg-background text-foreground">
      <TraceTitleBar
        icon="traces"
        health={infoQuery.data?.lastBatch?.status ?? "waiting"}
        isRefreshing={isRefreshing}
        liveStatus={liveStatus}
        liveUrl={infoQuery.data?.liveUrl ?? "ws://127.0.0.1:8800"}
        title="Trace Monitor"
        followLatest={followLatest}
        onFollowLatestChange={onFollowLatestChange}
        onCopy={copyIngestUrl}
        onClearData={() => setClearDialogOpen(true)}
        onImport={() => setImportDialogOpen(true)}
        onRefresh={refresh}
      />

      <div className="grid min-h-[calc(100vh-3.5rem)] grid-cols-[112px_300px_minmax(0,1fr)] pt-14">
        <WorkspaceNav active="traces" />
        <FilterSidebar
          agentName={agentName}
          dateRange={dateRange}
          description="Narrow local traces by runtime, model, and time."
          facets={facetsQuery.data?.categorical ?? {}}
          modelName={modelName}
          onAgentNameChange={setAgentName}
          onDateRangeChange={setDateRange}
          onModelNameChange={setModelName}
          onReset={() => {
            setDateRange("24h");
            setStatus("all");
            setScope("all");
            setServiceName("all");
            setAgentName("all");
            setModelName("all");
            setSource("all");
          }}
          onScopeChange={setScope}
          onServiceNameChange={setServiceName}
          onStatusChange={setStatus}
          scope={scope}
          serviceName={serviceName}
          source={source}
          status={status}
          onSourceChange={setSource}
        />

        <section className="min-w-0 overflow-hidden">
          <div className="flex h-full min-h-[calc(100vh-3.5rem)] flex-col">
            <MetricsStrip
              errorCount={visibleMetrics.errorCount}
              isLoading={isLoading}
              llmSpanCount={visibleMetrics.llmSpanCount}
              spanCount={visibleMetrics.spanCount}
              totalCost={visibleMetrics.totalCost}
              totalTokens={visibleMetrics.totalTokens}
              traceCount={totalCount}
            />

            <div className="border-b border-subtle px-6 py-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="min-w-0">
                  <h1 className="text-2xl tracking-normal">
                    Local agent traces
                  </h1>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {infoQuery.data?.lastBatch
                      ? `Last ingest ${relativeTime(infoQuery.data.lastBatch.receivedAt)} with ${infoQuery.data.lastBatch.acceptedSpanCount} spans`
                      : "Waiting for your first OTLP trace batch"}
                  </p>
                </div>
                <div className="flex min-w-0 items-center gap-2">
                  <Input
                    aria-label="Search traces"
                    className="h-9"
                    containerClassname="w-72"
                    icon={<Search className="h-4 w-4 text-muted-foreground" />}
                    onChange={(event) => setSearchText(event.currentTarget.value)}
                    placeholder="Search trace IDs, models, inputs..."
                    value={searchText}
                  />
                  <select
                    aria-label="Sort traces"
                    className="h-9 rounded-md border border-subtle bg-background px-3 text-sm"
                    onChange={(event) =>
                      setSortBy(event.currentTarget.value as TraceSortKey)
                    }
                    value={sortBy}
                  >
                    <option value="start_time">Newest</option>
                    <option value="duration">Duration</option>
                    <option value="span_count">Span count</option>
                    <option value="llm_span_count">LLM spans</option>
                    <option value="total_tokens">Tokens</option>
                    <option value="total_cost">Cost</option>
                  </select>
                </div>
              </div>
            </div>

            {infoQuery.data?.spanCount === 0 ? (
              <EmptyTraceState ingestUrl={ingestUrl} onCopy={copyIngestUrl} />
            ) : (
              <TraceList
                activeTraceId={selectedTraceId}
                isLoading={isLoading}
                onSelectTrace={onSelectTrace}
                recentTraceIds={recentTraceIds}
                totalCount={totalCount}
                traces={traces}
              />
            )}
          </div>
        </section>
      </div>

      <TelemetryDetailSheet
        followLatest={followLatest}
        mode="trace"
        onOpenChange={(open) => {
          if (!open) onSelectTrace(null);
        }}
        open={followLatest || Boolean(selectedTraceId)}
        traceId={selectedTraceId}
      />
      <LangfuseImportDialog
        onImported={refresh}
        onOpenChange={setImportDialogOpen}
        open={importDialogOpen}
      />
      <Dialog
        cancelTitle="Cancel"
        className="sm:!max-w-[520px] md:!w-[520px]"
        confirmButtonVariant="destructive"
        confirmTitle="Clear data"
        dialogDescription="This removes local traces, spans, search rows, ingest batches, and live telemetry history. Saved Langfuse connections stay intact."
        dialogTitle="Clear local telemetry data?"
        disabled={clearDataMutation.isPending}
        loading={clearDataMutation.isPending}
        onConfirm={() => clearDataMutation.mutate()}
        onOpenChange={setClearDialogOpen}
        open={clearDialogOpen}
      >
        <div className="rounded-md border border-destructive-border bg-destructive/5 p-4 text-sm">
          <div className="flex items-start gap-3">
            <Trash2 className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
            <div className="space-y-1">
              <p className="font-medium text-foreground">
                This cannot be undone.
              </p>
              <p className="text-muted-foreground">
                Current local database contains {infoQuery.data?.traceCount ?? 0}{" "}
                traces and {infoQuery.data?.spanCount ?? 0} spans.
              </p>
            </div>
          </div>
        </div>
      </Dialog>
    </main>
  );
}

export function SessionsPage({
  onSelectSession,
  selectedSessionId,
}: {
  onSelectSession: (sessionId: string | null) => void;
  selectedSessionId?: string;
}) {
  const [searchText, setSearchText] = useState("");
  const [dateRange, setDateRange] = useState<DateRange>("24h");
  const [status, setStatus] = useState<StatusFilter>("all");
  const [scope, setScope] = useState<ScopeFilter>("all");
  const [serviceName, setServiceName] = useState("all");
  const [agentName, setAgentName] = useState("all");
  const [modelName, setModelName] = useState("all");
  const [source, setSource] = useState<SourceFilter>("all");
  const [sortBy, setSortBy] = useState<SessionSortKey>("last_activity");
  const [clearDialogOpen, setClearDialogOpen] = useState(false);
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [liveStatus, setLiveStatus] = useState<LiveStatus>("connecting");
  const [recentSessionIds, setRecentSessionIds] = useState<Set<string>>(
    () => new Set(),
  );
  const recentSessionTimers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const workspaceInvalidateTimer = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );
  const utils = trpc.useUtils();

  const clearDataMutation = trpc.telemetry.clearData.useMutation({
    onError(error) {
      toast.error({
        title: "Could not clear telemetry data",
        description: error.message,
      });
    },
    async onSuccess(result) {
      setClearDialogOpen(false);
      setRecentSessionIds(new Set());
      setSearchText("");
      onSelectSession(null);
      await Promise.all([
        utils.telemetry.info.invalidate(),
        utils.sessions.facets.invalidate(),
        utils.sessions.list.invalidate(),
        utils.sessions.search.invalidate(),
        utils.sessions.get.invalidate(),
        utils.sessions.getSpans.invalidate(),
        utils.sessions.getTraces.invalidate(),
        utils.traces.facets.invalidate(),
        utils.traces.list.invalidate(),
      ]);
      toast.success({
        title: "Telemetry data cleared",
        description: `${result.traceCount} traces and ${result.spanCount} spans removed.`,
      });
    },
  });

  const markRecentSessionId = useCallback((sessionId: string | null | undefined) => {
    if (!sessionId) return;
    setRecentSessionIds((current) => {
      const next = new Set(current);
      next.add(sessionId);
      return next;
    });
    const existing = recentSessionTimers.current.get(sessionId);
    if (existing) clearTimeout(existing);
    const timer = setTimeout(() => {
      recentSessionTimers.current.delete(sessionId);
      setRecentSessionIds((current) => {
        const next = new Set(current);
        next.delete(sessionId);
        return next;
      });
    }, 1_800);
    recentSessionTimers.current.set(sessionId, timer);
  }, []);

  const invalidateWorkspace = useCallback(() => {
    if (workspaceInvalidateTimer.current) return;
    workspaceInvalidateTimer.current = setTimeout(() => {
      workspaceInvalidateTimer.current = null;
      void utils.telemetry.info.invalidate();
      void utils.sessions.facets.invalidate();
      void utils.sessions.list.invalidate();
      void utils.sessions.search.invalidate();
    }, 80);
  }, [utils]);

  trpc.live.workspace.useSubscription(undefined, {
    onComplete() {
      setLiveStatus("offline");
    },
    onData(eventEnvelope) {
      const event = eventEnvelope.data;
      setLiveStatus("live");
      if (event.payload.type === "trace.upserted") {
        markRecentSessionId(event.payload.trace.sessionId);
      }
      if (event.payload.type === "span.upserted") {
        markRecentSessionId(event.payload.span.sessionId);
      }
      invalidateWorkspace();
    },
    onError() {
      setLiveStatus("reconnecting");
    },
    onStarted() {
      setLiveStatus("live");
    },
  });

  useEffect(
    () => () => {
      if (workspaceInvalidateTimer.current) {
        clearTimeout(workspaceInvalidateTimer.current);
      }
      for (const timer of recentSessionTimers.current.values()) {
        clearTimeout(timer);
      }
    },
    [],
  );

  const activeSearch = useDeferredValue(searchText.trim());
  const filters = useMemo(() => {
    const startDate = startDateForRange(dateRange);
    return {
      agents: agentName === "all" ? undefined : [agentName],
      llmModelNames: modelName === "all" ? undefined : [modelName],
      scope: scope === "all" ? undefined : scope,
      serviceNames: serviceName === "all" ? undefined : [serviceName],
      sources: source === "all" ? undefined : [source],
      startDate,
      status: status === "all" ? undefined : status,
    };
  }, [agentName, dateRange, modelName, scope, serviceName, source, status]);

  const infoQuery = trpc.telemetry.info.useQuery();
  const facetsQuery = trpc.sessions.facets.useQuery({
    facetIds: [
      "agent_name",
      "llm_model_name",
      "observation_kind",
      "service_name",
      "source",
      "status",
    ],
  });
  const listQuery = trpc.sessions.list.useQuery(
    {
      filters,
      limit: 75,
      sortBy,
      sortOrder: "desc",
    },
    { enabled: activeSearch.length === 0 },
  );
  const searchQuery = trpc.sessions.search.useQuery(
    {
      filters,
      limit: 75,
      query: activeSearch,
    },
    { enabled: activeSearch.length > 0 },
  );
  const sessions = useMemo(
    () =>
      activeSearch
        ? (searchQuery.data?.results.map((result) => result.session) ?? [])
        : (listQuery.data?.sessions ?? []),
    [activeSearch, listQuery.data?.sessions, searchQuery.data?.results],
  );
  const totalCount = activeSearch
    ? (searchQuery.data?.totalCount ?? 0)
    : (listQuery.data?.totalCount ?? 0);
  const isLoading =
    infoQuery.isLoading ||
    (activeSearch ? searchQuery.isLoading : listQuery.isLoading);
  const isRefreshing =
    infoQuery.isFetching ||
    facetsQuery.isFetching ||
    listQuery.isFetching ||
    searchQuery.isFetching;
  const visibleMetrics = useMemo(() => summarizeVisibleSessions(sessions), [sessions]);
  const ingestUrl = infoQuery.data?.ingestUrl ?? DEFAULT_INGEST_URL;

  const copyIngestUrl = useCallback(async () => {
    await navigator.clipboard.writeText(ingestUrl);
    toast.success({
      title: "Ingest URL copied",
      description: "Paste it into your local agent telemetry config.",
    });
  }, [ingestUrl]);

  const refresh = useCallback(() => {
    void infoQuery.refetch();
    void facetsQuery.refetch();
    void (activeSearch ? searchQuery.refetch() : listQuery.refetch());
  }, [activeSearch, facetsQuery, infoQuery, listQuery, searchQuery]);

  useEffect(() => {
    const onPageCommand = (
      event: WindowEventMap[typeof TRACE_PAGE_COMMAND_EVENT],
    ) => {
      switch (event.detail.type) {
        case "copy-ingest-url":
          void copyIngestUrl();
          break;
        case "open-clear-data":
          setClearDialogOpen(true);
          break;
        case "open-import":
          setImportDialogOpen(true);
          break;
        case "refresh":
          refresh();
          break;
        default:
          break;
      }
    };

    window.addEventListener(TRACE_PAGE_COMMAND_EVENT, onPageCommand);
    return () =>
      window.removeEventListener(TRACE_PAGE_COMMAND_EVENT, onPageCommand);
  }, [copyIngestUrl, refresh]);

  return (
    <main className="min-h-screen bg-background text-foreground">
      <TraceTitleBar
        health={infoQuery.data?.lastBatch?.status ?? "waiting"}
        icon="sessions"
        isRefreshing={isRefreshing}
        liveStatus={liveStatus}
        liveUrl={infoQuery.data?.liveUrl ?? "ws://127.0.0.1:8800"}
        onClearData={() => setClearDialogOpen(true)}
        onCopy={copyIngestUrl}
        onImport={() => setImportDialogOpen(true)}
        onRefresh={refresh}
        title="Sessions"
      />

      <div className="grid min-h-[calc(100vh-3.5rem)] grid-cols-[112px_300px_minmax(0,1fr)] pt-14">
        <WorkspaceNav active="sessions" />
        <FilterSidebar
          agentName={agentName}
          dateRange={dateRange}
          description="Narrow local sessions by runtime, model, and time."
          facets={facetsQuery.data?.categorical ?? {}}
          modelName={modelName}
          onAgentNameChange={setAgentName}
          onDateRangeChange={setDateRange}
          onModelNameChange={setModelName}
          onReset={() => {
            setDateRange("24h");
            setStatus("all");
            setScope("all");
            setServiceName("all");
            setAgentName("all");
            setModelName("all");
            setSource("all");
          }}
          onScopeChange={setScope}
          onServiceNameChange={setServiceName}
          onSourceChange={setSource}
          onStatusChange={setStatus}
          scope={scope}
          serviceName={serviceName}
          source={source}
          status={status}
        />

        <section className="min-w-0 overflow-hidden">
          <div className="flex h-full min-h-[calc(100vh-3.5rem)] flex-col">
            <SessionMetricsStrip
              errorCount={visibleMetrics.errorCount}
              isLoading={isLoading}
              llmSpanCount={visibleMetrics.llmSpanCount}
              sessionCount={totalCount}
              spanCount={visibleMetrics.spanCount}
              totalCost={visibleMetrics.totalCost}
              totalTokens={visibleMetrics.totalTokens}
              traceCount={visibleMetrics.traceCount}
            />

            <div className="border-b border-subtle px-6 py-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="min-w-0">
                  <h1 className="text-2xl tracking-normal">
                    Local agent sessions
                  </h1>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Conversations grouped by session ID across all turns.
                  </p>
                </div>
                <div className="flex min-w-0 items-center gap-2">
                  <Input
                    aria-label="Search sessions"
                    className="h-9"
                    containerClassname="w-72"
                    icon={<Search className="h-4 w-4 text-muted-foreground" />}
                    onChange={(event) => setSearchText(event.currentTarget.value)}
                    placeholder="Search sessions, services, models..."
                    value={searchText}
                  />
                  <select
                    aria-label="Sort sessions"
                    className="h-9 rounded-md border border-subtle bg-background px-3 text-sm"
                    onChange={(event) =>
                      setSortBy(event.currentTarget.value as SessionSortKey)
                    }
                    value={sortBy}
                  >
                    <option value="last_activity">Latest activity</option>
                    <option value="start_time">First activity</option>
                    <option value="duration">Duration</option>
                    <option value="trace_count">Turns</option>
                    <option value="span_count">Spans</option>
                    <option value="llm_span_count">LLM spans</option>
                    <option value="total_tokens">Tokens</option>
                    <option value="total_cost">Cost</option>
                  </select>
                </div>
              </div>
            </div>

            <SessionList
              activeSessionId={selectedSessionId}
              isLoading={isLoading}
              onSelectSession={onSelectSession}
              recentSessionIds={recentSessionIds}
              sessions={sessions}
              totalCount={totalCount}
            />
          </div>
        </section>
      </div>

      <TelemetryDetailSheet
        mode="session"
        onOpenChange={(open) => {
          if (!open) onSelectSession(null);
        }}
        open={Boolean(selectedSessionId)}
        sessionId={selectedSessionId}
      />
      <LangfuseImportDialog
        onImported={refresh}
        onOpenChange={setImportDialogOpen}
        open={importDialogOpen}
      />
      <Dialog
        cancelTitle="Cancel"
        className="sm:!max-w-[520px] md:!w-[520px]"
        confirmButtonVariant="destructive"
        confirmTitle="Clear data"
        dialogDescription="This removes local traces, spans, search rows, ingest batches, and live telemetry history. Saved Langfuse connections stay intact."
        dialogTitle="Clear local telemetry data?"
        disabled={clearDataMutation.isPending}
        loading={clearDataMutation.isPending}
        onConfirm={() => clearDataMutation.mutate()}
        onOpenChange={setClearDialogOpen}
        open={clearDialogOpen}
      >
        <div className="rounded-md border border-destructive-border bg-destructive/5 p-4 text-sm">
          <div className="flex items-start gap-3">
            <Trash2 className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
            <div className="space-y-1">
              <p className="font-medium text-foreground">
                This cannot be undone.
              </p>
              <p className="text-muted-foreground">
                Current local database contains {infoQuery.data?.traceCount ?? 0}{" "}
                traces and {infoQuery.data?.spanCount ?? 0} spans.
              </p>
            </div>
          </div>
        </div>
      </Dialog>
    </main>
  );
}

function TraceTitleBar({
  followLatest,
  health,
  icon,
  isRefreshing,
  liveStatus,
  liveUrl,
  onClearData,
  onCopy,
  onFollowLatestChange,
  onImport,
  onRefresh,
  title,
}: {
  followLatest?: boolean;
  health: string;
  icon: "traces" | "sessions";
  isRefreshing: boolean;
  liveStatus: LiveStatus;
  liveUrl: string;
  onClearData: () => void;
  onCopy: () => void;
  onFollowLatestChange?: (enabled: boolean) => void;
  onImport: () => void;
  onRefresh: () => void;
  title: string;
}) {
  const Icon = icon === "sessions" ? MessageSquare : Activity;
  return (
    <div className="electrobun-webkit-app-region-drag fixed inset-x-0 top-0 z-40 grid h-14 select-none grid-cols-[112px_minmax(0,1fr)] border-b border-subtle bg-background/95 backdrop-blur">
      <div className="border-r border-subtle" />
      <div className="flex min-w-0 items-center justify-between gap-4 px-4">
        <div className="flex min-w-0 items-center gap-3">
          <div className="grid h-8 w-8 shrink-0 place-items-center rounded-md border border-subtle bg-card">
            <Icon className="h-4 w-4 text-detail-brand" />
          </div>
          <div className="min-w-0">
            <p className="text-xs font-medium text-muted-foreground">
              HALO
            </p>
            <p className="truncate text-sm font-semibold">{title}</p>
          </div>
        </div>

        <div className="electrobun-webkit-app-region-no-drag flex min-w-0 items-center gap-2">
          <ToolbarStatus health={health} liveStatus={liveStatus} liveUrl={liveUrl} />
          {onFollowLatestChange ? (
            <Button
              aria-label={
                followLatest ? "Stop following latest trace" : "Follow latest trace"
              }
              aria-pressed={followLatest}
              className={cn(
                "gap-2",
                followLatest && "border-detail-brand/50 text-detail-brand",
              )}
              onClick={() => onFollowLatestChange(!followLatest)}
              size="sm"
              variant={followLatest ? "secondary" : "outline"}
            >
              <Activity
                className={cn("h-4 w-4", followLatest && "animate-pulse")}
              />
              {followLatest ? "Following latest" : "Follow latest"}
            </Button>
          ) : null}
          <Button
            aria-label="Import Langfuse data"
            onClick={onImport}
            size="sm"
            variant="secondary"
          >
            <DownloadCloud className="mr-2 h-4 w-4" />
            Import Data
          </Button>
          <Button
            aria-label="Clear local telemetry data"
            className="border-destructive-border text-destructive hover:bg-destructive/10"
            onClick={onClearData}
            size="sm"
            variant="outline"
          >
            <Trash2 className="mr-2 h-4 w-4" />
            Clear Data
          </Button>
          <Button
            aria-label="Copy ingest URL"
            onClick={onCopy}
            size="sm"
            variant="outline"
          >
            <Copy className="mr-2 h-4 w-4" />
            Copy URL
          </Button>
          <Button
            aria-label="Refresh traces"
            onClick={onRefresh}
            size="icon"
            variant="ghost"
          >
            <RefreshCcw
              className={cn("h-4 w-4", isRefreshing && "animate-spin")}
            />
          </Button>
        </div>
      </div>
    </div>
  );
}

function ToolbarStatus({
  health,
  liveStatus,
  liveUrl,
}: {
  health: string;
  liveStatus: LiveStatus;
  liveUrl: string;
}) {
  const live = liveStatus === "live";
  const reconnecting = liveStatus === "reconnecting" || liveStatus === "connecting";
  const accepted = health === "accepted";
  const waiting = health === "waiting";
  const healthLabel = accepted
    ? "receiving telemetry"
    : waiting
      ? "waiting for telemetry"
      : health.replaceAll("_", " ");
  const liveLabel = live
    ? "live updates connected"
    : reconnecting
      ? "live updates connecting"
      : "live updates offline";
  const label = live
    ? accepted
      ? "Live ingest"
      : waiting
        ? "Live · waiting"
        : `Live · ${healthLabel}`
    : reconnecting
      ? "Connecting"
      : accepted
        ? "Ingest OK"
        : "Offline";
  const variant =
    live && accepted
      ? "status-success"
      : live || reconnecting
        ? "status-brand"
        : accepted
          ? "status-success"
          : "outline";
  return (
    <Badge
      className="gap-2"
      title={`Realtime: ${liveLabel}. Ingest: ${healthLabel}. Socket: ${liveUrl}`}
      variant={variant}
    >
      <span
        aria-hidden
        className={cn(
          "h-1.5 w-1.5 rounded-full",
          live ? "bg-detail-brand" : "bg-muted-foreground",
          reconnecting && "animate-pulse",
        )}
      />
      {accepted ? (
        <CheckCircle2 className="h-3 w-3 shrink-0" />
      ) : waiting || reconnecting ? (
        <Loader2 className="h-3 w-3 shrink-0 animate-spin" />
      ) : (
        <AlertCircle className="h-3 w-3 shrink-0" />
      )}
      <span>{label}</span>
    </Badge>
  );
}

function FilterSidebar({
  agentName,
  dateRange,
  description,
  facets,
  modelName,
  onAgentNameChange,
  onDateRangeChange,
  onModelNameChange,
  onReset,
  onScopeChange,
  onServiceNameChange,
  onSourceChange,
  onStatusChange,
  scope,
  serviceName,
  source,
  status,
}: {
  agentName: string;
  dateRange: DateRange;
  description: string;
  facets: Partial<Record<string, FacetOption[]>>;
  modelName: string;
  onAgentNameChange: (value: string) => void;
  onDateRangeChange: (value: DateRange) => void;
  onModelNameChange: (value: string) => void;
  onReset: () => void;
  onScopeChange: (value: ScopeFilter) => void;
  onServiceNameChange: (value: string) => void;
  onSourceChange: (value: SourceFilter) => void;
  onStatusChange: (value: StatusFilter) => void;
  scope: ScopeFilter;
  serviceName: string;
  source: SourceFilter;
  status: StatusFilter;
}) {
  return (
    <aside className="border-r border-subtle bg-sidebar">
      <div className="flex h-full flex-col gap-5 p-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <Filter className="h-4 w-4" />
            Filters
          </div>
          <p className="text-xs text-muted-foreground">
            {description}
          </p>
        </div>

        <div className="space-y-4">
          <FilterSelect
            icon={<CalendarClock className="h-4 w-4" />}
            label="Window"
            onChange={(value) => onDateRangeChange(value as DateRange)}
            options={[
              { label: "Last hour", value: "1h" },
              { label: "Last 24 hours", value: "24h" },
              { label: "Last 7 days", value: "7d" },
              { label: "All time", value: "all" },
            ]}
            value={dateRange}
          />
          <FilterSelect
            icon={<Activity className="h-4 w-4" />}
            label="Status"
            onChange={(value) => onStatusChange(value as StatusFilter)}
            options={[
              { label: "Any status", value: "all" },
              { label: "OK", value: "ok" },
              { label: "Errors", value: "error" },
            ]}
            value={status}
          />
          <FilterSelect
            icon={<ListTree className="h-4 w-4" />}
            label="Scope"
            onChange={(value) => onScopeChange(value as ScopeFilter)}
            options={[
              { label: "All spans", value: "all" },
              { label: "Root spans", value: "root" },
              { label: "Entrypoints", value: "entrypoint" },
            ]}
            value={scope}
          />
          <FilterSelect
            icon={<DownloadCloud className="h-4 w-4" />}
            label="Source"
            onChange={(value) => onSourceChange(value as SourceFilter)}
            options={toFacetOptions(facets.source, "Any source").map((option) => ({
              ...option,
              label: sourceLabel(option.value, option.label),
            }))}
            value={source}
          />
          <FilterSelect
            icon={<Server className="h-4 w-4" />}
            label="Service"
            onChange={onServiceNameChange}
            options={toFacetOptions(facets.service_name, "Any service")}
            value={serviceName}
          />
          <FilterSelect
            icon={<Boxes className="h-4 w-4" />}
            label="Agent"
            onChange={onAgentNameChange}
            options={toFacetOptions(facets.agent_name, "Any agent")}
            value={agentName}
          />
          <FilterSelect
            icon={<Code2 className="h-4 w-4" />}
            label="Model"
            onChange={onModelNameChange}
            options={toFacetOptions(facets.llm_model_name, "Any model")}
            value={modelName}
          />
        </div>

        <Button className="mt-auto" onClick={onReset} variant="outline">
          Reset filters
        </Button>
      </div>
    </aside>
  );
}

function FilterSelect({
  icon,
  label,
  onChange,
  options,
  value,
}: {
  icon: ReactNode;
  label: string;
  onChange: (value: string) => void;
  options: Array<{ label: string; value: string; count?: number }>;
  value: string;
}) {
  return (
    <label className="block space-y-2">
      <span className="flex items-center gap-2 text-xs font-semibold uppercase text-muted-foreground">
        {icon}
        {label}
      </span>
      <select
        className="h-10 w-full rounded-md border border-subtle bg-background px-3 text-sm"
        onChange={(event) => onChange(event.currentTarget.value)}
        value={value}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.count == null
              ? option.label
              : `${option.label} (${option.count})`}
          </option>
        ))}
      </select>
    </label>
  );
}

function MetricsStrip({
  errorCount,
  isLoading,
  llmSpanCount,
  spanCount,
  totalCost,
  totalTokens,
  traceCount,
}: {
  errorCount: number;
  isLoading: boolean;
  llmSpanCount: number;
  spanCount: number;
  totalCost: number;
  totalTokens: number;
  traceCount: number;
}) {
  return (
    <div className="grid grid-cols-6 border-b border-subtle">
      <MetricTile icon={Activity} label="Traces" loading={isLoading} value={traceCount} />
      <MetricTile icon={Layers3} label="Spans" loading={isLoading} value={spanCount} />
      <MetricTile icon={Code2} label="LLM spans" loading={isLoading} value={llmSpanCount} />
      <MetricTile icon={XCircle} label="Errors" loading={isLoading} value={errorCount} />
      <MetricTile icon={Zap} label="Tokens" loading={isLoading} value={compactNumber(totalTokens)} />
      <MetricTile
        icon={CircleDollarSign}
        label="Cost"
        loading={isLoading}
        value={formatMoney(totalCost)}
      />
    </div>
  );
}

function SessionMetricsStrip({
  errorCount,
  isLoading,
  llmSpanCount,
  sessionCount,
  spanCount,
  totalCost,
  totalTokens,
  traceCount,
}: {
  errorCount: number;
  isLoading: boolean;
  llmSpanCount: number;
  sessionCount: number;
  spanCount: number;
  totalCost: number;
  totalTokens: number;
  traceCount: number;
}) {
  return (
    <div className="grid grid-cols-7 border-b border-subtle">
      <MetricTile icon={MessageSquare} label="Sessions" loading={isLoading} value={sessionCount} />
      <MetricTile icon={Activity} label="Turns" loading={isLoading} value={traceCount} />
      <MetricTile icon={Layers3} label="Spans" loading={isLoading} value={spanCount} />
      <MetricTile icon={Code2} label="LLM spans" loading={isLoading} value={llmSpanCount} />
      <MetricTile icon={XCircle} label="Errors" loading={isLoading} value={errorCount} />
      <MetricTile icon={Zap} label="Tokens" loading={isLoading} value={compactNumber(totalTokens)} />
      <MetricTile
        icon={CircleDollarSign}
        label="Cost"
        loading={isLoading}
        value={formatMoney(totalCost)}
      />
    </div>
  );
}

function MetricTile({
  icon: Icon,
  label,
  loading,
  value,
}: {
  icon: typeof Activity;
  label: string;
  loading: boolean;
  value: ReactNode;
}) {
  return (
    <div className="min-w-0 border-r border-subtle px-4 py-3 last:border-r-0">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Icon className="h-3.5 w-3.5" />
        <span>{label}</span>
      </div>
      <div className="mt-1 text-xl font-semibold">
        {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : value}
      </div>
    </div>
  );
}

function EmptyTraceState({
  ingestUrl,
  onCopy,
}: {
  ingestUrl: string;
  onCopy: () => void;
}) {
  return (
    <div className="grid flex-1 place-items-center p-8">
      <div className="w-full max-w-2xl border border-dashed border-subtle bg-background-muted p-8 text-center">
        <div className="mx-auto grid h-12 w-12 place-items-center rounded-md border border-subtle bg-background">
          <Terminal className="h-5 w-5 text-detail-brand" />
        </div>
        <h2 className="mt-5 text-xl font-semibold">No local traces yet</h2>
        <p className="mx-auto mt-2 max-w-lg text-sm text-muted-foreground">
          Point a local Catalyst or OpenTelemetry JSON exporter at this desktop
          app and emitted spans will appear here live.
        </p>
        <div className="mt-5 flex items-center justify-center gap-2">
          <code className="min-w-0 rounded-md border border-subtle bg-background px-3 py-2 font-mono text-sm">
            {CATALYST_ENV_LINE}
          </code>
          <Button onClick={onCopy} variant="secondary">
            <Copy className="mr-2 h-4 w-4" />
            Copy
          </Button>
        </div>
        <p className="mt-3 font-mono text-xs text-muted-foreground">
          {ingestUrl}
        </p>
      </div>
    </div>
  );
}

function TraceList({
  activeTraceId,
  isLoading,
  onSelectTrace,
  recentTraceIds,
  totalCount,
  traces,
}: {
  activeTraceId?: string;
  isLoading: boolean;
  onSelectTrace: (traceId: string) => void;
  recentTraceIds: Set<string>;
  totalCount: number;
  traces: Trace[];
}) {
  if (isLoading && traces.length === 0) {
    return (
      <div className="grid flex-1 place-items-center">
        <Loader2 className="h-7 w-7 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (traces.length === 0) {
    return (
      <div className="grid flex-1 place-items-center p-8 text-center">
        <div>
          <Search className="mx-auto h-8 w-8 text-muted-foreground" />
          <h2 className="mt-3 text-lg font-semibold">No matching traces</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Broaden the filters or wait for another local ingest batch.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-0 flex-1 overflow-auto">
      <div className="sticky top-0 z-10 grid grid-cols-[minmax(260px,1.3fr)_120px_120px_120px_110px_160px] border-b border-subtle bg-background px-6 py-2 text-xs font-semibold uppercase text-muted-foreground">
        <div>Trace</div>
        <div>Service</div>
        <div>Duration</div>
        <div>Spans</div>
        <div>Tokens</div>
        <div>Started</div>
      </div>
      <div>
        {traces.map((trace) => (
          <button
            className={cn(
              "grid w-full grid-cols-[minmax(260px,1.3fr)_120px_120px_120px_110px_160px] border-b border-subtle px-6 py-4 text-left transition hover:bg-muted/45",
              activeTraceId === trace.traceId && "bg-muted",
              recentTraceIds.has(trace.traceId) && "live-trace-flash",
            )}
            key={trace.traceId}
            onContextMenu={(event) => {
              event.preventDefault();
              void showDesktopRowContextMenu({
                id: trace.traceId,
                kind: "trace",
                sourceUrl: trace.sourceUrl,
              });
            }}
            onClick={() => onSelectTrace(trace.traceId)}
            type="button"
          >
            <div className="min-w-0">
              <div className="flex min-w-0 items-center gap-2">
                <Badge
                  size="sm"
                  variant={trace.hasError ? "status-failure" : "status-success"}
                >
                  {trace.hasError ? "error" : "ok"}
                </Badge>
                {recentTraceIds.has(trace.traceId) ? (
                  <Badge size="sm" variant="status-brand">
                    new
                  </Badge>
                ) : null}
                <TraceSourceBadge trace={trace} />
                <span className="truncate font-medium">
                  {trace.rootSpanName || "unnamed trace"}
                </span>
              </div>
              <div className="mt-1 flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
                <span className="truncate font-mono">{trace.traceId}</span>
                {trace.agentName ? <span>{trace.agentName}</span> : null}
                {trace.source === "langfuse" && trace.sourceTraceId ? (
                  <span className="truncate">
                    Langfuse {trace.sourceTraceId.slice(0, 8)}
                  </span>
                ) : null}
              </div>
            </div>
            <TableCell>{trace.serviceName || "local"}</TableCell>
            <TableCell>{formatDuration(trace.durationMs)}</TableCell>
            <TableCell>
              {trace.spanCount} total
              <span className="ml-1 text-muted-foreground">
                {trace.llmSpanCount} LLM
              </span>
            </TableCell>
            <TableCell>{trace.totalTokens ?? 0}</TableCell>
            <TableCell>{formatTimestamp(trace.startTime)}</TableCell>
          </button>
        ))}
      </div>
      <div className="flex items-center justify-between px-6 py-3 text-xs text-muted-foreground">
        <span>Showing {traces.length} traces</span>
        <span>{totalCount} matching traces</span>
      </div>
    </div>
  );
}

function SessionList({
  activeSessionId,
  isLoading,
  onSelectSession,
  recentSessionIds,
  sessions,
  totalCount,
}: {
  activeSessionId?: string;
  isLoading: boolean;
  onSelectSession: (sessionId: string) => void;
  recentSessionIds: Set<string>;
  sessions: SessionSummary[];
  totalCount: number;
}) {
  if (isLoading && sessions.length === 0) {
    return (
      <div className="grid flex-1 place-items-center">
        <Loader2 className="h-7 w-7 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (sessions.length === 0) {
    return (
      <div className="grid flex-1 place-items-center p-8 text-center">
        <div>
          <MessageSquare className="mx-auto h-8 w-8 text-muted-foreground" />
          <h2 className="mt-3 text-lg font-semibold">No sessions yet</h2>
          <p className="mt-1 max-w-md text-sm text-muted-foreground">
            Sessions appear when traces include a session ID. Traces without one
            stay hidden here.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-0 flex-1 overflow-auto">
      <div className="sticky top-0 z-10 grid grid-cols-[minmax(280px,1.4fr)_150px_120px_120px_120px_160px] border-b border-subtle bg-background px-6 py-2 text-xs font-semibold uppercase text-muted-foreground">
        <div>Session</div>
        <div>Services</div>
        <div>Turns</div>
        <div>Spans</div>
        <div>Tokens</div>
        <div>Last activity</div>
      </div>
      <div>
        {sessions.map((session) => (
          <button
            className={cn(
              "grid w-full grid-cols-[minmax(280px,1.4fr)_150px_120px_120px_120px_160px] border-b border-subtle px-6 py-4 text-left transition hover:bg-muted/45",
              activeSessionId === session.sessionId && "bg-muted",
              recentSessionIds.has(session.sessionId) && "live-trace-flash",
            )}
            key={session.sessionId}
            onContextMenu={(event) => {
              event.preventDefault();
              void showDesktopRowContextMenu({
                id: session.sessionId,
                kind: "session",
              });
            }}
            onClick={() => onSelectSession(session.sessionId)}
            type="button"
          >
            <div className="min-w-0">
              <div className="flex min-w-0 items-center gap-2">
                <Badge
                  size="sm"
                  variant={session.hasError ? "status-failure" : "status-success"}
                >
                  {session.hasError ? "error" : "ok"}
                </Badge>
                {recentSessionIds.has(session.sessionId) ? (
                  <Badge size="sm" variant="status-brand">
                    live
                  </Badge>
                ) : null}
                <SessionSourceBadge session={session} />
                <span className="truncate font-medium">
                  {session.latestTraceName || "session"}
                </span>
              </div>
              <div className="mt-1 flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
                <span className="truncate font-mono">{session.sessionId}</span>
                {session.agentNames[0] ? <span>{session.agentNames[0]}</span> : null}
                {session.llmModelNames[0] ? (
                  <span className="truncate">{session.llmModelNames[0]}</span>
                ) : null}
              </div>
            </div>
            <TableCell>{shortList(session.serviceNames, "local")}</TableCell>
            <TableCell>{session.traceCount}</TableCell>
            <TableCell>
              {session.spanCount} total
              <span className="ml-1 text-muted-foreground">
                {session.llmSpanCount} LLM
              </span>
            </TableCell>
            <TableCell>{session.totalTokens ?? 0}</TableCell>
            <TableCell>{formatTimestamp(session.endTime)}</TableCell>
          </button>
        ))}
      </div>
      <div className="flex items-center justify-between px-6 py-3 text-xs text-muted-foreground">
        <span>Showing {sessions.length} sessions</span>
        <span>{totalCount} matching sessions</span>
      </div>
    </div>
  );
}

function TableCell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-w-0 items-center text-sm text-muted-foreground">
      <span className="truncate">{children}</span>
    </div>
  );
}

function TraceSourceBadge({ trace }: { trace: Trace }) {
  if (trace.source !== "langfuse") return null;
  const title = [
    trace.sourceConnectionName ? `Connection: ${trace.sourceConnectionName}` : null,
    trace.sourceImportedAt ? `Imported: ${formatTimestamp(trace.sourceImportedAt)}` : null,
    trace.sourceTraceId ? `Langfuse trace: ${trace.sourceTraceId}` : null,
  ]
    .filter(Boolean)
    .join("\n");
  return (
    <Badge className="gap-1" size="sm" title={title} variant="status-brand">
      <DownloadCloud className="h-3 w-3" />
      Langfuse
    </Badge>
  );
}

function SessionSourceBadge({ session }: { session: SessionSummary }) {
  if (!session.sources.includes("langfuse")) return null;
  const mixed = session.sources.includes("local");
  const title = [
    mixed ? "Includes local and Langfuse traces" : "Imported from Langfuse",
    ...session.sourceConnectionNames.map((name) => `Connection: ${name}`),
  ].join("\n");
  return (
    <Badge className="gap-1" size="sm" title={title} variant="status-brand">
      <DownloadCloud className="h-3 w-3" />
      {mixed ? "Mixed" : "Langfuse"}
    </Badge>
  );
}

function TelemetryDetailSheet({
  followLatest,
  mode,
  onOpenChange,
  open,
  sessionId,
  traceId,
}: {
  followLatest?: boolean;
  mode: "trace" | "session";
  onOpenChange: (open: boolean) => void;
  open: boolean;
  sessionId?: string;
  traceId?: string;
}) {
  const [activeDetailTab, setActiveDetailTab] = useState<DetailTab>("tree");
  const [selectedSpanKey, setSelectedSpanKey] = useState<string | null>(null);
  const [recentSpanIds, setRecentSpanIds] = useState<Set<string>>(() => new Set());
  const recentSpanTimers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const detailTraceInput = useMemo(() => ({ traceId: traceId ?? "" }), [traceId]);
  const detailSpansInput = useMemo(
    () => ({ limit: 500, traceId: traceId ?? "" }),
    [traceId],
  );
  const detailSessionInput = useMemo(
    () => ({ sessionId: sessionId ?? "" }),
    [sessionId],
  );
  const detailSessionSpansInput = useMemo(
    () => ({ limit: 1000, sessionId: sessionId ?? "" }),
    [sessionId],
  );
  const detailSessionTracesInput = useMemo(
    () => ({ limit: 500, sessionId: sessionId ?? "" }),
    [sessionId],
  );
  const utils = trpc.useUtils();
  const traceQuery = trpc.traces.get.useQuery(detailTraceInput, {
    enabled: mode === "trace" && open && Boolean(traceId),
  });
  const spansQuery = trpc.traces.getSpans.useQuery(detailSpansInput, {
    enabled: mode === "trace" && open && Boolean(traceId),
  });
  const sessionQuery = trpc.sessions.get.useQuery(detailSessionInput, {
    enabled: mode === "session" && open && Boolean(sessionId),
  });
  const sessionSpansQuery = trpc.sessions.getSpans.useQuery(detailSessionSpansInput, {
    enabled: mode === "session" && open && Boolean(sessionId),
  });
  const sessionTracesQuery = trpc.sessions.getTraces.useQuery(detailSessionTracesInput, {
    enabled: mode === "session" && open && Boolean(sessionId),
  });

  const markRecentSpanId = useCallback((span: Span) => {
    const key = spanKey(span);
    setRecentSpanIds((current) => {
      const next = new Set(current);
      next.add(key);
      return next;
    });
    const existing = recentSpanTimers.current.get(key);
    if (existing) clearTimeout(existing);
    const timer = setTimeout(() => {
      recentSpanTimers.current.delete(key);
      setRecentSpanIds((current) => {
        const next = new Set(current);
        next.delete(key);
        return next;
      });
    }, 2_200);
    recentSpanTimers.current.set(key, timer);
  }, []);

  trpc.live.trace.useSubscription(detailTraceInput, {
    enabled: mode === "trace" && open && Boolean(traceId),
    onData(eventEnvelope) {
      const event = eventEnvelope.data;
      if (!traceId) return;
      if (event.payload.type === "span.upserted") {
        const span = event.payload.span;
        if (span.traceId !== traceId) return;
        markRecentSpanId(span);
        utils.traces.getSpans.setData(detailSpansInput, (current) => {
          if (!current) return current;
          const spans = upsertSpan(current.spans, span);
          return {
            ...current,
            spans,
            tree: buildClientSpanTree(spans),
          };
        });
        return;
      }
      if (
        event.payload.type === "trace.upserted" &&
        event.payload.trace.traceId === traceId
      ) {
        utils.traces.get.setData(detailTraceInput, event.payload.trace);
      }
    },
  });

  trpc.live.workspace.useSubscription(undefined, {
    enabled: mode === "session" && open && Boolean(sessionId),
    onData(eventEnvelope) {
      const event = eventEnvelope.data;
      if (!sessionId) return;
      const traceIds = new Set(
        sessionTracesQuery.data?.traces.map((trace) => trace.traceId) ?? [],
      );
      if (event.payload.type === "span.upserted") {
        const span = event.payload.span;
        if (span.sessionId !== sessionId && !traceIds.has(span.traceId)) return;
        markRecentSpanId(span);
        void utils.sessions.getSpans.invalidate(detailSessionSpansInput);
        void utils.sessions.get.invalidate(detailSessionInput);
        void utils.sessions.getTraces.invalidate(detailSessionTracesInput);
      }
      if (event.payload.type === "trace.upserted") {
        const trace = event.payload.trace;
        if (trace.sessionId !== sessionId && !traceIds.has(trace.traceId)) return;
        void utils.sessions.get.invalidate(detailSessionInput);
        void utils.sessions.getSpans.invalidate(detailSessionSpansInput);
        void utils.sessions.getTraces.invalidate(detailSessionTracesInput);
      }
    },
  });

  useEffect(
    () => () => {
      for (const timer of recentSpanTimers.current.values()) {
        clearTimeout(timer);
      }
    },
    [],
  );

  useEffect(() => {
    setSelectedSpanKey(null);
    setRecentSpanIds(new Set());
  }, [sessionId, traceId]);

  const spans =
    mode === "session"
      ? (sessionSpansQuery.data?.spans ?? EMPTY_SPANS)
      : (spansQuery.data?.spans ?? EMPTY_SPANS);
  const sessionTraces = sessionTracesQuery.data?.traces ?? [];
  const displayTree = useMemo(
    () =>
      mode === "session"
        ? buildSessionSpanTree(spans, sessionTraces)
        : buildClientSpanTree(spans),
    [mode, sessionTraces, spans],
  );
  const displaySpans = useMemo(() => flattenSpanTree(displayTree), [displayTree]);
  const firstInspectableSpan = useMemo(
    () => findFirstInspectableSpan(displayTree) ?? displaySpans[0] ?? null,
    [displaySpans, displayTree],
  );
  const firstInspectableSpanKey = firstInspectableSpan
    ? spanKey(firstInspectableSpan)
    : null;
  const timelineSpans = useMemo(
    () =>
      mode === "session"
        ? displaySpans.filter((span) => !isSessionTraceGroupSpan(span))
        : displaySpans,
    [displaySpans, mode],
  );

  useEffect(() => {
    if (!open) {
      setSelectedSpanKey(null);
      setRecentSpanIds(new Set());
    }
  }, [open]);

  useEffect(() => {
    if (open && firstInspectableSpanKey && !selectedSpanKey) {
      setSelectedSpanKey(firstInspectableSpanKey);
    }
  }, [firstInspectableSpanKey, open, selectedSpanKey]);

  const selectedSpanCandidate =
    displaySpans.find((span) => spanKey(span) === selectedSpanKey) ?? null;
  const selectedSpan =
    selectedSpanCandidate && !isSessionTraceGroupSpan(selectedSpanCandidate)
      ? selectedSpanCandidate
      : (firstInspectableSpan ?? null);
  const session = sessionQuery.data ?? null;
  const traceMap = useMemo(
    () => new Map(sessionTraces.map((trace) => [trace.traceId, trace])),
    [sessionTraces],
  );
  const trace =
    mode === "session"
      ? selectedSpan
        ? (traceMap.get(selectedSpan.traceId) ?? null)
        : null
      : (traceQuery.data ?? null);
  const waitingForLatest = mode === "trace" && followLatest && !traceId;
  const rootSpan = displayTree[0]?.span ?? null;
  const title =
    waitingForLatest
      ? "Waiting for next trace..."
      : mode === "session"
        ? (session?.latestTraceName || "Session detail")
      : rootSpan && isSyntheticSpan(rootSpan)
        ? rootSpan.spanName
        : (trace?.rootSpanName ?? rootSpan?.spanName ?? "Trace detail");
  const description =
    mode === "session"
      ? (sessionId ?? "Session")
      : waitingForLatest
        ? "Fire a local request and the sheet will switch automatically."
        : traceId;
  const loading =
    mode === "session"
      ? sessionQuery.isLoading || sessionSpansQuery.isLoading
      : spansQuery.isLoading;

  return (
    <Sheet onOpenChange={onOpenChange} open={open}>
      <SheetContent
        className="flex w-[80vw] max-w-[80vw] flex-col overflow-hidden p-0 max-md:w-[92vw] max-md:max-w-[92vw] sm:max-w-[80vw]"
        side="right"
      >
        <SheetHeader className="border-b border-subtle px-6 py-5 pr-12">
          <div className="flex min-w-0 items-start justify-between gap-4">
            <div className="min-w-0">
              <SheetTitle className="truncate text-lg font-semibold">
                {title}
              </SheetTitle>
              <SheetDescription className="mt-1 truncate font-mono">
                {description}
              </SheetDescription>
              {session ? (
                <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                  <SessionSourceBadge session={session} />
                  <span>{session.traceCount} turns</span>
                  <span>{session.spanCount} spans</span>
                  {session.agentNames.slice(0, 2).map((agent) => (
                    <Badge key={agent} size="sm" variant="outline">
                      {agent}
                    </Badge>
                  ))}
                </div>
              ) : null}
              {trace?.source === "langfuse" ? (
                <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                  <TraceSourceBadge trace={trace} />
                  {trace.sourceConnectionName ? (
                    <span>{trace.sourceConnectionName}</span>
                  ) : null}
                  {trace.sourceImportedAt ? (
                    <span>Imported {relativeTime(trace.sourceImportedAt)}</span>
                  ) : null}
                  {trace.sourceTags.slice(0, 3).map((tag) => (
                    <Badge key={tag} size="sm" variant="outline">
                      {tag}
                    </Badge>
                  ))}
                </div>
              ) : null}
            </div>
            <div className="flex shrink-0 items-center gap-2">
              {mode === "trace" && followLatest ? (
                <Badge className="gap-1.5" variant="status-brand">
                  <Activity className="h-3 w-3 animate-pulse" />
                  Following latest
                </Badge>
              ) : null}
              {session ? (
                <Badge
                  variant={session.hasError ? "status-failure" : "status-success"}
                >
                  {session.hasError ? "error" : "ok"}
                </Badge>
              ) : trace ? (
                <Badge
                  variant={trace.hasError ? "status-failure" : "status-success"}
                >
                  {trace.hasError ? "error" : "ok"}
                </Badge>
              ) : null}
            </div>
          </div>
        </SheetHeader>

        {waitingForLatest ? (
          <WaitingForLatestTrace />
        ) : loading ? (
          <div className="grid flex-1 place-items-center">
            <Loader2 className="h-7 w-7 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <Tabs
            className="flex min-h-0 flex-1 flex-col"
            onValueChange={(value) => setActiveDetailTab(value as DetailTab)}
            value={activeDetailTab}
          >
            <div className="border-b border-subtle px-6 py-3">
              <TabsList>
                <TabsTrigger value="tree">
                  <ListTree className="mr-2 h-4 w-4" />
                  Tree
                </TabsTrigger>
                <TabsTrigger value="timeline">
                  <Activity className="mr-2 h-4 w-4" />
                  Timeline
                </TabsTrigger>
                <TabsTrigger value="span">
                  <Braces className="mr-2 h-4 w-4" />
                  Span
                </TabsTrigger>
                <TabsTrigger value="raw">
                  <Code2 className="mr-2 h-4 w-4" />
                  Raw
                </TabsTrigger>
              </TabsList>
            </div>

            <TabsContent className="min-h-0 flex-1 overflow-auto p-0" value="tree">
              <div className="grid min-h-full grid-cols-[360px_minmax(0,1fr)]">
                <div className="border-r border-subtle p-4">
                  <SpanTreeList
                    nodes={displayTree}
                    onSelectSpan={setSelectedSpanKey}
                    recentSpanIds={recentSpanIds}
                    selectedSpanId={selectedSpan ? spanKey(selectedSpan) : undefined}
                  />
                </div>
                <SpanInspector span={selectedSpan} trace={trace} />
              </div>
            </TabsContent>

            <TabsContent className="min-h-0 flex-1 overflow-auto p-6" value="timeline">
              <Timeline recentSpanIds={recentSpanIds} spans={timelineSpans} />
            </TabsContent>

            <TabsContent className="min-h-0 flex-1 overflow-auto p-0" value="span">
              <SpanInspector span={selectedSpan} trace={trace} />
            </TabsContent>

            <TabsContent className="min-h-0 flex-1 overflow-auto p-6" value="raw">
              <pre className="overflow-auto rounded-md border border-subtle bg-background-muted p-4 text-xs">
                {JSON.stringify({ session, spans, trace, traces: sessionTraces }, null, 2)}
              </pre>
            </TabsContent>
          </Tabs>
        )}
      </SheetContent>
    </Sheet>
  );
}

function WaitingForLatestTrace() {
  return (
    <div className="grid flex-1 place-items-center p-8">
      <div className="max-w-md border border-dashed border-subtle bg-background-muted p-8 text-center">
        <div className="mx-auto grid h-12 w-12 place-items-center rounded-md border border-subtle bg-background">
          <Activity className="h-5 w-5 animate-pulse text-detail-brand" />
        </div>
        <h3 className="mt-5 text-lg font-semibold">Waiting for next trace</h3>
        <p className="mt-2 text-sm text-muted-foreground">
          Keep this sheet open and fire a local request. The newest trace will
          appear here as soon as its first span is ingested.
        </p>
      </div>
    </div>
  );
}

function SpanTreeList({
  nodes,
  onSelectSpan,
  recentSpanIds,
  selectedSpanId,
}: {
  nodes: SpanNode[];
  onSelectSpan: (spanId: string) => void;
  recentSpanIds: Set<string>;
  selectedSpanId?: string;
}) {
  if (nodes.length === 0) {
    return <p className="text-sm text-muted-foreground">No spans found.</p>;
  }

  return (
    <div className="space-y-1">
      {nodes.map((node) => (
        <SpanTreeNode
          key={spanKey(node.span)}
          node={node}
          onSelectSpan={onSelectSpan}
          recentSpanIds={recentSpanIds}
          selectedSpanId={selectedSpanId}
        />
      ))}
    </div>
  );
}

function SpanTreeNode({
  depth = 0,
  node,
  onSelectSpan,
  recentSpanIds,
  selectedSpanId,
}: {
  depth?: number;
  node: SpanNode;
  onSelectSpan: (spanId: string) => void;
  recentSpanIds: Set<string>;
  selectedSpanId?: string;
}) {
  const key = spanKey(node.span);
  const sessionGroup = isSessionTraceGroupSpan(node.span);
  const inspectableSpan = sessionGroup
    ? findFirstInspectableSpan(node.children)
    : node.span;
  const inspectableKey = inspectableSpan ? spanKey(inspectableSpan) : null;
  const active = selectedSpanId === key;
  const recent = recentSpanIds.has(key);
  const synthetic = isSyntheticSpan(node.span);
  const traceName = node.span.spanAttributes["halo.synthetic.trace_name"];
  return (
    <div>
      <button
        className={cn(
          "flex w-full items-center gap-2 rounded-md px-2 py-2 text-left text-sm transition hover:bg-muted",
          active && !sessionGroup && "bg-muted text-foreground",
          synthetic && "border border-dashed border-detail-brand/30 bg-detail-brand/5",
          recent && "live-span-flash",
        )}
        onClick={() => {
          if (inspectableKey) onSelectSpan(inspectableKey);
        }}
        style={{ paddingLeft: `${8 + depth * 18}px` }}
        type="button"
      >
        <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground" />
        <Badge size="sm" variant={kindVariant(node.span.observationKind)}>
          {node.span.observationKind}
        </Badge>
        <span className="min-w-0 flex-1">
          <span className="block truncate">{node.span.spanName}</span>
          {sessionGroup && traceName ? (
            <span className="block truncate text-[11px] text-muted-foreground">
              {traceName}
            </span>
          ) : null}
        </span>
        {synthetic ? (
          <Badge size="sm" variant="outline">
            {syntheticBadgeLabel(node.span)}
          </Badge>
        ) : null}
        {recent ? (
          <span className="h-1.5 w-1.5 rounded-full bg-detail-brand" />
        ) : null}
        <span className="text-xs text-muted-foreground">
          {formatDuration(node.span.durationMs)}
        </span>
      </button>
      {node.children.map((child) => (
        <SpanTreeNode
          depth={depth + 1}
          key={spanKey(child.span)}
          node={child}
          onSelectSpan={onSelectSpan}
          recentSpanIds={recentSpanIds}
          selectedSpanId={selectedSpanId}
        />
      ))}
    </div>
  );
}

function SpanInspector({
  span,
  trace,
}: {
  span: Span | null;
  trace: Trace | null;
}) {
  if (!span) {
    return (
      <div className="grid min-h-full place-items-center p-6 text-muted-foreground">
        Select a span to inspect it.
      </div>
    );
  }

  const attributes = {
    resource: span.resourceAttributes,
    span: span.spanAttributes,
    spanDouble: span.spanAttributesDouble,
    spanInt: span.spanAttributesInt,
  };
  const synthetic = isSyntheticSpan(span);

  return (
    <div className="min-w-0 space-y-5 p-6">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={kindVariant(span.observationKind)}>
            {span.observationKind}
          </Badge>
          {synthetic ? (
            <Badge variant="status-brand">{syntheticBadgeLabel(span)}</Badge>
          ) : null}
          <Badge variant={span.statusCode.includes("ERROR") ? "status-failure" : "outline"}>
            {span.statusCode.replace("STATUS_CODE_", "").toLowerCase()}
          </Badge>
          {span.llmModelName ? <Badge variant="secondary">{span.llmModelName}</Badge> : null}
        </div>
        <h3 className="mt-3 truncate text-xl font-semibold">{span.spanName}</h3>
        <p className="mt-1 truncate font-mono text-xs text-muted-foreground">
          {span.spanId}
        </p>
      </div>

      <div className="grid grid-cols-4 gap-3">
        <MiniStat label="Duration" value={formatDuration(span.durationMs)} />
        <MiniStat label="Tokens" value={span.totalTokens ?? 0} />
        <MiniStat label="Cost" value={formatMoney(Number(span.costTotal ?? 0))} />
        <MiniStat label="Service" value={span.serviceName || trace?.serviceName || "local"} />
      </div>

      <Separator />

      <div className="grid gap-4">
        <TextBlock
          empty="No captured input"
          label="Input"
          value={span.inputMessages ?? span.input}
        />
        <TextBlock
          empty="No captured output"
          label="Output"
          value={span.outputMessages ?? span.output}
        />
      </div>

      <Separator />

      <div className="grid grid-cols-2 gap-4">
        <JsonBlock label="Attributes" value={attributes} />
        <JsonBlock label="Events" value={span.events} />
      </div>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="min-w-0 border border-subtle bg-background-muted px-3 py-2">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-1 truncate text-sm font-semibold">{value}</p>
    </div>
  );
}

function TextBlock({
  empty,
  label,
  value,
}: {
  empty: string;
  label: string;
  value: string | null | undefined;
}) {
  return (
    <div>
      <p className="mb-2 text-sm font-semibold">{label}</p>
      <pre className="max-h-56 overflow-auto whitespace-pre-wrap rounded-md border border-subtle bg-background-muted p-3 text-xs leading-relaxed">
        {prettyMaybeJson(value) || empty}
      </pre>
    </div>
  );
}

function JsonBlock({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="min-w-0">
      <p className="mb-2 text-sm font-semibold">{label}</p>
      <pre className="max-h-72 overflow-auto rounded-md border border-subtle bg-background-muted p-3 text-xs">
        {JSON.stringify(value, null, 2)}
      </pre>
    </div>
  );
}

function spanKey(span: Span) {
  return `${span.traceId}:${span.spanId}`;
}

function syntheticBadgeLabel(span: Span) {
  return isSessionTraceGroupSpan(span) ? "turn" : "pending";
}

function Timeline({
  recentSpanIds,
  spans,
}: {
  recentSpanIds: Set<string>;
  spans: Span[];
}) {
  if (spans.length === 0) {
    return <p className="text-sm text-muted-foreground">No spans to render.</p>;
  }
  const minStart = Math.min(...spans.map((span) => span.startTimeMs));
  const maxEnd = Math.max(...spans.map((span) => span.endTimeMs));
  const total = Math.max(1, maxEnd - minStart);

  return (
    <div className="space-y-3">
      {spans.map((span) => {
        const key = spanKey(span);
        const left = ((span.startTimeMs - minStart) / total) * 100;
        const width = Math.max(1, (span.durationMs / total) * 100);
        const recent = recentSpanIds.has(key);
        const synthetic = isSyntheticSpan(span);
        return (
          <div
            className={cn(
              "grid grid-cols-[240px_minmax(0,1fr)_90px] items-center gap-3 rounded-md px-2 py-1 transition",
              synthetic && "border border-dashed border-detail-brand/25 bg-detail-brand/5",
              recent && "live-span-flash",
            )}
            key={key}
          >
            <div className="min-w-0">
              <p className="truncate text-sm font-medium">{span.spanName}</p>
              <p className="text-xs text-muted-foreground">{span.observationKind}</p>
            </div>
            <div className="relative h-8 rounded-md border border-subtle bg-background-muted">
              <div
                className={cn(
                  "absolute top-1/2 h-3 -translate-y-1/2 rounded-full transition-[left,width]",
                  span.statusCode.includes("ERROR")
                    ? "bg-detail-failure"
                    : synthetic
                      ? "bg-detail-brand/40"
                      : "bg-detail-brand",
                  recent && "live-timeline-pulse",
                )}
                style={{ left: `${left}%`, width: `${width}%` }}
              />
            </div>
            <div className="text-right text-sm text-muted-foreground">
              {formatDuration(span.durationMs)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ClipboardButton({ value }: { value: string }) {
  return (
    <Button
      onClick={async () => {
        await navigator.clipboard.writeText(value);
        toast.success({ title: "Copied" });
      }}
      size="icon"
      variant="ghost"
    >
      <Clipboard className="h-4 w-4" />
    </Button>
  );
}

function upsertSpan(spans: Span[], span: Span) {
  const index = spans.findIndex((item) => spanKey(item) === spanKey(span));
  const next = [...spans];
  if (index === -1) next.push(span);
  else next[index] = span;
  return next.sort((a, b) =>
    a.startTimeMs === b.startTimeMs
      ? a.spanId.localeCompare(b.spanId)
      : a.startTimeMs - b.startTimeMs,
  );
}

function toFacetOptions(options: FacetOption[] | undefined, allLabel: string) {
  return [
    { label: allLabel, value: "all" },
    ...(options ?? []).map((option) => ({
      count: option.count,
      label: option.label || option.value,
      value: option.value,
    })),
  ];
}

function sourceLabel(value: string, fallback = value) {
  if (value === "local") return "Local";
  if (value === "langfuse") return "Langfuse";
  return fallback;
}

function startDateForRange(range: DateRange) {
  if (range === "all") return undefined;
  const ms =
    range === "1h"
      ? 60 * 60 * 1000
      : range === "24h"
        ? 24 * 60 * 60 * 1000
        : 7 * 24 * 60 * 60 * 1000;
  return new Date(Date.now() - ms);
}

function summarizeVisibleTraces(traces: Trace[]) {
  return traces.reduce(
    (acc, trace) => {
      acc.errorCount += trace.hasError ? 1 : 0;
      acc.llmSpanCount += trace.llmSpanCount;
      acc.spanCount += trace.spanCount;
      acc.totalCost += Number(trace.totalCost ?? 0);
      acc.totalTokens += trace.totalTokens ?? 0;
      return acc;
    },
    {
      errorCount: 0,
      llmSpanCount: 0,
      spanCount: 0,
      totalCost: 0,
      totalTokens: 0,
    },
  );
}

function summarizeVisibleSessions(sessions: SessionSummary[]) {
  return sessions.reduce(
    (acc, session) => {
      acc.errorCount += session.hasError ? 1 : 0;
      acc.llmSpanCount += session.llmSpanCount;
      acc.spanCount += session.spanCount;
      acc.totalCost += Number(session.totalCost ?? 0);
      acc.totalTokens += session.totalTokens ?? 0;
      acc.traceCount += session.traceCount;
      return acc;
    },
    {
      errorCount: 0,
      llmSpanCount: 0,
      spanCount: 0,
      totalCost: 0,
      totalTokens: 0,
      traceCount: 0,
    },
  );
}

function shortList(values: string[], fallback: string) {
  if (values.length === 0) return fallback;
  if (values.length === 1) return values[0];
  return `${values[0]} +${values.length - 1}`;
}

function kindVariant(kind: string) {
  if (kind === "LLM") return "status-brand";
  if (kind === "TOOL") return "status-warning";
  if (kind === "AGENT" || kind === "CHAIN") return "status-running";
  return "outline";
}

function compactNumber(value: number) {
  return new Intl.NumberFormat(undefined, { notation: "compact" }).format(value);
}

function formatDuration(ms: number) {
  if (ms < 1) return "<1 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)} s`;
  return `${(ms / 60_000).toFixed(1)} min`;
}

function formatMoney(value: number) {
  if (!value) return "$0";
  if (value < 0.01) return `$${value.toFixed(5)}`;
  return `$${value.toFixed(2)}`;
}

function formatTimestamp(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "unknown";
  return new Intl.DateTimeFormat(undefined, {
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    month: "short",
  }).format(date);
}

function relativeTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "recently";
  const seconds = Math.max(1, Math.round((Date.now() - date.getTime()) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  return `${hours}h ago`;
}

function prettyMaybeJson(value: string | null | undefined) {
  if (!value) return "";
  try {
    return JSON.stringify(JSON.parse(value), null, 2);
  } catch {
    return value;
  }
}

export function TraceSetupLink() {
  return (
    <Link
      className={buttonVariants({ size: "sm", variant: "outline" })}
      to="/components"
    >
      Component gallery
    </Link>
  );
}
