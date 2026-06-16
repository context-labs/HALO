import {
  useDeferredValue,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { keepPreviousData } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import {
  Activity,
  ArrowUpDown,
  DownloadCloud,
  MoreHorizontal,
  RefreshCcw,
  Search,
} from "lucide-react";

import {
  Button,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  Input,
  cn,
} from "~/lib/ui";
import { trpc } from "~/trpc";
import { WorkspaceNav } from "~/workspace/WorkspaceNav";
import {
  TRACE_PAGE_COMMAND_EVENT,
} from "~/desktop/desktopBridge";
import { AppHeader } from "~/components/AppHeader";
import { FilterSelect } from "~/components/FilterSelect";
import { relativeTime, startDateForRange, type DateRange } from "~/lib/format";
import { ImportDataScreen, LocalAgentSetupDialog } from "./ImportDataScreen";
import { LangfuseImportDialog } from "./langfuse/LangfuseImportDialog";
import { PhoenixImportDialog } from "./phoenix/PhoenixImportDialog";
import { FileImportDialog } from "./fileimport/FileImportDialog";
import { LiveStatusBadge, type LiveStatus } from "./TraceTitleBar";
import type {
  SessionSortKey,
  SessionSummary,
  Trace,
  TraceSortKey,
} from "../../server/telemetry/types";
import {
  nextFollowLatestTraceId,
  traceIdsForLiveEvent,
} from "./followLatest";
import { FilterSidebar } from "./FilterSidebar";
import { LiveRangeControl } from "./logTable";
import { SessionList } from "./SessionList";
import { TelemetryStatStrip } from "./TelemetryStatStrip";
import { TraceList } from "./TraceList";
import { TelemetryDetailSheet } from "./detail/TelemetryDetailSheet";
import {
  TELEMETRY_FACET_IDS,
  type ScopeFilter,
  type SourceFilter,
  type StatusFilter,
  type TraceMonitorViewMode,
} from "./filters";

const DEFAULT_INGEST_URL = "http://127.0.0.1:8799/v1/traces";

export type { TraceMonitorViewMode } from "./filters";

export function TraceMonitorPage({
  followLatest,
  onFollowLatestChange,
  onSelectLatestTrace,
  onSelectSession,
  onSelectTrace,
  onOpenImportData,
  onViewModeChange,
  selectedSessionId,
  selectedTraceId,
  viewMode,
}: {
  followLatest: boolean;
  onFollowLatestChange: (enabled: boolean) => void;
  onSelectLatestTrace: (traceId: string) => void;
  onSelectSession: (sessionId: string | null) => void;
  onSelectTrace: (traceId: string | null) => void;
  onOpenImportData: () => void;
  onViewModeChange: (viewMode: TraceMonitorViewMode) => void;
  selectedSessionId?: string;
  selectedTraceId?: string;
  viewMode: TraceMonitorViewMode;
}) {
  const isTracesMode = viewMode === "traces";
  const [searchText, setSearchText] = useState("");
  const [dateRange, setDateRange] = useState<DateRange>("24h");
  const [status, setStatus] = useState<StatusFilter>("all");
  const [scope, setScope] = useState<ScopeFilter>("all");
  const [serviceName, setServiceName] = useState("all");
  const [agentName, setAgentName] = useState("all");
  const [modelName, setModelName] = useState("all");
  const [source, setSource] = useState<SourceFilter>("all");
  const [traceSortBy, setTraceSortBy] = useState<TraceSortKey>("start_time");
  const [sessionSortBy, setSessionSortBy] =
    useState<SessionSortKey>("last_activity");
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [phoenixDialogOpen, setPhoenixDialogOpen] = useState(false);
  const [fileDialogOpen, setFileDialogOpen] = useState(false);
  const [localAgentSetupOpen, setLocalAgentSetupOpen] = useState(false);
  const [liveStatus, setLiveStatus] = useState<LiveStatus>("connecting");
  const [recentTraceIds, setRecentTraceIds] = useState<Set<string>>(() => new Set());
  const [recentSessionIds, setRecentSessionIds] = useState<Set<string>>(
    () => new Set(),
  );
  const recentTraceTimers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const recentSessionTimers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const workspaceInvalidateTimer = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );
  const followLatestRef = useRef(followLatest);
  const selectedTraceIdRef = useRef<string | undefined>(selectedTraceId);
  const viewModeRef = useRef(viewMode);
  const utils = trpc.useUtils();

  useEffect(() => {
    followLatestRef.current = followLatest;
    selectedTraceIdRef.current = selectedTraceId;
    viewModeRef.current = viewMode;
  }, [followLatest, selectedTraceId, viewMode]);

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
      if (event.payload.type === "trace.upserted") {
        markRecentSessionId(event.payload.trace.sessionId);
      }
      if (event.payload.type === "span.upserted") {
        markRecentSessionId(event.payload.span.sessionId);
      }
      if (viewModeRef.current === "traces") {
        const latestTraceId = nextFollowLatestTraceId({
          currentTraceId: selectedTraceIdRef.current,
          event,
          followLatest: followLatestRef.current,
        });
        if (latestTraceId) {
          selectedTraceIdRef.current = latestTraceId;
          onSelectLatestTrace(latestTraceId);
        }
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
  // keepPreviousData everywhere below: filter/sort changes swap query keys,
  // and showing the previous rows beats flashing a spinner.
  const traceFacetsQuery = trpc.traces.facets.useQuery(
    { facetIds: TELEMETRY_FACET_IDS },
    { enabled: isTracesMode, placeholderData: keepPreviousData },
  );
  const traceListQuery = trpc.traces.list.useQuery(
    {
      filters,
      limit: 75,
      sortBy: traceSortBy,
      sortOrder: "desc",
    },
    {
      enabled: isTracesMode && activeSearch.length === 0,
      placeholderData: keepPreviousData,
    },
  );
  const traceSearchQuery = trpc.traces.search.useQuery(
    {
      filters,
      limit: 75,
      query: activeSearch,
    },
    {
      enabled: isTracesMode && activeSearch.length > 0,
      placeholderData: keepPreviousData,
    },
  );
  const sessionFacetsQuery = trpc.sessions.facets.useQuery(
    { facetIds: TELEMETRY_FACET_IDS },
    { enabled: !isTracesMode, placeholderData: keepPreviousData },
  );
  const sessionListQuery = trpc.sessions.list.useQuery(
    {
      filters,
      limit: 75,
      sortBy: sessionSortBy,
      sortOrder: "desc",
    },
    {
      enabled: !isTracesMode && activeSearch.length === 0,
      placeholderData: keepPreviousData,
    },
  );
  const sessionSearchQuery = trpc.sessions.search.useQuery(
    {
      filters,
      limit: 75,
      query: activeSearch,
    },
    {
      enabled: !isTracesMode && activeSearch.length > 0,
      placeholderData: keepPreviousData,
    },
  );

  // Warm the other view mode so the traces/sessions toggle never jumps.
  useEffect(() => {
    if (isTracesMode) {
      void utils.sessions.facets.prefetch({ facetIds: TELEMETRY_FACET_IDS });
      void utils.sessions.list.prefetch({
        filters,
        limit: 75,
        sortBy: sessionSortBy,
        sortOrder: "desc",
      });
    } else {
      void utils.traces.facets.prefetch({ facetIds: TELEMETRY_FACET_IDS });
      void utils.traces.list.prefetch({
        filters,
        limit: 75,
        sortBy: traceSortBy,
        sortOrder: "desc",
      });
    }
  }, [filters, isTracesMode, sessionSortBy, traceSortBy, utils]);

  const traces = useMemo(
    () =>
      activeSearch
        ? (traceSearchQuery.data?.results.map((result) => result.trace) ?? [])
        : (traceListQuery.data?.traces ?? []),
    [activeSearch, traceListQuery.data?.traces, traceSearchQuery.data?.results],
  );
  const sessions = useMemo(
    () =>
      activeSearch
        ? (sessionSearchQuery.data?.results.map((result) => result.session) ?? [])
        : (sessionListQuery.data?.sessions ?? []),
    [activeSearch, sessionListQuery.data?.sessions, sessionSearchQuery.data?.results],
  );
  const traceTotalCount = activeSearch
    ? (traceSearchQuery.data?.totalCount ?? 0)
    : (traceListQuery.data?.totalCount ?? 0);
  const sessionTotalCount = activeSearch
    ? (sessionSearchQuery.data?.totalCount ?? 0)
    : (sessionListQuery.data?.totalCount ?? 0);
  const traceLoading =
    infoQuery.isLoading ||
    (activeSearch ? traceSearchQuery.isLoading : traceListQuery.isLoading);
  const sessionLoading =
    infoQuery.isLoading ||
    (activeSearch ? sessionSearchQuery.isLoading : sessionListQuery.isLoading);
  const isLoading = isTracesMode ? traceLoading : sessionLoading;
  const isRefreshing =
    infoQuery.isFetching ||
    (isTracesMode
      ? traceFacetsQuery.isFetching ||
        traceListQuery.isFetching ||
        traceSearchQuery.isFetching
      : sessionFacetsQuery.isFetching ||
        sessionListQuery.isFetching ||
        sessionSearchQuery.isFetching);

  const traceMetrics = useMemo(() => summarizeVisibleTraces(traces), [traces]);
  const sessionMetrics = useMemo(
    () => summarizeVisibleSessions(sessions),
    [sessions],
  );
  const activeFacets = isTracesMode
    ? traceFacetsQuery.data?.categorical
    : sessionFacetsQuery.data?.categorical;
  const ingestUrl = infoQuery.data?.ingestUrl ?? DEFAULT_INGEST_URL;
  const catalystEnvLine = `CATALYST_OTLP_ENDPOINT=${ingestUrl}`;
  const isTelemetryEmpty =
    Boolean(infoQuery.data) &&
    infoQuery.data?.traceCount === 0 &&
    infoQuery.data?.spanCount === 0;
  const latestVisibleTraceId = useMemo(
    () =>
      traces.reduce<Trace | undefined>(
        (latest, trace) =>
          !latest || trace.startTimeMs > latest.startTimeMs ? trace : latest,
        undefined,
      )?.traceId,
    [traces],
  );

  const refresh = () => {
    void infoQuery.refetch();
    if (isTracesMode) {
      void traceFacetsQuery.refetch();
      void (activeSearch ? traceSearchQuery.refetch() : traceListQuery.refetch());
      return;
    }
    void sessionFacetsQuery.refetch();
    void (activeSearch ? sessionSearchQuery.refetch() : sessionListQuery.refetch());
  };

  const handleFollowLatestChange = (enabled: boolean) => {
    if (!enabled) {
      onFollowLatestChange(false);
      return;
    }
    if (latestVisibleTraceId) {
      selectedTraceIdRef.current = latestVisibleTraceId;
      onSelectLatestTrace(latestVisibleTraceId);
      return;
    }
    onFollowLatestChange(true);
  };

  useEffect(() => {
    const onPageCommand = (
      event: WindowEventMap[typeof TRACE_PAGE_COMMAND_EVENT],
    ) => {
      switch (event.detail.type) {
        case "open-import":
          onOpenImportData();
          break;
        case "refresh":
          refresh();
          break;
        case "toggle-follow-latest":
          if (isTracesMode) {
            handleFollowLatestChange(!followLatestRef.current);
          }
          break;
      }
    };

    window.addEventListener(TRACE_PAGE_COMMAND_EVENT, onPageCommand);
    return () => {
      window.removeEventListener(TRACE_PAGE_COMMAND_EVENT, onPageCommand);
    };
  }, [handleFollowLatestChange, isTracesMode, onOpenImportData, refresh]);

  return (
    <main className="h-screen overflow-hidden bg-background text-foreground">
      <AppHeader
        status={
          <LiveStatusBadge
            health={infoQuery.data?.lastBatch?.status ?? "waiting"}
            liveStatus={liveStatus}
            liveUrl={infoQuery.data?.liveUrl ?? "ws://127.0.0.1:8800"}
          />
        }
        title="Trace Monitor"
        actions={
          <>
            {isTracesMode ? (
              <Button
                aria-label={
                  followLatest
                    ? "Stop following latest trace"
                    : "Follow latest trace"
                }
                aria-pressed={followLatest}
                className={cn(
                  "gap-2",
                  followLatest && "border-detail-brand/50 text-detail-brand",
                )}
                onClick={() => handleFollowLatestChange(!followLatest)}
                size="sm"
                variant={followLatest ? "secondary" : "outline"}
              >
                <Activity
                  className={cn("h-4 w-4", followLatest && "animate-pulse")}
                />
                {followLatest ? "Following latest" : "Follow latest"}
              </Button>
            ) : null}
            <Button aria-label="Open import data" asChild size="sm" variant="secondary">
              <Link onClick={onOpenImportData} to="/import-data">
                <DownloadCloud className="mr-2 h-4 w-4" />
                Import Data
              </Link>
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button aria-label="More actions" size="icon" variant="ghost">
                  <MoreHorizontal className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={refresh}>
                  <RefreshCcw
                    className={cn("mr-2 h-4 w-4", isRefreshing && "animate-spin")}
                  />
                  Refresh
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </>
        }
      />

      <div
        className={cn(
          "grid h-full min-h-0 pt-14",
          isTelemetryEmpty
            ? "grid-cols-[14rem_minmax(0,1fr)]"
            : "grid-cols-[14rem_300px_minmax(0,1fr)]",
        )}
      >
        <WorkspaceNav active="traces" />
        {isTelemetryEmpty ? null : (
          <FilterSidebar
            agentName={agentName}
            description="Switch views, then narrow local telemetry by runtime, model, and time."
            facets={activeFacets ?? {}}
            modelName={modelName}
            onAgentNameChange={setAgentName}
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
            onViewModeChange={onViewModeChange}
            scope={scope}
            serviceName={serviceName}
            source={source}
            status={status}
            onSourceChange={setSource}
            viewMode={viewMode}
          />
        )}

        <section className="min-h-0 min-w-0 overflow-hidden">
          {isTelemetryEmpty ? (
            <ImportDataScreen
              onConnectLocalAgent={() => setLocalAgentSetupOpen(true)}
              onImportJsonl={() => setFileDialogOpen(true)}
              onImportLangfuse={() => setImportDialogOpen(true)}
              onImportPhoenix={() => setPhoenixDialogOpen(true)}
            />
          ) : (
            <div className="flex h-full min-h-0 flex-col">
              <TelemetryStatStrip
                errorCount={
                  isTracesMode ? traceMetrics.errorCount : sessionMetrics.errorCount
                }
                isLoading={isLoading}
                llmSpanCount={
                  isTracesMode
                    ? traceMetrics.llmSpanCount
                    : sessionMetrics.llmSpanCount
                }
                mode={viewMode}
                sessionCount={sessionTotalCount}
                spanCount={
                  isTracesMode ? traceMetrics.spanCount : sessionMetrics.spanCount
                }
                totalCost={
                  isTracesMode ? traceMetrics.totalCost : sessionMetrics.totalCost
                }
                totalTokens={
                  isTracesMode
                    ? traceMetrics.totalTokens
                    : sessionMetrics.totalTokens
                }
                traceCount={
                  isTracesMode ? traceTotalCount : sessionMetrics.traceCount
                }
              />

              <div className="border-b border-subtle px-6 py-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="min-w-0">
                    <h1 className="text-2xl tracking-normal">
                      {isTracesMode ? "Local agent traces" : "Local agent sessions"}
                    </h1>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {isTracesMode
                        ? infoQuery.data?.lastBatch
                          ? `Last ingest ${relativeTime(infoQuery.data.lastBatch.receivedAt)} with ${infoQuery.data.lastBatch.acceptedSpanCount} spans`
                          : "Waiting for your first OTLP trace batch"
                        : "Conversations grouped by session ID across all turns."}
                    </p>
                  </div>
                  <div className="flex min-w-0 items-center gap-2">
                    <Input
                      aria-label={isTracesMode ? "Search traces" : "Search sessions"}
                      className="h-9"
                      containerClassname="w-72"
                      icon={<Search className="h-4 w-4 text-muted-foreground" />}
                      onChange={(event) => setSearchText(event.currentTarget.value)}
                      placeholder={
                        isTracesMode
                          ? "Search trace IDs, models, inputs..."
                          : "Search sessions, services, models..."
                      }
                      value={searchText}
                    />
                    {isTracesMode ? (
                      <FilterSelect
                        ariaLabel="Sort traces"
                        icon={<ArrowUpDown className="h-3.5 w-3.5" />}
                        onChange={(value) => setTraceSortBy(value as TraceSortKey)}
                        options={[
                          { label: "Newest", value: "start_time" },
                          { label: "Duration", value: "duration" },
                          { label: "Span count", value: "span_count" },
                          { label: "LLM spans", value: "llm_span_count" },
                          { label: "Tokens", value: "total_tokens" },
                          { label: "Cost", value: "total_cost" },
                        ]}
                        triggerClassName="h-9 w-40"
                        value={traceSortBy}
                      />
                    ) : (
                      <FilterSelect
                        ariaLabel="Sort sessions"
                        icon={<ArrowUpDown className="h-3.5 w-3.5" />}
                        onChange={(value) =>
                          setSessionSortBy(value as SessionSortKey)
                        }
                        options={[
                          { label: "Latest activity", value: "last_activity" },
                          { label: "First activity", value: "start_time" },
                          { label: "Duration", value: "duration" },
                          { label: "Turns", value: "trace_count" },
                          { label: "Spans", value: "span_count" },
                          { label: "LLM spans", value: "llm_span_count" },
                          { label: "Tokens", value: "total_tokens" },
                          { label: "Cost", value: "total_cost" },
                        ]}
                        triggerClassName="h-9 w-44"
                        value={sessionSortBy}
                      />
                    )}
                    <LiveRangeControl
                      dateRange={dateRange}
                      liveStatus={liveStatus}
                      onDateRangeChange={setDateRange}
                    />
                  </div>
                </div>
              </div>

              {isTracesMode ? (
                <TraceList
                  activeTraceId={selectedTraceId}
                  isLoading={isLoading}
                  onSelectTrace={onSelectTrace}
                  recentTraceIds={recentTraceIds}
                  totalCount={traceTotalCount}
                  traces={traces}
                />
              ) : (
                <SessionList
                  activeSessionId={selectedSessionId}
                  isLoading={isLoading}
                  onSelectSession={onSelectSession}
                  recentSessionIds={recentSessionIds}
                  sessions={sessions}
                  totalCount={sessionTotalCount}
                />
              )}
            </div>
          )}
        </section>
      </div>

      <TelemetryDetailSheet
        followLatest={isTracesMode ? followLatest : false}
        mode="trace"
        onOpenChange={(open) => {
          if (!open) onSelectTrace(null);
        }}
        open={isTracesMode && (followLatest || Boolean(selectedTraceId))}
        traceId={selectedTraceId}
      />
      <TelemetryDetailSheet
        mode="session"
        onOpenChange={(open) => {
          if (!open) onSelectSession(null);
        }}
        open={!isTracesMode && Boolean(selectedSessionId)}
        sessionId={selectedSessionId}
      />
      <LangfuseImportDialog
        onImported={refresh}
        onOpenChange={setImportDialogOpen}
        open={importDialogOpen}
      />
      <PhoenixImportDialog
        onImported={refresh}
        onOpenChange={setPhoenixDialogOpen}
        open={phoenixDialogOpen}
      />
      <FileImportDialog
        onImported={refresh}
        onOpenChange={setFileDialogOpen}
        open={fileDialogOpen}
      />
      <LocalAgentSetupDialog
        envLine={catalystEnvLine}
        ingestUrl={ingestUrl}
        onOpenChange={setLocalAgentSetupOpen}
        open={localAgentSetupOpen}
      />
    </main>
  );
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
