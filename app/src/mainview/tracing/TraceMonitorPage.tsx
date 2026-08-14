import {
  useDeferredValue,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { keepPreviousData } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import {
  DownloadCloud,
  Play,
} from "lucide-react";

import {
  Button,
  cn,
} from "~/lib/ui";
import { trpc } from "~/trpc";
import { WorkspaceNav } from "~/workspace/WorkspaceNav";
import {
  openExternalUrl,
  TRACE_PAGE_COMMAND_EVENT,
} from "~/desktop/desktopBridge";
import { AppHeader } from "~/components/AppHeader";
import { startDateForRange, type DateRange } from "~/lib/format";
import { ImportDataScreen, LocalAgentSetupDialog } from "./ImportDataScreen";
import { LangfuseImportDialog } from "./langfuse/LangfuseImportDialog";
import { PhoenixImportDialog } from "./phoenix/PhoenixImportDialog";
import { FileImportDialog } from "./fileimport/FileImportDialog";
import { DemoTracesImportDialog } from "./DemoTracesImportDialog";
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
import { RunConfigDialog, type RunConfigInitialValues } from "~/halo/RunConfigDialog";
import type { LogSortOrder } from "./logTable";
import { SessionList } from "./SessionList";
import { TelemetryStatStrip } from "./TelemetryStatStrip";
import { TraceList } from "./TraceList";
import { TelemetryDetailSheet } from "./detail/TelemetryDetailSheet";
import { APP_DOCS_URL } from "../../desktop/commands";
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
  const navigate = useNavigate();
  const isTracesMode = viewMode === "traces";
  const [searchText, setSearchText] = useState("");
  const [dateRange, setDateRange] = useState<DateRange>("all");
  const [status, setStatus] = useState<StatusFilter>("all");
  const [scope, setScope] = useState<ScopeFilter>("all");
  const [serviceName, setServiceName] = useState("all");
  const [agentName, setAgentName] = useState("all");
  const [modelName, setModelName] = useState("all");
  const [source, setSource] = useState<SourceFilter>("all");
  const [traceSortBy, setTraceSortBy] = useState<TraceSortKey>("start_time");
  const [traceSortOrder, setTraceSortOrder] =
    useState<LogSortOrder>("desc");
  const [sessionSortBy, setSessionSortBy] =
    useState<SessionSortKey>("start_time");
  const [sessionSortOrder, setSessionSortOrder] =
    useState<LogSortOrder>("desc");
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [phoenixDialogOpen, setPhoenixDialogOpen] = useState(false);
  const [fileDialogOpen, setFileDialogOpen] = useState(false);
  const [demoDialogOpen, setDemoDialogOpen] = useState(false);
  const [localAgentSetupOpen, setLocalAgentSetupOpen] = useState(false);
  const [runConfigOpen, setRunConfigOpen] = useState(false);
  const [runConfigInitialValues, setRunConfigInitialValues] = useState<
    RunConfigInitialValues | undefined
  >(undefined);
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
    onData(eventEnvelope) {
      const event = eventEnvelope.data;
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
  const traceListQuery = trpc.traces.list.useInfiniteQuery(
    {
      filters,
      limit: 75,
      sortBy: traceSortBy,
      sortOrder: traceSortOrder,
    },
    {
      enabled: isTracesMode && activeSearch.length === 0,
      getNextPageParam: (lastPage) => lastPage.nextCursor ?? undefined,
      placeholderData: keepPreviousData,
    },
  );
  const traceSearchQuery = trpc.traces.search.useInfiniteQuery(
    {
      filters,
      limit: 75,
      query: activeSearch,
      sortBy: traceSortBy,
      sortOrder: traceSortOrder,
    },
    {
      enabled: isTracesMode && activeSearch.length > 0,
      getNextPageParam: (lastPage) => lastPage.nextCursor ?? undefined,
      placeholderData: keepPreviousData,
    },
  );
  const sessionFacetsQuery = trpc.sessions.facets.useQuery(
    { facetIds: TELEMETRY_FACET_IDS },
    { enabled: !isTracesMode, placeholderData: keepPreviousData },
  );
  const sessionListQuery = trpc.sessions.list.useInfiniteQuery(
    {
      filters,
      limit: 75,
      sortBy: sessionSortBy,
      sortOrder: sessionSortOrder,
    },
    {
      enabled: !isTracesMode && activeSearch.length === 0,
      getNextPageParam: (lastPage) => lastPage.nextCursor ?? undefined,
      placeholderData: keepPreviousData,
    },
  );
  const sessionSearchQuery = trpc.sessions.search.useInfiniteQuery(
    {
      filters,
      limit: 75,
      query: activeSearch,
      sortBy: sessionSortBy,
      sortOrder: sessionSortOrder,
    },
    {
      enabled: !isTracesMode && activeSearch.length > 0,
      getNextPageParam: (lastPage) => lastPage.nextCursor ?? undefined,
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
        sortOrder: sessionSortOrder,
      });
    } else {
      void utils.traces.facets.prefetch({ facetIds: TELEMETRY_FACET_IDS });
      void utils.traces.list.prefetch({
        filters,
        limit: 75,
        sortBy: traceSortBy,
        sortOrder: traceSortOrder,
      });
    }
  }, [
    filters,
    isTracesMode,
    sessionSortBy,
    sessionSortOrder,
    traceSortBy,
    traceSortOrder,
    utils,
  ]);

  const traces = useMemo(
    () =>
      activeSearch
        ? (traceSearchQuery.data?.pages.flatMap((page) =>
            page.results.map((result) => result.trace),
          ) ?? [])
        : (traceListQuery.data?.pages.flatMap((page) => page.traces) ?? []),
    [activeSearch, traceListQuery.data?.pages, traceSearchQuery.data?.pages],
  );
  const sessions = useMemo(
    () =>
      activeSearch
        ? (sessionSearchQuery.data?.pages.flatMap((page) =>
            page.results.map((result) => result.session),
          ) ?? [])
        : (sessionListQuery.data?.pages.flatMap((page) => page.sessions) ?? []),
    [activeSearch, sessionListQuery.data?.pages, sessionSearchQuery.data?.pages],
  );
  const traceTotalCount = activeSearch
    ? (traceSearchQuery.data?.pages[0]?.totalCount ?? 0)
    : (traceListQuery.data?.pages[0]?.totalCount ?? 0);
  const sessionTotalCount = activeSearch
    ? (sessionSearchQuery.data?.pages[0]?.totalCount ?? 0)
    : (sessionListQuery.data?.pages[0]?.totalCount ?? 0);
  const traceLoading =
    infoQuery.isLoading ||
    (activeSearch ? traceSearchQuery.isLoading : traceListQuery.isLoading);
  const sessionLoading =
    infoQuery.isLoading ||
    (activeSearch ? sessionSearchQuery.isLoading : sessionListQuery.isLoading);
  const isLoading = isTracesMode ? traceLoading : sessionLoading;
  const hasNextTracePage = activeSearch
    ? Boolean(traceSearchQuery.hasNextPage)
    : Boolean(traceListQuery.hasNextPage);
  const hasNextSessionPage = activeSearch
    ? Boolean(sessionSearchQuery.hasNextPage)
    : Boolean(sessionListQuery.hasNextPage);
  const isFetchingNextTracePage = activeSearch
    ? traceSearchQuery.isFetchingNextPage
    : traceListQuery.isFetchingNextPage;
  const isFetchingNextSessionPage = activeSearch
    ? sessionSearchQuery.isFetchingNextPage
    : sessionListQuery.isFetchingNextPage;
  const fetchNextTracePage = useCallback(() => {
    if (activeSearch) {
      if (traceSearchQuery.hasNextPage && !traceSearchQuery.isFetchingNextPage) {
        void traceSearchQuery.fetchNextPage();
      }
      return;
    }
    if (traceListQuery.hasNextPage && !traceListQuery.isFetchingNextPage) {
      void traceListQuery.fetchNextPage();
    }
  }, [activeSearch, traceListQuery, traceSearchQuery]);
  const fetchNextSessionPage = useCallback(() => {
    if (activeSearch) {
      if (
        sessionSearchQuery.hasNextPage &&
        !sessionSearchQuery.isFetchingNextPage
      ) {
        void sessionSearchQuery.fetchNextPage();
      }
      return;
    }
    if (sessionListQuery.hasNextPage && !sessionListQuery.isFetchingNextPage) {
      void sessionListQuery.fetchNextPage();
    }
  }, [activeSearch, sessionListQuery, sessionSearchQuery]);

  const traceMetrics = useMemo(() => summarizeVisibleTraces(traces), [traces]);
  const sessionMetrics = useMemo(
    () => summarizeVisibleSessions(sessions),
    [sessions],
  );
  const activeFacets = isTracesMode
    ? traceFacetsQuery.data?.categorical
    : sessionFacetsQuery.data?.categorical;
  const ingestUrl = infoQuery.data?.ingestUrl ?? DEFAULT_INGEST_URL;
  const inferenceEnvLine = `INFERENCE_OTLP_ENDPOINT=${ingestUrl}`;
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
  const handleReadDocumentation = useCallback(() => {
    void openExternalUrl(APP_DOCS_URL);
  }, []);

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

  const handleTraceSortChange = useCallback(
    (sortBy: TraceSortKey, sortOrder: LogSortOrder) => {
      setTraceSortBy(sortBy);
      setTraceSortOrder(sortOrder);
    },
    [],
  );

  const handleSessionSortChange = useCallback(
    (sortBy: SessionSortKey, sortOrder: LogSortOrder) => {
      setSessionSortBy(sortBy);
      setSessionSortOrder(sortOrder);
    },
    [],
  );

  const openRunAnalysis = useCallback(() => {
    setRunConfigInitialValues({
      dateRange,
      filters: {
        ...filters,
        freeText: activeSearch || undefined,
      },
      targetType: isTracesMode ? "trace_group" : "session_group",
    });
    setRunConfigOpen(true);
  }, [activeSearch, dateRange, filters, isTracesMode]);

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
        title="Trace Monitor"
        actions={
          <>
            <Button aria-label="Open import data" asChild size="sm" variant="secondary">
              <Link onClick={onOpenImportData} to="/import-data">
                <DownloadCloud className="mr-2 h-4 w-4" />
                Import Data
              </Link>
            </Button>
            <Button
              aria-label="Run analysis on current filters"
              onClick={openRunAnalysis}
              size="sm"
              variant="default"
            >
              <Play className="mr-2 h-4 w-4" />
              Run Analysis
            </Button>
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
        <WorkspaceNav active="data" />
        {isTelemetryEmpty ? null : (
          <FilterSidebar
            agentName={agentName}
            dateRange={dateRange}
            description="Search, filter, and sort sessions and traces"
            facets={activeFacets ?? {}}
            modelName={modelName}
            onAgentNameChange={setAgentName}
            onDateRangeChange={setDateRange}
            onModelNameChange={setModelName}
            onReset={() => {
              setSearchText("");
              setDateRange("all");
              setStatus("all");
              setScope("all");
              setServiceName("all");
              setAgentName("all");
              setModelName("all");
              setSource("all");
            }}
            onScopeChange={setScope}
            onSearchTextChange={setSearchText}
            onServiceNameChange={setServiceName}
            onStatusChange={setStatus}
            onViewModeChange={onViewModeChange}
            scope={scope}
            searchText={searchText}
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
              onLoadDemoTraces={() => setDemoDialogOpen(true)}
              onReadDocumentation={handleReadDocumentation}
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

              {isTracesMode ? (
                <TraceList
                  activeTraceId={selectedTraceId}
                  hasNextPage={hasNextTracePage}
                  isLoading={isLoading}
                  isFetchingNextPage={isFetchingNextTracePage}
                  onLoadMore={fetchNextTracePage}
                  onSortChange={handleTraceSortChange}
                  onSelectTrace={onSelectTrace}
                  recentTraceIds={recentTraceIds}
                  sortBy={traceSortBy}
                  sortOrder={traceSortOrder}
                  totalCount={traceTotalCount}
                  traces={traces}
                />
              ) : (
                <SessionList
                  activeSessionId={selectedSessionId}
                  hasNextPage={hasNextSessionPage}
                  isLoading={isLoading}
                  isFetchingNextPage={isFetchingNextSessionPage}
                  onLoadMore={fetchNextSessionPage}
                  onSortChange={handleSessionSortChange}
                  onSelectSession={onSelectSession}
                  recentSessionIds={recentSessionIds}
                  sessions={sessions}
                  sortBy={sessionSortBy}
                  sortOrder={sessionSortOrder}
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
      <DemoTracesImportDialog
        onImported={refresh}
        onOpenChange={setDemoDialogOpen}
        open={demoDialogOpen}
      />
      <LocalAgentSetupDialog
        envLine={inferenceEnvLine}
        ingestUrl={ingestUrl}
        onOpenChange={setLocalAgentSetupOpen}
        open={localAgentSetupOpen}
      />
      <RunConfigDialog
        initialValues={runConfigInitialValues}
        onOpenChange={setRunConfigOpen}
        onStarted={(run) => {
          void navigate({ params: { runId: run.id }, to: "/analysis/$runId" });
        }}
        open={runConfigOpen}
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
