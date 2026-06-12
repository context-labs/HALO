import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Activity, Loader2 } from "lucide-react";

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "~/lib/ui";
import { trpc } from "~/trpc";
import type { Span, SpanNode } from "../../../server/telemetry/types";
import {
  buildClientSpanTree,
  buildSessionSpanTree,
  isSessionTraceGroupSpan,
  isSyntheticSpan,
} from "../spanTree";
import { ConversationView } from "./ConversationView";
import { SpanDetailPanel } from "./SpanDetailPanel";
import { TimelineView } from "./TimelineView";
import { timelineDomain } from "./timelineMath";
import {
  TraceDetailHeader,
  type TraceDetailStatus,
  type TraceDetailViewMode,
} from "./TraceDetailHeader";
import { maxLlmCost } from "./spanKinds";
import { spanKey, upsertSpan } from "./spanUtils";
import type { WaterfallHandle } from "./WaterfallCanvas";

const EMPTY_SPANS: Span[] = [];
const VIEW_MODE_STORAGE_KEY = "halo.traceViewer.view";

export function TelemetryDetailSheet({
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
  const [selectedSpanKey, setSelectedSpanKey] = useState<string | null>(null);
  const [expandedSpanKeys, setExpandedSpanKeys] = useState<Set<string>>(
    () => new Set(),
  );
  const [heat, setHeat] = useState(false);
  const [viewMode, setViewModeRaw] = useState<TraceDetailViewMode>(() => {
    try {
      const stored = localStorage.getItem(VIEW_MODE_STORAGE_KEY);
      return stored === "conversation" ? "conversation" : "timeline";
    } catch {
      return "timeline";
    }
  });
  const setViewMode = useCallback((value: TraceDetailViewMode) => {
    setViewModeRaw(value);
    try {
      localStorage.setItem(VIEW_MODE_STORAGE_KEY, value);
    } catch {
      // best effort
    }
  }, []);
  const [recentSpanIds, setRecentSpanIds] = useState<Set<string>>(() => new Set());
  const recentSpanTimers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const expansionInitializedFor = useRef<string | null>(null);
  const waterfallRef = useRef<WaterfallHandle | null>(null);
  const sheetBodyRef = useRef<HTMLDivElement | null>(null);
  const [narrow, setNarrow] = useState(false);
  const [nowMs, setNowMs] = useState(() => Date.now());

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
    expansionInitializedFor.current = null;
  }, [sessionId, traceId]);

  useEffect(() => {
    if (!open) {
      setSelectedSpanKey(null);
      setRecentSpanIds(new Set());
    }
  }, [open]);

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
  const nodeByKey = useMemo(() => {
    const map = new Map<string, SpanNode>();
    const visit = (node: SpanNode) => {
      map.set(spanKey(node.span), node);
      node.children.forEach(visit);
    };
    displayTree.forEach(visit);
    return map;
  }, [displayTree]);

  // Expand every parent once per opened item; auto-expand parents that stream
  // in later, preserving the user's manual collapses otherwise.
  const knownParentKeys = useRef<Set<string>>(new Set());
  useEffect(() => {
    const id = mode === "session" ? sessionId : traceId;
    if (!id || nodeByKey.size === 0) return;
    const parentKeys = new Set<string>();
    for (const [key, node] of nodeByKey) {
      if (node.children.length > 0) parentKeys.add(key);
    }
    if (expansionInitializedFor.current !== id) {
      expansionInitializedFor.current = id;
      knownParentKeys.current = parentKeys;
      setExpandedSpanKeys(parentKeys);
      return;
    }
    const fresh = [...parentKeys].filter(
      (key) => !knownParentKeys.current.has(key),
    );
    if (fresh.length > 0) {
      knownParentKeys.current = parentKeys;
      setExpandedSpanKeys((current) => {
        const next = new Set(current);
        fresh.forEach((key) => next.add(key));
        return next;
      });
    }
  }, [mode, nodeByKey, sessionId, traceId]);

  const onToggleExpanded = useCallback((key: string) => {
    setExpandedSpanKeys((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const realSpans = useMemo(
    () => spans.filter((span) => !isSyntheticSpan(span)),
    [spans],
  );
  const maxCost = useMemo(() => maxLlmCost(realSpans), [realSpans]);

  const selectedSpanCandidate = selectedSpanKey
    ? (nodeByKey.get(selectedSpanKey) ?? null)
    : null;
  const selectedNode =
    selectedSpanCandidate && !isSessionTraceGroupSpan(selectedSpanCandidate.span)
      ? selectedSpanCandidate
      : null;
  const selectedSpan = selectedNode?.span ?? null;

  const session = sessionQuery.data ?? null;
  const trace = mode === "trace" ? (traceQuery.data ?? null) : null;
  const waitingForLatest = mode === "trace" && followLatest && !traceId;

  // Status: live spans still open → running; any error → failed; else done.
  const hasOpenSpan = realSpans.some(
    (span) => span.endTimeMs <= span.startTimeMs,
  );
  const running = recentSpanIds.size > 0 || (realSpans.length > 0 && hasOpenSpan);
  const hasError =
    mode === "session" ? Boolean(session?.hasError) : Boolean(trace?.hasError);
  const status: TraceDetailStatus = running
    ? "running"
    : hasError
      ? "failed"
      : "completed";

  // Tick a shared clock while running so in-flight bars/durations advance.
  useEffect(() => {
    if (!open || !running) return;
    const timer = setInterval(() => setNowMs(Date.now()), 500);
    setNowMs(Date.now());
    return () => clearInterval(timer);
  }, [open, running]);

  // Collapse header stats when the sheet is narrow (matches the prototype).
  useLayoutEffect(() => {
    const el = sheetBodyRef.current;
    if (!el || !open) return;
    const observer = new ResizeObserver(() =>
      setNarrow(el.clientWidth < 1080),
    );
    observer.observe(el);
    setNarrow(el.clientWidth < 1080);
    return () => observer.disconnect();
  }, [open]);

  const title = waitingForLatest
    ? "Waiting for next trace…"
    : mode === "session"
      ? (session?.latestTraceName || "Session detail")
      : (trace?.rootSpanName || displayTree[0]?.span.spanName || "Trace detail");
  const description = mode === "session" ? (sessionId ?? null) : (traceId ?? null);
  const startedAt =
    mode === "session" ? (session?.startTime ?? null) : (trace?.startTime ?? null);
  const durationMs =
    mode === "session"
      ? (session?.durationMs ?? null)
      : (trace?.durationMs ?? null);
  const tokens =
    mode === "session"
      ? (session?.totalTokens ?? null)
      : (trace?.totalTokens ?? null);
  const costTotal =
    mode === "session"
      ? nullableNumber(session?.totalCost)
      : nullableNumber(trace?.totalCost);
  const liveDurationMs =
    running && startedAt
      ? Math.max(0, nowMs - Date.parse(startedAt))
      : durationMs;

  const loading =
    mode === "session"
      ? sessionQuery.isLoading || sessionSpansQuery.isLoading
      : spansQuery.isLoading;

  return (
    <Sheet onOpenChange={onOpenChange} open={open}>
      <SheetContent
        className="flex w-[88vw] max-w-[88vw] flex-col overflow-hidden p-0 max-md:w-[95vw] max-md:max-w-[95vw] sm:max-w-[88vw] [&>button]:top-3"
        onEscapeKeyDown={(event) => {
          if (selectedSpanKey) {
            event.preventDefault();
            setSelectedSpanKey(null);
          }
        }}
        side="right"
      >
        <SheetHeader className="sr-only">
          <SheetTitle>{title}</SheetTitle>
          <SheetDescription>{description ?? "Trace detail"}</SheetDescription>
        </SheetHeader>

        <div className="relative flex min-h-0 flex-1 flex-col" ref={sheetBodyRef}>
          <TraceDetailHeader
            costTotal={costTotal}
            description={description}
            durationMs={liveDurationMs}
            followingLatest={mode === "trace" && Boolean(followLatest)}
            heat={heat}
            narrow={narrow}
            onHeatChange={setHeat}
            onViewModeChange={setViewMode}
            onZoomFit={() => waterfallRef.current?.zoomToFit()}
            onZoomIn={() => waterfallRef.current?.zoomIn()}
            onZoomOut={() => waterfallRef.current?.zoomOut()}
            spanCount={realSpans.length}
            startedAt={startedAt}
            status={status}
            title={title}
            tokens={tokens}
            viewMode={viewMode}
          />

          {waitingForLatest ? (
            <WaitingForLatestTrace />
          ) : loading ? (
            <div className="grid flex-1 place-items-center">
              <Loader2 className="h-7 w-7 animate-spin text-muted-foreground" />
            </div>
          ) : viewMode === "conversation" ? (
            <ConversationView
              heat={heat}
              maxCost={maxCost}
              mode={mode}
              onSelect={setSelectedSpanKey}
              selectedSpanKey={selectedSpanKey}
              spans={spans}
              tree={displayTree}
            />
          ) : (
            <TimelineView
              expanded={expandedSpanKeys}
              heat={heat}
              maxCost={maxCost}
              nodeByKey={nodeByKey}
              nowMs={nowMs}
              onSelect={setSelectedSpanKey}
              onToggle={onToggleExpanded}
              recentSpanIds={recentSpanIds}
              ref={waterfallRef}
              running={running}
              selectedSpanKey={selectedSpanKey}
              tree={displayTree}
            />
          )}

          {/* Span details pop over the views as a nested, resizable sheet. */}
          {selectedSpan && !waitingForLatest && !loading ? (
            <SpanDetailPanel
              domainStartMs={timelineDomain(realSpans, nowMs).startMs}
              heat={heat}
              maxCost={maxCost}
              node={selectedNode}
              onClose={() => setSelectedSpanKey(null)}
              span={selectedSpan}
            />
          ) : null}
        </div>
      </SheetContent>
    </Sheet>
  );
}

function WaitingForLatestTrace() {
  return (
    <div className="grid flex-1 place-items-center p-8">
      <div className="max-w-md rounded-xl border border-dashed border-subtle bg-background-muted p-8 text-center">
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

function nullableNumber(value: string | null | undefined): number | null {
  if (value == null) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}
