import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import {
  Activity,
  ArrowLeft,
  ArrowUp,
  BrainCircuit,
  Clipboard,
  Copy,
  FileBox,
  Filter,
  Info,
  Loader2,
  MoreHorizontal,
  RotateCcw,
  SquareArrowOutUpRight,
  Trash2,
} from "lucide-react";

import {
  Badge,
  Button,
  Dialog,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  EmptyState,
  Textarea,
  cn,
  toast,
} from "~/lib/ui";
import { trpc } from "~/trpc";
import { WorkspaceNav } from "~/workspace/WorkspaceNav";
import { AppHeader } from "~/components/AppHeader";
import { ProgressBar, StatusBadge } from "~/components/StatusBadge";
import { formatTimestamp } from "~/lib/format";
import type { TelemetryFilters } from "../../server/telemetry/types";
import { OpenInToolBar } from "./OpenInToolBar";
import { RunConfigDialog, type RunConfigInitialValues } from "./RunConfigDialog";
import { RunConversation } from "./RunConversation";
import { RunPhaseTimeline } from "./RunPhaseTimeline";
import { isActiveRun } from "./RunsTable";
import { targetLabel, type HaloRunView } from "./runShared";

const RAIL_MIN_WIDTH = 280;
const RAIL_MAX_WIDTH = 560;
const RAIL_DEFAULT_WIDTH = 340;
const RAIL_WIDTH_STORAGE_KEY = "halo-run-rail-width";

export function RunDetailPage({ runId }: { runId: string }) {
  const navigate = useNavigate();
  const utils = trpc.useUtils();
  const [message, setMessage] = useState("");
  const [configOpen, setConfigOpen] = useState(false);
  const [configInitialValues, setConfigInitialValues] = useState<
    RunConfigInitialValues | undefined
  >(undefined);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollPaneRef = useRef<HTMLDivElement>(null);
  const nearBottomRef = useRef(true);
  const didInitialScrollRef = useRef(false);
  const rail = useResizableRail();

  // Streamed text per assistant turn, accumulated from live delta events.
  // Persisted turn content takes over once a turn completes.
  const [streamText, setStreamText] = useState<Record<number, string>>({});

  const runQuery = trpc.halo.runs.get.useQuery({ runId });
  const turnsQuery = trpc.halo.runs.getTurns.useQuery({ runId });
  // Deltas are intentionally excluded — they arrive via the live subscription
  // and would blow through the row limit on chatty runs, truncating steps.
  const eventsInput = useMemo(
    () => ({ eventTypes: ["agent_step"], limit: 1000, runId }),
    [runId],
  );
  const eventsQuery = trpc.halo.runs.getEvents.useQuery(eventsInput);

  trpc.live.haloRun.useSubscription(
    { runId },
    {
      onData(eventEnvelope) {
        const payload = eventEnvelope.data.payload;
        if (
          payload.type !== "halo.run.updated" &&
          payload.type !== "halo.run.event" &&
          payload.type !== "halo.run.completed" &&
          payload.type !== "halo.run.failed"
        ) {
          return;
        }
        utils.halo.runs.get.setData({ runId }, (current) =>
          current ? { ...current, ...payload.run } : current,
        );
        if (payload.type !== "halo.run.updated") {
          const runEvent = payload.event;
          if (runEvent.eventType === "agent_step") {
            utils.halo.runs.getEvents.setData(eventsInput, (current) => {
              if (!current) return current;
              if (current.some((item) => item.id === runEvent.id)) return current;
              return [...current, runEvent];
            });
          }
          if (runEvent.eventType === "delta") {
            const turnIndex = runEvent.turnIndex ?? 1;
            const text = String(runEvent.payload.text_delta ?? "");
            if (text) {
              setStreamText((current) => ({
                ...current,
                [turnIndex]: (current[turnIndex] ?? "") + text,
              }));
            }
          }
        }
        const terminal =
          payload.type === "halo.run.completed" ||
          payload.type === "halo.run.failed";
        if (terminal || payload.type === "halo.run.updated") {
          // Turn rows change on queue/stream/terminal transitions.
          void utils.halo.runs.getTurns.invalidate({ runId });
        }
        if (terminal) {
          setStreamText({});
          void utils.halo.runs.get.invalidate({ runId });
          void utils.halo.runs.getEvents.invalidate(eventsInput);
          void utils.halo.runs.list.invalidate();
        }
      },
    },
  );

  const continueMutation = trpc.halo.runs.continue.useMutation({
    async onSuccess() {
      setMessage("");
      await Promise.all([
        utils.halo.runs.get.invalidate({ runId }),
        utils.halo.runs.getTurns.invalidate({ runId }),
      ]);
    },
    onError(error) {
      toast.error({ title: "Could not send follow-up", description: error.message });
    },
  });
  const retryMutation = trpc.halo.runs.retry.useMutation({
    async onSuccess() {
      await Promise.all([
        utils.halo.runs.get.invalidate({ runId }),
        utils.halo.runs.getTurns.invalidate({ runId }),
      ]);
    },
  });
  const deleteMutation = trpc.halo.runs.delete.useMutation({
    async onSuccess() {
      toast.success({ title: "HALO run deleted" });
      await utils.halo.runs.list.invalidate();
      void navigate({ to: "/analysis" });
    },
    onError(error) {
      toast.error({ title: "Could not delete run", description: error.message });
    },
  });

  const run = runQuery.data ?? null;
  const turns = turnsQuery.data ?? [];
  const events = useMemo(() => eventsQuery.data ?? [], [eventsQuery.data]);
  const active = run ? isActiveRun(run) : false;

  const scrollPaneToBottom = () => {
    const pane = scrollPaneRef.current;
    if (!pane) return;
    // Scroll the pane itself — scrollIntoView would tuck content under the
    // sticky composer.
    pane.scrollTop = pane.scrollHeight;
  };

  // Open at the latest exchange, like any chat.
  useEffect(() => {
    if (didInitialScrollRef.current || turns.length === 0) return;
    didInitialScrollRef.current = true;
    scrollPaneToBottom();
  }, [turns.length]);

  // Stay pinned to the bottom while HALO streams — unless the user has
  // scrolled up to read something.
  const streamLength = useMemo(
    () => Object.values(streamText).reduce((total, text) => total + text.length, 0),
    [streamText],
  );
  const streamSignature = `${turns.length}:${events.length}:${streamLength}:${run?.status ?? ""}`;
  useEffect(() => {
    if (active && nearBottomRef.current) {
      scrollPaneToBottom();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, streamSignature]);

  const trackScrollPosition = () => {
    const pane = scrollPaneRef.current;
    if (!pane) return;
    nearBottomRef.current =
      pane.scrollHeight - pane.scrollTop - pane.clientHeight < 120;
  };

  const sendFollowUp = () => {
    const trimmed = message.trim();
    if (!trimmed || active || continueMutation.isPending) return;
    continueMutation.mutate({ message: trimmed, runId });
  };

  const copyRunLink = async () => {
    await navigator.clipboard.writeText(`#/analysis/${runId}`);
    toast.success({ title: "Run link copied" });
  };

  if (runQuery.isLoading) {
    return (
      <Shell title="HALO run">
        <div className="grid min-h-60 flex-1 place-items-center">
          <Loader2 className="h-7 w-7 animate-spin text-muted-foreground" />
        </div>
      </Shell>
    );
  }

  if (!run) {
    return (
      <Shell title="HALO run">
        <div className="mx-auto w-full max-w-md py-16">
          <EmptyState
            action={
              <Button asChild size="sm" variant="outline">
                <Link to="/analysis">Back to Analysis</Link>
              </Button>
            }
            description="It may have been deleted, or the link is stale."
            icon={BrainCircuit}
            title="Run not found"
          />
        </div>
      </Shell>
    );
  }

  return (
    <Shell
      actions={
        <>
          <Button
            onClick={() => {
              setConfigInitialValues({
                filters: (run.filters ?? {}) as TelemetryFilters,
                filtersSourceTitle: run.title,
                maxDepth: run.maxDepth,
                maxParallel: run.maxParallel,
                maxTurns: run.maxTurns,
                prompt: run.prompt,
                providerId: run.providerId ?? undefined,
                targetType: run.targetType,
                title: run.title.endsWith("(re-run)")
                  ? run.title
                  : `${run.title} (re-run)`,
              });
              setConfigOpen(true);
            }}
            size="sm"
            variant="outline"
          >
            <RotateCcw className="mr-2 h-3.5 w-3.5" />
            Re-run with changes
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button aria-label="More actions" size="icon" variant="ghost">
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => void copyRunLink()}>
                <Clipboard className="mr-2 h-4 w-4" />
                Copy run link
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                className="text-destructive focus:text-destructive"
                onClick={() => setDeleteOpen(true)}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                Delete run…
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </>
      }
      title={run.title}
    >
      <div className="flex h-full min-w-0">
        {/* Conversation pane — its own scroll context. */}
        <div
          className="flex min-w-0 flex-1 flex-col overflow-y-auto"
          onScroll={trackScrollPosition}
          ref={scrollPaneRef}
        >
          <div className="mx-auto w-full max-w-3xl flex-1 px-8 pb-6">
            <div className="py-4">
              <Link
                className="inline-flex items-center gap-1.5 text-sm text-muted-foreground transition hover:text-foreground"
                to="/analysis"
              >
                <ArrowLeft className="h-3.5 w-3.5" />
                Analysis
              </Link>
            </div>

            {turnsQuery.isLoading ? (
              <div className="grid min-h-40 place-items-center">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : (
              <RunConversation
                events={events}
                onRetry={() => retryMutation.mutate({ runId })}
                run={run}
                streamText={streamText}
                turns={turns}
              />
            )}
            <div ref={bottomRef} />
          </div>

          <div className="sticky bottom-0 z-10">
            {/* Soft blur + fade so scrolling content dissolves instead of clipping. */}
            <div className="pointer-events-none absolute inset-x-0 -top-12 bottom-0 backdrop-blur-[2px] [mask-image:linear-gradient(to_top,black_60%,transparent)]" />
            <div className="pointer-events-none absolute inset-x-0 -top-12 bottom-0 bg-gradient-to-t from-background via-background/85 to-transparent" />
            <div className="relative mx-auto w-full max-w-3xl px-8 pb-4">
              <div className="rounded-xl border border-subtle bg-card shadow-sm focus-within:border-foreground/30">
                <Textarea
                  aria-label="Ask a follow-up"
                  className="min-h-20 resize-none border-0 bg-transparent px-4 pt-3 text-sm shadow-none focus-visible:border-0"
                  disabled={active || continueMutation.isPending}
                  onChange={(event) => setMessage(event.currentTarget.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                      event.preventDefault();
                      sendFollowUp();
                    }
                  }}
                  placeholder={
                    active
                      ? "HALO is working — wait for this turn to finish…"
                      : "Ask a follow-up about this analysis…"
                  }
                  value={message}
                />
                <div className="flex items-center justify-between px-3 pb-2.5">
                  <span className="text-[11px] text-muted-foreground">
                    Follow-ups re-run HALO over the same traces with full context.
                  </span>
                  <Button
                    aria-label="Send follow-up"
                    disabled={!message.trim() || active || continueMutation.isPending}
                    onClick={sendFollowUp}
                    size="icon"
                    title="Send (⌘↵)"
                  >
                    {continueMutation.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <ArrowUp className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Divider + drag handle between the panes. */}
        <div
          aria-label="Resize details panel"
          aria-orientation="vertical"
          className="group relative hidden w-1.5 shrink-0 cursor-col-resize lg:block"
          onKeyDown={rail.onKeyDown}
          onPointerDown={rail.onPointerDown}
          role="separator"
          tabIndex={0}
        >
          <div
            className={cn(
              "absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-border/50 transition-colors",
              "group-hover:bg-detail-brand/50 group-focus-visible:bg-detail-brand",
              rail.dragging && "bg-detail-brand",
            )}
          />
          <div
            className={cn(
              "absolute left-1/2 top-1/2 h-9 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-border transition-colors",
              "group-hover:bg-detail-brand/70 group-focus-visible:bg-detail-brand",
              rail.dragging && "bg-detail-brand",
            )}
          />
        </div>

        {/* Details rail — flat sections, no cards. */}
        <aside
          className="hidden shrink-0 overflow-y-auto lg:block"
          style={{ width: rail.width }}
        >
          <RailSection
            icon={Activity}
            title="Status"
            trailing={<StatusBadge size="sm" status={run.status} />}
          >
            <div className="space-y-3">
              <RunPhaseTimeline run={run} />
              {active ? <ProgressBar value={run.progress} /> : null}
            </div>
          </RailSection>

          {turns.some(
            (turn) =>
              turn.role === "assistant" &&
              (turn.status === "completed" || turn.status === "incomplete") &&
              turn.content.trim(),
          ) ? (
            <RailSection
              description="Open the findings in a coding agent and let it implement the fixes."
              icon={SquareArrowOutUpRight}
              title="Act on this report"
            >
              <OpenInToolBar runId={run.id} />
            </RailSection>
          ) : null}

          <RailSection icon={Info} title="Run details">
            <div className="divide-y divide-subtle">
              <DetailRow label="Target" value={targetLabel(run.targetType)} />
              <DetailRow label="Provider" value={run.providerName || "provider"} />
              <DetailRow label="Model" value={run.model || "model"} />
              <DetailRow
                label="Started"
                value={run.startedAt ? formatTimestamp(run.startedAt) : "queued"}
              />
              <DetailRow
                label="Turns"
                value={String(turns.filter((turn) => turn.role === "assistant").length)}
              />
              <DetailRow label="Traces" value={run.traceCount.toLocaleString()} />
              {run.sessionCount > 0 ? (
                <DetailRow label="Sessions" value={run.sessionCount.toLocaleString()} />
              ) : null}
              <DetailRow label="Spans" value={run.spanCount.toLocaleString()} />
            </div>
          </RailSection>

          <FiltersSection filters={(run.filters ?? {}) as TelemetryFilters} />

          <RailSection icon={FileBox} title="Artifacts">
            <div className="space-y-2 text-xs text-muted-foreground">
              <CopyLine label="Export" value={run.exportPath} />
              <CopyLine label="Result" value={run.resultPath} />
            </div>
          </RailSection>
        </aside>
      </div>

      <RunConfigDialog
        initialValues={configInitialValues}
        onOpenChange={setConfigOpen}
        onStarted={(started) => {
          void navigate({ params: { runId: started.id }, to: "/analysis/$runId" });
        }}
        open={configOpen}
      />

      <Dialog
        cancelTitle="Cancel"
        confirmButtonVariant="destructive"
        confirmTitle="Delete run"
        dialogDescription={`This permanently removes "${run.title}", its conversation, events, and report files.`}
        dialogTitle="Delete this HALO run?"
        disabled={deleteMutation.isPending}
        loading={deleteMutation.isPending}
        onConfirm={() => deleteMutation.mutate({ runId })}
        onOpenChange={setDeleteOpen}
        open={deleteOpen}
      />
    </Shell>
  );
}

/** Width state for the details rail, persisted across sessions. */
function useResizableRail() {
  const [width, setWidth] = useState(RAIL_DEFAULT_WIDTH);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    const stored = Number(localStorage.getItem(RAIL_WIDTH_STORAGE_KEY));
    if (Number.isFinite(stored) && stored >= RAIL_MIN_WIDTH && stored <= RAIL_MAX_WIDTH) {
      setWidth(stored);
    }
  }, []);

  const persist = (value: number) => {
    localStorage.setItem(RAIL_WIDTH_STORAGE_KEY, String(value));
  };

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = width;
    setDragging(true);
    let latest = startWidth;
    const onMove = (moveEvent: PointerEvent) => {
      latest = clampRailWidth(startWidth + (startX - moveEvent.clientX));
      setWidth(latest);
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      setDragging(false);
      persist(latest);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const delta = event.key === "ArrowLeft" ? 16 : -16;
    setWidth((current) => {
      const next = clampRailWidth(current + delta);
      persist(next);
      return next;
    });
  };

  return { dragging, onKeyDown, onPointerDown, width };
}

function clampRailWidth(value: number) {
  return Math.min(RAIL_MAX_WIDTH, Math.max(RAIL_MIN_WIDTH, value));
}

function Shell({
  actions,
  children,
  title,
}: {
  actions?: ReactNode;
  children: ReactNode;
  title: string;
}) {
  return (
    <main className="h-screen overflow-hidden bg-background text-foreground">
      <AppHeader actions={actions} title={title} />
      {/* Fixed-height shell: each pane scrolls itself, the nav rail stays put. */}
      <div className="grid h-full min-h-0 grid-cols-[14rem_minmax(0,1fr)] pt-14">
        <WorkspaceNav active="analysis" />
        <section className="flex min-h-0 min-w-0 flex-col overflow-hidden">{children}</section>
      </div>
    </main>
  );
}

function RailSection({
  children,
  description,
  icon: Icon,
  title,
  trailing,
}: {
  children: ReactNode;
  description?: string;
  icon: typeof Activity;
  title: string;
  trailing?: ReactNode;
}) {
  return (
    <section className="border-b border-subtle px-5 py-4 last:border-b-0">
      <div className={description ? "mb-1 flex items-center gap-2" : "mb-3 flex items-center gap-2"}>
        <Icon className="h-3.5 w-3.5 text-muted-foreground" />
        <h2 className="text-sm font-semibold">{title}</h2>
        {trailing ? <span className="ml-auto">{trailing}</span> : null}
      </div>
      {description ? (
        <p className="mb-3 text-xs text-muted-foreground">{description}</p>
      ) : null}
      {children}
    </section>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex min-w-0 items-center justify-between gap-3 py-2.5 text-sm first:pt-0 last:pb-0">
      <span className="text-muted-foreground">{label}</span>
      <span className="min-w-0 truncate text-right font-medium">{value}</span>
    </div>
  );
}

function CopyLine({ label, value }: { label: string; value: string | null }) {
  if (!value) return null;
  return (
    <div className="flex items-center justify-between gap-3">
      <span>{label}</span>
      <button
        className="flex min-w-0 items-center gap-2 font-mono text-foreground"
        onClick={async () => {
          await navigator.clipboard.writeText(value);
          toast.success({ title: "Copied" });
        }}
        type="button"
      >
        <span className="truncate">{value}</span>
        <Copy className="h-3.5 w-3.5 shrink-0" />
      </button>
    </div>
  );
}

function FiltersSection({ filters }: { filters: TelemetryFilters }) {
  const chips = filterChips(filters);
  if (chips.length === 0) return null;
  return (
    <RailSection icon={Filter} title="Filters">
      <div className="flex flex-wrap gap-1.5">
        {chips.map((chip) => (
          <Badge key={chip} size="sm" variant="outline">
            {chip}
          </Badge>
        ))}
      </div>
    </RailSection>
  );
}

function filterChips(filters: TelemetryFilters): string[] {
  const chips: string[] = [];
  if (filters.startDate) {
    chips.push(`since ${formatTimestamp(new Date(filters.startDate).toISOString())}`);
  }
  if (filters.status) chips.push(`status: ${filters.status}`);
  for (const source of filters.sources ?? []) chips.push(`source: ${source}`);
  for (const service of filters.serviceNames ?? []) chips.push(`service: ${service}`);
  for (const agent of filters.agents ?? []) chips.push(`agent: ${agent}`);
  for (const model of filters.llmModelNames ?? []) chips.push(`model: ${model}`);
  return chips;
}
