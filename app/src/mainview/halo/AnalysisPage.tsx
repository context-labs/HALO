import { useMemo, useRef, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { BrainCircuit, Filter, Play, Trash2 } from "lucide-react";

import { Button, Dialog, EmptyState, cn, toast } from "~/lib/ui";
import { trpc } from "~/trpc";
import { SetupNudgeBanner } from "~/onboarding/OnboardingPage";
import { WorkspaceNav } from "~/workspace/WorkspaceNav";
import { AppHeader } from "~/components/AppHeader";
import { FilterSelect } from "~/components/FilterSelect";
import { RunConfigDialog, type RunConfigInitialValues } from "./RunConfigDialog";
import { RunsTable, isActiveRun } from "./RunsTable";
import type { HaloRunView } from "./runShared";

type StatusGroup = "all" | "running" | "completed" | "failed";
type SortOrder = "newest" | "oldest";

const STATUS_GROUPS: Array<{ id: StatusGroup; label: string }> = [
  { id: "all", label: "All" },
  { id: "running", label: "Running" },
  { id: "completed", label: "Completed" },
  { id: "failed", label: "Failed" },
];

function statusGroupOf(run: HaloRunView): Exclude<StatusGroup, "all"> {
  if (isActiveRun(run)) return "running";
  if (run.status === "completed" || run.status === "incomplete") return "completed";
  return "failed";
}

export function AnalysisPage() {
  const navigate = useNavigate();
  const utils = trpc.useUtils();
  const [statusGroup, setStatusGroup] = useState<StatusGroup>("all");
  const [sortOrder, setSortOrder] = useState<SortOrder>("newest");
  const [configOpen, setConfigOpen] = useState(false);
  const [configInitialValues, setConfigInitialValues] = useState<
    RunConfigInitialValues | undefined
  >(undefined);
  const [runPendingDelete, setRunPendingDelete] = useState<HaloRunView | null>(null);
  const listInvalidateTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const runsQuery = trpc.halo.runs.list.useQuery({ limit: 100 });

  trpc.live.workspace.useSubscription(undefined, {
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
      // Merge progress/status in place; refetch (debounced) only on terminal
      // events so brand-new runs appear without a per-delta refetch storm.
      utils.halo.runs.list.setData({ limit: 100 }, (current) =>
        current?.map((run) =>
          run.id === payload.run.id ? { ...run, ...payload.run } : run,
        ),
      );
      const isNewRun =
        payload.type === "halo.run.updated" &&
        !utils.halo.runs.list
          .getData({ limit: 100 })
          ?.some((run) => run.id === payload.run.id);
      const terminal =
        payload.type === "halo.run.completed" || payload.type === "halo.run.failed";
      if ((terminal || isNewRun) && !listInvalidateTimer.current) {
        listInvalidateTimer.current = setTimeout(() => {
          listInvalidateTimer.current = null;
          void utils.halo.runs.list.invalidate();
        }, 250);
      }
    },
  });

  const cancelMutation = trpc.halo.runs.cancel.useMutation({
    async onSuccess() {
      toast.success({ title: "HALO run cancelled" });
      await utils.halo.runs.list.invalidate();
    },
  });
  const deleteMutation = trpc.halo.runs.delete.useMutation({
    async onSuccess() {
      setRunPendingDelete(null);
      toast.success({ title: "HALO run deleted" });
      await utils.halo.runs.list.invalidate();
    },
    onError(error) {
      toast.error({ title: "Could not delete run", description: error.message });
    },
  });

  const runs = useMemo(() => runsQuery.data ?? [], [runsQuery.data]);
  const groupCounts = useMemo(() => {
    const counts: Record<StatusGroup, number> = {
      all: runs.length,
      completed: 0,
      failed: 0,
      running: 0,
    };
    for (const run of runs) counts[statusGroupOf(run)] += 1;
    return counts;
  }, [runs]);

  const visibleRuns = useMemo(() => {
    const filtered =
      statusGroup === "all"
        ? runs
        : runs.filter((run) => statusGroupOf(run) === statusGroup);
    return [...filtered].sort((a, b) => {
      const delta = Date.parse(b.createdAt) - Date.parse(a.createdAt);
      return sortOrder === "newest" ? delta : -delta;
    });
  }, [runs, sortOrder, statusGroup]);

  const openRun = (run: HaloRunView) => {
    void navigate({ params: { runId: run.id }, to: "/analysis/$runId" });
  };

  return (
    <main className="min-h-screen bg-background text-foreground">
      <AppHeader
        icon={<BrainCircuit className="h-4 w-4 text-detail-brand" />}
        title="Analysis"
      />
      <div className="grid min-h-[calc(100vh-3.5rem)] grid-cols-[14rem_minmax(0,1fr)] pt-14">
        <WorkspaceNav active="analysis" />
        <section className="min-w-0 overflow-auto">
          <div className="mx-auto flex max-w-6xl flex-col gap-6 p-8">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <h1 className="text-2xl tracking-normal">Analysis</h1>
                <p className="mt-1 text-sm text-muted-foreground">
                  HALO's RLM digs through your traces and reports what's broken
                  and how to fix it.
                </p>
              </div>
              <Button
                onClick={() => {
                  setConfigInitialValues(undefined);
                  setConfigOpen(true);
                }}
              >
                <Play className="mr-2 h-4 w-4" />
        
            <SetupNudgeBanner />
        Run Analysis
              </Button>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-2">
                {STATUS_GROUPS.map((group) => (
                  <button
                    className={cn(
                      "flex items-center gap-1.5 rounded-md border border-border/60 px-3 py-1.5 text-sm transition hover:bg-muted/50",
                      statusGroup === group.id &&
                        "border-border bg-accent font-medium",
                    )}
                    key={group.id}
                    onClick={() => setStatusGroup(group.id)}
                    type="button"
                  >
                    {group.label}
                    <span className="text-xs tabular-nums text-muted-foreground">
                      {groupCounts[group.id]}
                    </span>
                  </button>
                ))}
              </div>
              <FilterSelect
                ariaLabel="Sort runs"
                onChange={(value) => setSortOrder(value as SortOrder)}
                options={[
                  { label: "Newest first", value: "newest" },
                  { label: "Oldest first", value: "oldest" },
                ]}
                triggerClassName="h-9 w-36"
                value={sortOrder}
              />
            </div>

            {runs.length > 0 && visibleRuns.length === 0 ? (
              <EmptyState
                action={
                  <Button
                    onClick={() => setStatusGroup("all")}
                    size="sm"
                    variant="outline"
                  >
                    Show all runs
                  </Button>
                }
                className="w-full py-16"
                description={`None of your ${runs.length} runs are ${statusGroup} right now.`}
                icon={Filter}
                title={`No ${statusGroup} runs`}
              />
            ) : (
              <RunsTable
                onCancel={(run) => cancelMutation.mutate({ runId: run.id })}
                onDelete={setRunPendingDelete}
                onOpen={openRun}
                onRunAnalysis={() => {
                  setConfigInitialValues(undefined);
                  setConfigOpen(true);
                }}
                runs={visibleRuns}
              />
            )}
          </div>
        </section>
      </div>

      <RunConfigDialog
        initialValues={configInitialValues}
        onOpenChange={setConfigOpen}
        onStarted={(run) => {
          void navigate({ params: { runId: run.id }, to: "/analysis/$runId" });
        }}
        open={configOpen}
      />

      <Dialog
        cancelTitle="Cancel"
        confirmButtonVariant="destructive"
        confirmTitle="Delete run"
        dialogDescription={`This permanently removes "${runPendingDelete?.title ?? ""}", its conversation, events, and report files.`}
        dialogTitle="Delete this HALO run?"
        disabled={deleteMutation.isPending}
        loading={deleteMutation.isPending}
        onConfirm={() => {
          if (runPendingDelete) {
            deleteMutation.mutate({ runId: runPendingDelete.id });
          }
        }}
        onOpenChange={(open) => {
          if (!open) setRunPendingDelete(null);
        }}
        open={Boolean(runPendingDelete)}
      >
        {runPendingDelete && isActiveRun(runPendingDelete) ? (
          <div className="rounded-md border border-destructive-border bg-destructive/5 p-4 text-sm">
            <div className="flex items-start gap-3">
              <Trash2 className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
              <p className="text-muted-foreground">
                This run is still in progress — deleting it cancels the analysis
                first.
              </p>
            </div>
          </div>
        ) : null}
      </Dialog>
    </main>
  );
}
