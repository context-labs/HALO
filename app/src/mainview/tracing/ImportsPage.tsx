import { useMemo, useRef, useState } from "react";
import { Link } from "@tanstack/react-router";
import { DownloadCloud, Filter, Square } from "lucide-react";

import { FileUp } from "lucide-react";

import { Badge, Button, EmptyState, cn, toast } from "~/lib/ui";
import { trpc } from "~/trpc";
import { WorkspaceNav } from "~/workspace/WorkspaceNav";
import { AppHeader } from "~/components/AppHeader";
import { FilterSelect } from "~/components/FilterSelect";
import { StatusBadge, ProgressBar } from "~/components/StatusBadge";
import { formatTimestamp } from "~/lib/format";
import type { FileImportJob } from "../../server/fileimport/types";
import type { LangfuseImportJob } from "../../server/langfuse/types";
import type { PhoenixImportJob } from "../../server/phoenix/types";
import { LangfuseLogo, PhoenixLogo } from "./ImportDataScreen";

type StatusGroup = "all" | "running" | "completed" | "failed";
type SortOrder = "newest" | "oldest";

/** One row in the merged imports table, tagged with its integration. */
type ImportRow =
  | { provider: "langfuse"; job: LangfuseImportJob }
  | { provider: "phoenix"; job: PhoenixImportJob }
  | { provider: "file"; job: FileImportJob };

const STATUS_GROUPS: Array<{ id: StatusGroup; label: string }> = [
  { id: "all", label: "All" },
  { id: "running", label: "Running" },
  { id: "completed", label: "Completed" },
  { id: "failed", label: "Failed" },
];

const GRID_COLS =
  "grid-cols-[minmax(260px,1.6fr)_minmax(150px,0.7fr)_150px_150px_64px]";

function isActiveImport(row: ImportRow) {
  return row.job.status === "queued" || row.job.status === "running";
}

function statusGroupOf(row: ImportRow): Exclude<StatusGroup, "all"> {
  if (isActiveImport(row)) return "running";
  if (row.job.status === "completed") return "completed";
  return "failed";
}

export function ImportsPage() {
  const utils = trpc.useUtils();
  const [statusGroup, setStatusGroup] = useState<StatusGroup>("all");
  const [sortOrder, setSortOrder] = useState<SortOrder>("newest");
  const listInvalidateTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const langfuseJobsQuery = trpc.langfuse.imports.list.useQuery({ limit: 100 });
  const phoenixJobsQuery = trpc.phoenix.imports.list.useQuery({ limit: 100 });
  const fileJobsQuery = trpc.fileImport.imports.list.useQuery({ limit: 100 });

  trpc.live.workspace.useSubscription(undefined, {
    onData(eventEnvelope) {
      if (eventEnvelope.data.payload.type !== "import.job.updated") return;
      if (listInvalidateTimer.current) return;
      listInvalidateTimer.current = setTimeout(() => {
        listInvalidateTimer.current = null;
        void utils.langfuse.imports.list.invalidate();
        void utils.phoenix.imports.list.invalidate();
        void utils.fileImport.imports.list.invalidate();
      }, 300);
    },
  });

  const cancelLangfuse = trpc.langfuse.imports.cancel.useMutation({
    async onSuccess() {
      toast.success({ title: "Import cancelled" });
      await utils.langfuse.imports.list.invalidate();
    },
  });
  const cancelPhoenix = trpc.phoenix.imports.cancel.useMutation({
    async onSuccess() {
      toast.success({ title: "Import cancelled" });
      await utils.phoenix.imports.list.invalidate();
    },
  });
  const cancelFile = trpc.fileImport.imports.cancel.useMutation({
    async onSuccess() {
      toast.success({ title: "Import cancelled" });
      await utils.fileImport.imports.list.invalidate();
    },
  });

  const rows = useMemo<ImportRow[]>(
    () => [
      ...(langfuseJobsQuery.data ?? []).map(
        (job): ImportRow => ({ job, provider: "langfuse" }),
      ),
      ...(phoenixJobsQuery.data ?? []).map(
        (job): ImportRow => ({ job, provider: "phoenix" }),
      ),
      ...(fileJobsQuery.data ?? []).map(
        (job): ImportRow => ({ job, provider: "file" }),
      ),
    ],
    [fileJobsQuery.data, langfuseJobsQuery.data, phoenixJobsQuery.data],
  );
  const groupCounts = useMemo(() => {
    const counts: Record<StatusGroup, number> = {
      all: rows.length,
      completed: 0,
      failed: 0,
      running: 0,
    };
    for (const row of rows) counts[statusGroupOf(row)] += 1;
    return counts;
  }, [rows]);

  const visibleRows = useMemo(() => {
    const filtered =
      statusGroup === "all"
        ? rows
        : rows.filter((row) => statusGroupOf(row) === statusGroup);
    return [...filtered].sort((a, b) => {
      const delta = Date.parse(b.job.createdAt) - Date.parse(a.job.createdAt);
      return sortOrder === "newest" ? delta : -delta;
    });
  }, [rows, sortOrder, statusGroup]);

  const cancelRow = (row: ImportRow) => {
    if (row.provider === "phoenix") {
      cancelPhoenix.mutate({ jobId: row.job.id });
    } else if (row.provider === "file") {
      cancelFile.mutate({ jobId: row.job.id });
    } else {
      cancelLangfuse.mutate({ jobId: row.job.id });
    }
  };

  return (
    <main className="h-screen overflow-hidden bg-background text-foreground">
      <AppHeader title="Imports" />
      <div className="grid h-full min-h-0 grid-cols-[14rem_minmax(0,1fr)] pt-14">
        <WorkspaceNav active="imports" />
        <section className="min-h-0 min-w-0 overflow-y-auto">
          <div className="mx-auto flex max-w-6xl flex-col gap-6 p-8">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <h1 className="text-2xl tracking-normal">Imports</h1>
                <p className="mt-1 text-sm text-muted-foreground">
                  Trace history brought into HALO from other observability
                  tools.
                </p>
              </div>
              <Button asChild>
                <Link to="/import-data">
                  <DownloadCloud className="mr-2 h-4 w-4" />
                  Import Data
                </Link>
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
                ariaLabel="Sort imports"
                onChange={(value) => setSortOrder(value as SortOrder)}
                options={[
                  { label: "Newest first", value: "newest" },
                  { label: "Oldest first", value: "oldest" },
                ]}
                triggerClassName="h-9 w-36"
                value={sortOrder}
              />
            </div>

            {rows.length === 0 ? (
              <EmptyState
                action={
                  <Button asChild size="sm" variant="outline">
                    <Link to="/import-data">Import Data</Link>
                  </Button>
                }
                className="w-full py-16"
                description="Bring historical traces in from Langfuse or Phoenix and they'll show up here."
                icon={DownloadCloud}
                title="No imports yet"
              />
            ) : visibleRows.length === 0 ? (
              <EmptyState
                action={
                  <Button
                    onClick={() => setStatusGroup("all")}
                    size="sm"
                    variant="outline"
                  >
                    Show all imports
                  </Button>
                }
                className="w-full py-16"
                description={`None of your ${groupCounts.all} imports are ${statusGroupLabel(statusGroup)} right now.`}
                icon={Filter}
                title={`No ${statusGroupLabel(statusGroup)} imports`}
              />
            ) : (
              <ImportsTable onCancel={cancelRow} rows={visibleRows} />
            )}
          </div>
        </section>
      </div>
    </main>
  );
}

function ImportsTable({
  onCancel,
  rows,
}: {
  onCancel: (row: ImportRow) => void;
  rows: ImportRow[];
}) {
  return (
    <div className="rounded-xl border border-border/55">
      <div
        className={cn(
          "grid rounded-t-xl border-b border-border/50 bg-muted/30 px-4 py-2.5 text-xs font-medium uppercase tracking-wide text-muted-foreground",
          GRID_COLS,
        )}
      >
        <div>Import</div>
        <div className="text-right">Imported</div>
        <div className="pl-6">Status</div>
        <div className="text-right">Created</div>
        <div />
      </div>
      <div>
        {rows.map((row) => {
          const { job, provider } = row;
          const providerLabel =
            provider === "phoenix"
              ? "Phoenix"
              : provider === "file"
                ? "File"
                : "Langfuse";
          const active = isActiveImport(row);
          const created = new Date(job.createdAt);
          return (
            <div
              className={cn(
                "grid w-full items-center border-b border-border/40 px-4 py-3.5 last:rounded-b-xl last:border-b-0",
                GRID_COLS,
              )}
              key={`${provider}:${job.id}`}
            >
              <div className="flex min-w-0 items-center gap-3 pr-4">
                <span className="grid h-9 w-9 shrink-0 place-items-center rounded-md bg-background-muted">
                  {provider === "phoenix" ? (
                    <PhoenixLogo className="h-5 w-5" />
                  ) : provider === "file" ? (
                    <FileUp className="h-5 w-5 text-detail-brand" />
                  ) : (
                    <LangfuseLogo className="h-5 w-5" />
                  )}
                </span>
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">
                    {rowTitle(row) ?? providerLabel}
                  </p>
                  <div className="mt-1 flex min-w-0 items-center gap-2 overflow-hidden whitespace-nowrap">
                    <Badge size="sm" variant="outline">
                      {providerLabel}
                    </Badge>
                    <span className="truncate text-xs text-muted-foreground">
                      {filtersSummary(row)}
                    </span>
                  </div>
                </div>
              </div>

              <div className="text-right tabular-nums">
                <p className="text-sm">
                  {job.importedTraces.toLocaleString()}
                  {job.totalTraces > 0
                    ? ` / ${job.totalTraces.toLocaleString()}`
                    : ""}{" "}
                  traces
                </p>
                <p className="text-xs text-muted-foreground">
                  {job.importedObservations.toLocaleString()} observations
                  {job.failedTraces > 0 ? (
                    <span className="text-detail-failure">
                      {" "}
                      · {job.failedTraces.toLocaleString()} failed
                    </span>
                  ) : null}
                </p>
              </div>

              <div className="min-w-0 pl-6">
                <StatusBadge status={job.status} />
                {active ? (
                  <div className="mt-1.5 w-24">
                    <ProgressBar value={job.progress} />
                  </div>
                ) : null}
              </div>

              <div className="text-right tabular-nums">
                <p className="text-sm">
                  {created.toLocaleDateString(undefined, {
                    day: "numeric",
                    month: "short",
                    year: "numeric",
                  })}
                </p>
                <p className="text-xs text-muted-foreground">
                  {created.toLocaleTimeString(undefined, {
                    hour: "numeric",
                    minute: "2-digit",
                  })}
                </p>
              </div>

              <div className="flex items-center justify-end">
                {active ? (
                  <Button
                    aria-label="Cancel import"
                    onClick={() => onCancel(row)}
                    size="icon"
                    title="Cancel import"
                    variant="ghost"
                  >
                    <Square className="h-4 w-4" />
                  </Button>
                ) : null}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function statusGroupLabel(group: StatusGroup) {
  return group === "all" ? "matching" : group;
}

function rowTitle(row: ImportRow) {
  if (row.provider === "file") return row.job.fileName;
  return row.job.connectionName;
}

/** One-line description of what the import covered. */
function filtersSummary(row: ImportRow) {
  const parts: string[] = [];
  if (row.provider === "file") {
    if (row.job.skippedLines > 0) {
      parts.push(`${row.job.skippedLines.toLocaleString()} lines skipped`);
    }
    return parts.length > 0 ? parts.join(" \u00b7 ") : "whole file";
  }
  if (row.provider === "phoenix") {
    if (row.job.filters.projectName) {
      parts.push(`project: ${row.job.filters.projectName}`);
    }
    if (row.job.filters.fromTimestamp) {
      parts.push(`since ${formatTimestamp(row.job.filters.fromTimestamp)}`);
    }
  } else {
    const filters = row.job.filters;
    if (filters.fromTimestamp) {
      parts.push(`since ${formatTimestamp(filters.fromTimestamp)}`);
    }
    if (filters.environment) parts.push(`env: ${filters.environment}`);
    if (filters.traceName) parts.push(`name: ${filters.traceName}`);
    if (filters.tag) parts.push(`tag: ${filters.tag}`);
    if (filters.userId) parts.push(`user: ${filters.userId}`);
    if (filters.sessionId) parts.push(`session: ${filters.sessionId}`);
  }
  if (parts.length === 0) return "all traces";
  return parts.join(" · ");
}
