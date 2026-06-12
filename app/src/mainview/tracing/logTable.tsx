import type { ReactNode } from "react";
import {
  Activity,
  Bot,
  DownloadCloud,
  Layers3,
  ListTree,
  Sparkles,
  Wrench,
} from "lucide-react";

import { Skeleton, cn } from "~/lib/ui";
import { FilterSelect } from "~/components/FilterSelect";
import type { DateRange } from "~/lib/format";
import type { ObservationKind } from "../../server/telemetry/types";
import type { LiveStatus } from "./TraceTitleBar";

export type LogRowStatus = "error" | "ok" | "running";

/** Visual status for a list row: live rows pulse, error rows tint red. */
export function logRowStatus(input: {
  hasError: boolean;
  isRecent: boolean;
}): LogRowStatus {
  if (input.hasError) return "error";
  if (input.isRecent) return "running";
  return "ok";
}

export function LogTableHeader({
  columns,
  gridClassName,
}: {
  columns: Array<{ align?: "left" | "right"; label: string }>;
  gridClassName: string;
}) {
  return (
    <div
      className={cn(
        "sticky top-0 z-10 grid items-center gap-3 border-b border-border/50 bg-background px-6 py-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground",
        gridClassName,
      )}
    >
      {columns.map((column) => (
        <span
          className={cn(column.align === "right" && "text-right")}
          key={column.label}
        >
          {column.label}
        </span>
      ))}
    </div>
  );
}

const KIND_ICONS: Partial<Record<ObservationKind, typeof Activity>> = {
  AGENT: Bot,
  CHAIN: ListTree,
  EMBEDDING: Layers3,
  LLM: Sparkles,
  TOOL: Wrench,
};

/** Small rounded glyph tile: icon by span kind, tint by row status. */
export function KindStatusTile({
  kind,
  status,
}: {
  kind: ObservationKind;
  status: LogRowStatus;
}) {
  const Icon = KIND_ICONS[kind] ?? Activity;
  return (
    <span
      className={cn(
        "grid h-7 w-7 shrink-0 place-items-center rounded-md border",
        status === "error" &&
          "border-detail-failure/40 bg-detail-failure/10 text-detail-failure",
        status === "running" &&
          "trace-pulse border-detail-brand/40 bg-detail-brand/10 text-detail-brand",
        status === "ok" &&
          "border-detail-success/35 bg-detail-success/10 text-detail-success",
      )}
    >
      <Icon className="h-3.5 w-3.5" strokeWidth={1.75} />
    </span>
  );
}

/**
 * Icon-only import marker for table rows — the full source badge is too wide
 * for the Name column; details live in the tooltip and the detail sheet.
 */
export function SourceGlyph({ title }: { title: string }) {
  return (
    <span
      className="grid h-4 w-4 shrink-0 place-items-center text-detail-brand"
      title={title}
    >
      <DownloadCloud className="h-3.5 w-3.5" />
    </span>
  );
}

/** One-line input/output preview with status-aware tone. */
export function PreviewCell({
  status = "ok",
  text,
}: {
  status?: LogRowStatus;
  text: string | null;
}) {
  if (!text) {
    return <span className="truncate text-sm text-muted-foreground/50">—</span>;
  }
  return (
    <span
      className={cn(
        "truncate text-sm",
        status === "error"
          ? "text-detail-failure"
          : status === "running"
            ? "text-detail-brand"
            : "text-muted-foreground",
      )}
      title={text}
    >
      {text}
    </span>
  );
}

export function LogTableFooter({
  label,
  shownCount,
  totalCount,
}: {
  label: string;
  shownCount: number;
  totalCount: number;
}) {
  return (
    <div className="flex items-center justify-between px-6 py-3 text-xs text-muted-foreground">
      <span>
        Showing {shownCount.toLocaleString()} {label}
      </span>
      <span>
        {totalCount.toLocaleString()} matching {label}
      </span>
    </div>
  );
}

export function LogTableSkeleton({ gridClassName }: { gridClassName: string }) {
  return (
    <div>
      {Array.from({ length: 8 }, (_, index) => (
        <div
          className={cn(
            "grid items-center gap-3 border-b border-border/40 px-6 py-3",
            gridClassName,
          )}
          key={index}
        >
          <Skeleton className="h-3.5 w-24 rounded" />
          <span className="flex items-center gap-2.5">
            <Skeleton className="h-7 w-7 rounded-md" />
            <Skeleton className="h-3.5 w-32 rounded" />
          </span>
          <Skeleton className="h-3.5 w-full max-w-64 rounded" />
          <Skeleton className="h-3.5 w-full max-w-64 rounded" />
          <Skeleton className="ml-auto h-3.5 w-12 rounded" />
          <Skeleton className="ml-auto h-3.5 w-10 rounded" />
        </div>
      ))}
    </div>
  );
}

/** Pulsing LIVE badge fused to the time-window select. */
export function LiveRangeControl({
  dateRange,
  liveStatus,
  onDateRangeChange,
}: {
  dateRange: DateRange;
  liveStatus: LiveStatus;
  onDateRangeChange: (value: DateRange) => void;
}) {
  const live = liveStatus === "live";
  return (
    <div className="flex items-center">
      <span
        className={cn(
          "flex h-9 items-center rounded-l-md border border-r-0 px-2.5 text-[10px] font-semibold tracking-[0.08em]",
          live
            ? "border-detail-brand/45 bg-detail-brand/10 text-detail-brand"
            : "border-border/60 bg-muted/40 text-muted-foreground",
        )}
      >
        <span className={cn(live && "trace-pulse")}>LIVE</span>
      </span>
      <FilterSelect
        ariaLabel="Time window"
        onChange={(value) => onDateRangeChange(value as DateRange)}
        options={[
          { label: "Last hour", value: "1h" },
          { label: "Last 24 hours", value: "24h" },
          { label: "Last 7 days", value: "7d" },
          { label: "All time", value: "all" },
        ]}
        triggerClassName="h-9 w-36 rounded-l-none"
        value={dateRange}
      />
    </div>
  );
}

/** Shared row chrome: hover/active/error backgrounds + accent left edge. */
export function logRowClassName(input: {
  active: boolean;
  flash: boolean;
  status: LogRowStatus;
}): string {
  return cn(
    "relative grid w-full items-center gap-3 border-b border-border/40 px-6 py-2.5 text-left transition-colors",
    "before:absolute before:bottom-0 before:left-0 before:top-0 before:w-0.5 before:bg-transparent",
    input.active
      ? "bg-muted before:bg-detail-brand"
      : input.status === "error"
        ? "bg-detail-failure/5 hover:bg-detail-failure/10"
        : "hover:bg-muted/50",
    input.flash && "live-trace-flash",
  );
}

export function MonoCell({
  children,
  className,
  muted = true,
}: {
  children: ReactNode;
  className?: string;
  muted?: boolean;
}) {
  return (
    <span
      className={cn(
        "truncate font-geist-mono text-xs tabular-nums",
        muted ? "text-muted-foreground" : "text-foreground",
        className,
      )}
    >
      {children}
    </span>
  );
}
