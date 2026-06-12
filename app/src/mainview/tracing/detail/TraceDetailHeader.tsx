import type { ReactNode } from "react";
import {
  Clock3,
  DollarSign,
  Maximize2,
  MessageSquare,
  Minus,
  Plus,
} from "lucide-react";

import { Button, cn } from "~/lib/ui";
import {
  compactNumber,
  formatDuration,
  formatMoney,
  formatTimestamp,
} from "~/lib/format";

export type TraceDetailStatus = "completed" | "failed" | "running";
export type TraceDetailViewMode = "conversation" | "timeline";

export function TraceDetailHeader({
  costTotal,
  description,
  durationMs,
  followingLatest,
  heat,
  narrow,
  onHeatChange,
  onViewModeChange,
  onZoomFit,
  onZoomIn,
  onZoomOut,
  spanCount,
  startedAt,
  status,
  title,
  tokens,
  viewMode,
}: {
  costTotal: number | null;
  description: string | null;
  durationMs: number | null;
  followingLatest: boolean;
  heat: boolean;
  narrow: boolean;
  onHeatChange: (heat: boolean) => void;
  onViewModeChange: (mode: TraceDetailViewMode) => void;
  onZoomFit: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  spanCount: number;
  startedAt: string | null;
  status: TraceDetailStatus;
  title: string;
  tokens: number | null;
  viewMode: TraceDetailViewMode;
}) {
  const statusTone =
    status === "failed"
      ? "border-detail-failure/45 bg-detail-failure/10 text-detail-failure"
      : status === "running"
        ? "border-detail-brand/45 bg-detail-brand/10 text-detail-brand"
        : "border-detail-success/40 bg-detail-success/10 text-detail-success";

  return (
    <header className="flex flex-none items-center gap-3 border-b border-subtle bg-background-muted/60 py-2.5 pl-5 pr-14">
      <div className="flex min-w-0 shrink items-center gap-2.5">
        <span className="grid h-7 w-7 flex-none place-items-center rounded-md bg-gradient-to-br from-trace-llm to-trace-agent text-[13px] font-semibold text-white">
          ⊶
        </span>
        <div className="flex min-w-0 flex-col">
          <div className="flex min-w-0 items-center gap-2">
            <span className="min-w-16 truncate text-sm font-semibold">
              {title}
            </span>
            {description ? (
              <span className="hidden flex-none font-geist-mono text-[11px] text-muted-foreground xl:inline">
                {shortenId(description)}
              </span>
            ) : null}
            <span
              className={cn(
                "flex flex-none items-center gap-1.5 rounded-full border px-2 py-px text-[11px]",
                statusTone,
              )}
            >
              <span
                className={cn(
                  "h-1.5 w-1.5 rounded-full bg-current",
                  status === "running" && "trace-pulse",
                )}
              />
              {status}
            </span>
            {followingLatest ? (
              <span className="flex flex-none items-center gap-1 rounded-full border border-detail-brand/40 bg-detail-brand/10 px-2 py-px text-[11px] text-detail-brand">
                following latest
              </span>
            ) : null}
          </div>
          {startedAt ? (
            <span className="truncate text-[11px] text-muted-foreground">
              {formatTimestamp(startedAt)}
            </span>
          ) : null}
        </div>
      </div>

      <div className="flex flex-none gap-0.5 rounded-lg border border-subtle bg-background p-0.5">
        {(
          [
            ["timeline", "Timeline", Clock3],
            ["conversation", "Conversation", MessageSquare],
          ] as const
        ).map(([mode, label, Icon]) => (
          <button
            className={cn(
              "flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition-colors",
              viewMode === mode
                ? "bg-muted text-foreground shadow-[inset_0_0_0_1px_var(--color-border)]"
                : "text-muted-foreground hover:text-foreground",
            )}
            key={mode}
            onClick={() => onViewModeChange(mode)}
            type="button"
          >
            <Icon className="h-3.5 w-3.5" />
            {label}
          </button>
        ))}
      </div>

      <div className="flex-1" />

      {!narrow ? (
        <>
          <div className="flex flex-none items-center gap-4 pr-1">
            <HeaderStat label="duration">
              {durationMs == null ? "—" : formatDuration(durationMs)}
            </HeaderStat>
            <HeaderStat label="spans">{spanCount}</HeaderStat>
            <HeaderStat label="tokens">
              {tokens == null ? "—" : compactNumber(tokens)}
            </HeaderStat>
            <HeaderStat label="cost">
              {costTotal == null ? "—" : formatMoney(costTotal)}
            </HeaderStat>
          </div>
          <div className="h-6 w-px flex-none bg-border/60" />
        </>
      ) : null}

      <div className="flex flex-none items-center gap-1.5">
        <Button
          className={cn(
            "h-8 gap-1.5 px-2.5 text-xs",
            heat &&
              "border-detail-warning/60 bg-detail-warning/10 text-detail-warning hover:bg-detail-warning/15 hover:text-detail-warning",
          )}
          onClick={() => onHeatChange(!heat)}
          size="sm"
          title="Color spans by cost"
          variant="outline"
        >
          <DollarSign className="h-3.5 w-3.5" />
          Cost heatmap
        </Button>
        {viewMode === "timeline" ? (
          <div className="flex items-center gap-0.5">
            <Button
              aria-label="Zoom in"
              className="h-8 w-8"
              onClick={onZoomIn}
              size="icon"
              variant="outline"
            >
              <Plus className="h-3.5 w-3.5" />
            </Button>
            <Button
              aria-label="Zoom out"
              className="h-8 w-8"
              onClick={onZoomOut}
              size="icon"
              variant="outline"
            >
              <Minus className="h-3.5 w-3.5" />
            </Button>
            <Button
              aria-label="Fit trace"
              className="h-8 w-8"
              onClick={onZoomFit}
              size="icon"
              variant="outline"
            >
              <Maximize2 className="h-3.5 w-3.5" />
            </Button>
          </div>
        ) : null}
      </div>
    </header>
  );
}

function HeaderStat({ children, label }: { children: ReactNode; label: string }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className="text-[10px] uppercase tracking-[0.07em] text-muted-foreground">
        {label}
      </span>
      <span className="font-geist-mono text-xs text-foreground">{children}</span>
    </div>
  );
}

function shortenId(value: string) {
  return value.length > 18 ? `${value.slice(0, 16)}…` : value;
}
