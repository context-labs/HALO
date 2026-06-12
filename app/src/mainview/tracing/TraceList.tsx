import { Search } from "lucide-react";

import { EmptyState } from "~/lib/ui";
import {
  formatDuration,
  formatMoney,
  formatTimestamp,
  sourceLabel,
} from "~/lib/format";
import { showDesktopRowContextMenu } from "~/desktop/desktopBridge";
import type { Trace } from "../../server/telemetry/types";
import {
  KindStatusTile,
  LogTableFooter,
  LogTableHeader,
  LogTableSkeleton,
  MonoCell,
  PreviewCell,
  SourceGlyph,
  logRowClassName,
  logRowStatus,
} from "./logTable";

const GRID_COLS =
  "grid-cols-[120px_minmax(150px,1fr)_minmax(170px,1.4fr)_minmax(170px,1.4fr)_80px_72px]";

const COLUMNS = [
  { label: "Created" },
  { label: "Name" },
  { label: "Input" },
  { label: "Output" },
  { align: "right" as const, label: "Duration" },
  { align: "right" as const, label: "Cost" },
];

export function TraceList({
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
      <div className="min-h-0 flex-1 overflow-auto">
        <LogTableHeader columns={COLUMNS} gridClassName={GRID_COLS} />
        <LogTableSkeleton gridClassName={GRID_COLS} />
      </div>
    );
  }

  if (traces.length === 0) {
    return (
      <div className="flex flex-1 p-8">
        <EmptyState
          className="w-full self-center"
          description="Broaden the filters or wait for another local ingest batch."
          icon={Search}
          title="No matching traces"
        />
      </div>
    );
  }

  return (
    <div className="min-h-0 flex-1 overflow-auto">
      <LogTableHeader columns={COLUMNS} gridClassName={GRID_COLS} />
      {traces.map((trace) => {
          const status = logRowStatus({
            hasError: trace.hasError,
            isRecent: recentTraceIds.has(trace.traceId),
          });
          return (
            <button
              className={`${logRowClassName({
                active: trace.traceId === activeTraceId,
                flash: recentTraceIds.has(trace.traceId),
                status,
              })} ${GRID_COLS}`}
              key={trace.traceId}
              onClick={() => onSelectTrace(trace.traceId)}
              onContextMenu={(event) => {
                event.preventDefault();
                void showDesktopRowContextMenu({
                  id: trace.traceId,
                  kind: "trace",
                  sourceName:
                    trace.source === "local" ? null : sourceLabel(trace.source),
                  sourceUrl: trace.sourceUrl,
                });
              }}
              type="button"
            >
              <MonoCell>{formatTimestamp(trace.startTime)}</MonoCell>
              <span className="flex min-w-0 items-center gap-2.5">
                <KindStatusTile
                  kind={trace.rootObservationKind}
                  status={status}
                />
                <span className="min-w-0 truncate text-sm font-medium">
                  {trace.rootSpanName || "unnamed trace"}
                </span>
                {trace.source !== "local" ? (
                  <SourceGlyph
                    title={[
                      `Imported from ${sourceLabel(trace.source)}`,
                      trace.sourceConnectionName
                        ? `Connection: ${trace.sourceConnectionName}`
                        : null,
                    ]
                      .filter(Boolean)
                      .join("\n")}
                  />
                ) : null}
              </span>
              <PreviewCell text={trace.inputPreview} />
              <PreviewCell status={status} text={trace.outputPreview} />
              <MonoCell className="text-right">
                {formatDuration(trace.durationMs)}
              </MonoCell>
              <MonoCell className="text-right" muted={false}>
                {trace.totalCost == null
                  ? "—"
                  : formatMoney(Number(trace.totalCost))}
              </MonoCell>
            </button>
          );
        })}
      <LogTableFooter
        label="traces"
        shownCount={traces.length}
        totalCount={totalCount}
      />
    </div>
  );
}
