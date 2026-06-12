import { MessageSquare } from "lucide-react";

import { EmptyState } from "~/lib/ui";
import {
  formatDuration,
  formatMoney,
  formatTimestamp,
  sourceLabel,
} from "~/lib/format";
import { showDesktopRowContextMenu } from "~/desktop/desktopBridge";
import type { SessionSummary } from "../../server/telemetry/types";
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
  "grid-cols-[120px_minmax(150px,1fr)_minmax(150px,1.3fr)_minmax(150px,1.3fr)_52px_80px_72px]";

const COLUMNS = [
  { label: "Created" },
  { label: "Name" },
  { label: "Input" },
  { label: "Output" },
  { align: "right" as const, label: "Turns" },
  { align: "right" as const, label: "Duration" },
  { align: "right" as const, label: "Cost" },
];

export function SessionList({
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
      <div className="min-h-0 flex-1 overflow-auto">
        <LogTableHeader columns={COLUMNS} gridClassName={GRID_COLS} />
        <LogTableSkeleton gridClassName={GRID_COLS} />
      </div>
    );
  }

  if (sessions.length === 0) {
    return (
      <div className="flex flex-1 p-8">
        <EmptyState
          className="w-full self-center"
          description="Sessions appear when traces include a session ID. Traces without one stay hidden here."
          icon={MessageSquare}
          title="No sessions yet"
        />
      </div>
    );
  }

  return (
    <div className="min-h-0 flex-1 overflow-auto">
      <LogTableHeader columns={COLUMNS} gridClassName={GRID_COLS} />
      {sessions.map((session) => {
        const status = logRowStatus({
          hasError: session.hasError,
          isRecent: recentSessionIds.has(session.sessionId),
        });
        return (
          <button
            className={`${logRowClassName({
              active: session.sessionId === activeSessionId,
              flash: recentSessionIds.has(session.sessionId),
              status,
            })} ${GRID_COLS}`}
            key={session.sessionId}
            onClick={() => onSelectSession(session.sessionId)}
            onContextMenu={(event) => {
              event.preventDefault();
              void showDesktopRowContextMenu({
                id: session.sessionId,
                kind: "session",
              });
            }}
            type="button"
          >
            <MonoCell>{formatTimestamp(session.startTime)}</MonoCell>
            <span className="flex min-w-0 items-center gap-2.5">
              <KindStatusTile kind="CHAIN" status={status} />
              <span className="min-w-0 truncate text-sm font-medium">
                {session.latestTraceName || "unnamed session"}
              </span>
              {session.sources.some((source) => source !== "local") ? (
                <SourceGlyph
                  title={`Imported from ${session.sources
                    .filter((source) => source !== "local")
                    .map((source) => sourceLabel(source))
                    .join(", ")}`}
                />
              ) : null}
            </span>
            <PreviewCell text={session.inputPreview} />
            <PreviewCell status={status} text={session.outputPreview} />
            <MonoCell className="text-right">{session.traceCount}</MonoCell>
            <MonoCell className="text-right">
              {formatDuration(session.durationMs)}
            </MonoCell>
            <MonoCell className="text-right" muted={false}>
              {session.totalCost == null
                ? "—"
                : formatMoney(Number(session.totalCost))}
            </MonoCell>
          </button>
        );
      })}
      <LogTableFooter
        label="sessions"
        shownCount={sessions.length}
        totalCount={totalCount}
      />
    </div>
  );
}
