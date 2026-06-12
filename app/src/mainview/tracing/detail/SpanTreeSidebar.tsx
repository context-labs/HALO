import { ChevronRight } from "lucide-react";

import { cn } from "~/lib/ui";
import { formatDuration } from "~/lib/format";
import { spanColor, useTraceColors } from "./spanKinds";
import { spanEndMs, type TimelineRow } from "./timelineMath";

export function SpanTreeSidebar({
  expanded,
  heat,
  maxCost,
  nowMs,
  onSelect,
  onToggle,
  recentSpanIds,
  rows,
  selectedKey,
}: {
  expanded: ReadonlySet<string>;
  heat: boolean;
  maxCost: number;
  nowMs: number;
  onSelect: (key: string) => void;
  onToggle: (key: string) => void;
  recentSpanIds: Set<string>;
  rows: TimelineRow[];
  selectedKey: string | null;
}) {
  const colors = useTraceColors();
  return (
    <div className="w-62 flex-none overflow-y-auto border-r border-subtle bg-background-muted/40 py-1.5">
      <div className="px-3.5 pb-1.5 pt-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
        Spans
      </div>
      <div className="flex flex-col">
        {rows.map((row) => {
          const span = row.span;
          const selected = row.key === selectedKey;
          const error = span.statusCode === "STATUS_CODE_ERROR";
          const color = spanColor(span, colors, heat, maxCost);
          return (
            <div
              className={cn(
                "flex cursor-pointer items-center gap-1.5 border-l-2 py-1 pr-2.5",
                selected
                  ? "border-l-detail-brand bg-muted"
                  : "border-l-transparent hover:bg-muted/50",
                recentSpanIds.has(row.key) && "live-span-flash",
              )}
              key={row.key}
              onClick={() => onSelect(row.key)}
              role="button"
              style={{ paddingLeft: 10 + row.depth * 14 }}
              tabIndex={0}
            >
              {row.hasChildren ? (
                <button
                  aria-label="Toggle children"
                  className="grid h-4 w-4 flex-none place-items-center text-muted-foreground hover:text-foreground"
                  onClick={(event) => {
                    event.stopPropagation();
                    onToggle(row.key);
                  }}
                  type="button"
                >
                  <ChevronRight
                    className={cn(
                      "h-3 w-3 transition-transform",
                      expanded.has(row.key) && "rotate-90",
                    )}
                  />
                </button>
              ) : (
                <span className="w-4 flex-none" />
              )}
              <span
                className="h-[7px] w-[7px] flex-none rounded-sm"
                style={{ background: color }}
              />
              <span
                className={cn(
                  "min-w-0 flex-1 truncate text-xs",
                  selected
                    ? "font-semibold text-foreground"
                    : "text-muted-foreground",
                )}
              >
                {span.spanName}
              </span>
              <span
                className={cn(
                  "flex-none font-geist-mono text-[10px]",
                  error ? "text-detail-failure" : "text-muted-foreground/80",
                )}
              >
                {error ? "✕ " : ""}
                {formatDuration(
                  Math.max(0, spanEndMs(span, nowMs) - span.startTimeMs),
                )}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
