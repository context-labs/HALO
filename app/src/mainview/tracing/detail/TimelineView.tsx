import { forwardRef, useMemo, useState } from "react";

import type { SpanNode } from "../../../server/telemetry/types";
import { isSessionTraceGroupSpan } from "../spanTree";
import { SpanTooltip, type SpanHover } from "./SpanTooltip";
import { SpanTreeSidebar } from "./SpanTreeSidebar";
import { WaterfallCanvas, type WaterfallHandle } from "./WaterfallCanvas";
import {
  buildGroups,
  buildRows,
  timelineDomain,
  type TimelineRow,
} from "./timelineMath";

export const TimelineView = forwardRef<
  WaterfallHandle,
  {
    expanded: ReadonlySet<string>;
    heat: boolean;
    maxCost: number;
    nodeByKey: Map<string, SpanNode>;
    nowMs: number;
    onSelect: (key: string | null) => void;
    onToggle: (key: string) => void;
    recentSpanIds: Set<string>;
    running: boolean;
    selectedSpanKey: string | null;
    tree: SpanNode[];
  }
>(function TimelineView(
  {
    expanded,
    heat,
    maxCost,
    nodeByKey,
    nowMs,
    onSelect,
    onToggle,
    recentSpanIds,
    running,
    selectedSpanKey,
    tree,
  },
  waterfallRef,
) {
  const [hover, setHover] = useState<SpanHover | null>(null);

  const rows: TimelineRow[] = useMemo(
    () => buildRows(tree, expanded),
    [expanded, tree],
  );
  const domain = useMemo(
    () =>
      timelineDomain(
        rows.map((row) => row.span),
        nowMs,
      ),
    [nowMs, rows],
  );
  const groups = useMemo(
    () =>
      buildGroups(rows, (span) => {
        if (isSessionTraceGroupSpan(span)) return true;
        return (
          (span.observationKind === "AGENT" ||
            span.observationKind === "CHAIN") &&
          span.parentSpanId !== ""
        );
      }),
    [rows],
  );

  return (
    <div className="flex min-h-0 min-w-0 flex-1">
      <SpanTreeSidebar
        expanded={expanded}
        heat={heat}
        maxCost={maxCost}
        nowMs={nowMs}
        onSelect={(key) => onSelect(key)}
        onToggle={onToggle}
        recentSpanIds={recentSpanIds}
        rows={rows}
        selectedKey={selectedSpanKey}
      />
      <WaterfallCanvas
        domainDur={domain.dur}
        domainStartMs={domain.startMs}
        groups={groups}
        heat={heat}
        maxCost={maxCost}
        nowMs={nowMs}
        onSelect={onSelect}
        recentSpanIds={recentSpanIds}
        ref={waterfallRef}
        rows={rows}
        running={running}
        selectedKey={selectedSpanKey}
        setHover={setHover}
      />
      <SpanTooltip
        heat={heat}
        hover={hover}
        maxCost={maxCost}
        nodeByKey={nodeByKey}
      />
    </div>
  );
});
