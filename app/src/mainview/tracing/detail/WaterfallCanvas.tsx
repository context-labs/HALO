import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { cn } from "~/lib/ui";
import { formatDuration } from "~/lib/format";
import type { Span } from "../../../server/telemetry/types";
import { isSessionTraceGroupSpan, isSyntheticSpan } from "../spanTree";
import { spanColor, useTraceColors, type TraceColors } from "./spanKinds";
import {
  clampView,
  fitView,
  fmtTick,
  niceTicks,
  spanEndMs,
  spanZoomView,
  zoomAtAnchor,
  type TimelineGroup,
  type TimelineRow,
  type TimelineView,
} from "./timelineMath";
import type { SpanHover } from "./SpanTooltip";

export type WaterfallHandle = {
  zoomIn: () => void;
  zoomOut: () => void;
  zoomToFit: () => void;
  zoomToSpan: (key: string) => void;
};

const ROW_HEIGHT = 28;
const BAR_HEIGHT = ROW_HEIGHT - 13;
const LABEL_MIN_WIDTH = 72;
const OVERSCAN_ROWS = 10;

export const WaterfallCanvas = forwardRef<
  WaterfallHandle,
  {
    domainDur: number;
    domainStartMs: number;
    groups: TimelineGroup[];
    heat: boolean;
    maxCost: number;
    nowMs: number;
    onSelect: (key: string | null) => void;
    recentSpanIds: Set<string>;
    rows: TimelineRow[];
    running: boolean;
    selectedKey: string | null;
    setHover: (hover: SpanHover | null) => void;
  }
>(function WaterfallCanvas(
  {
    domainDur,
    domainStartMs,
    groups,
    heat,
    maxCost,
    nowMs,
    onSelect,
    recentSpanIds,
    rows,
    running,
    selectedKey,
    setHover,
  },
  ref,
) {
  const colors = useTraceColors();
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(900);
  const [view, setView] = useState<TimelineView>(() => fitView(domainDur));
  const viewRef = useRef(view);
  viewRef.current = view;
  const durRef = useRef(domainDur);
  durRef.current = domainDur;
  // Track the domain automatically until the user zooms or pans.
  const autoFitRef = useRef(true);
  const dragRef = useRef<{
    moved: boolean;
    startView: TimelineView;
    x: number;
  } | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportH, setViewportH] = useState(600);

  const rel = useCallback((ms: number) => ms - domainStartMs, [domainStartMs]);

  useLayoutEffect(() => {
    const wrap = wrapRef.current;
    const scroller = scrollRef.current;
    if (!wrap || !scroller) return;
    const observer = new ResizeObserver(() => {
      setWidth(wrap.clientWidth);
      setViewportH(scroller.clientHeight);
    });
    observer.observe(wrap);
    observer.observe(scroller);
    setWidth(wrap.clientWidth);
    setViewportH(scroller.clientHeight);
    return () => observer.disconnect();
  }, []);

  // While auto-fitting, follow the (possibly growing) domain.
  if (autoFitRef.current) {
    const fitted = fitView(domainDur);
    if (fitted.t0 !== view.t0 || fitted.t1 !== view.t1) {
      setView(fitted);
    }
  }

  const userSetView = useCallback((next: TimelineView) => {
    autoFitRef.current = false;
    setView(next);
  }, []);

  const zoomAt = useCallback(
    (factor: number, anchorT: number) => {
      userSetView(zoomAtAnchor(viewRef.current, factor, anchorT, durRef.current));
    },
    [userSetView],
  );

  useImperativeHandle(
    ref,
    () => ({
      zoomIn() {
        const current = viewRef.current;
        zoomAt(1 / 1.45, (current.t0 + current.t1) / 2);
      },
      zoomOut() {
        const current = viewRef.current;
        zoomAt(1.45, (current.t0 + current.t1) / 2);
      },
      zoomToFit() {
        autoFitRef.current = true;
        setView(fitView(durRef.current));
      },
      zoomToSpan(key: string) {
        const row = rows.find((item) => item.key === key);
        if (!row) return;
        const start = rel(row.span.startTimeMs);
        const end = rel(spanEndMs(row.span, nowMs));
        userSetView(spanZoomView(start, end, durRef.current));
      },
    }),
    [nowMs, rel, rows, userSetView, zoomAt],
  );

  // ⌘/ctrl + wheel zooms at the cursor; horizontal-dominant wheel pans.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onWheel = (event: WheelEvent) => {
      const current = viewRef.current;
      const viewWidth = current.t1 - current.t0;
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        const rect = wrapRef.current?.getBoundingClientRect();
        if (!rect || rect.width === 0) return;
        const anchorT =
          current.t0 + ((event.clientX - rect.left) / rect.width) * viewWidth;
        zoomAt(Math.exp(event.deltaY * 0.0022), anchorT);
      } else if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
        event.preventDefault();
        const dt = (event.deltaX / el.clientWidth) * viewWidth;
        userSetView(
          clampView(current.t0 + dt, current.t1 + dt, durRef.current),
        );
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [userSetView, zoomAt]);

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    dragRef.current = {
      moved: false,
      startView: viewRef.current,
      x: event.clientX,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    const dx = event.clientX - drag.x;
    if (Math.abs(dx) > 3) drag.moved = true;
    if (!drag.moved) return;
    const viewWidth = drag.startView.t1 - drag.startView.t0;
    const dt = (-dx / Math.max(1, width)) * viewWidth;
    userSetView(
      clampView(drag.startView.t0 + dt, drag.startView.t1 + dt, durRef.current),
    );
  };

  const onPointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    dragRef.current = null;
    event.currentTarget.releasePointerCapture(event.pointerId);
    if (drag && !drag.moved) onSelect(null);
  };

  const px = useCallback(
    (t: number) => ((t - view.t0) / (view.t1 - view.t0)) * width,
    [view.t0, view.t1, width],
  );

  const ticks = useMemo(
    () => niceTicks(view.t0, view.t1, width),
    [view.t0, view.t1, width],
  );

  const contentH = rows.length * ROW_HEIGHT + 10;
  const firstRow = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN_ROWS);
  const lastRow = Math.min(
    rows.length - 1,
    Math.ceil((scrollTop + viewportH) / ROW_HEIGHT) + OVERSCAN_ROWS,
  );

  return (
    <div
      className="flex min-h-0 min-w-0 flex-1 flex-col bg-background"
      ref={wrapRef}
    >
      {/* time axis */}
      <div className="relative h-[30px] flex-none overflow-hidden border-b border-subtle">
        {ticks.map((tick) => (
          <div
            className="absolute bottom-0 top-0 flex items-center"
            key={tick}
            style={{ left: px(tick) }}
          >
            <div className="absolute bottom-0 top-0 w-px bg-border/70" />
            <span className="select-none pl-1.5 font-geist-mono text-[11px] text-muted-foreground">
              {fmtTick(tick)}
            </span>
          </div>
        ))}
      </div>

      <div
        className={cn(
          "relative min-h-0 flex-1 overflow-y-auto overflow-x-hidden",
          dragRef.current?.moved ? "cursor-grabbing" : "cursor-default",
        )}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
        ref={scrollRef}
      >
        <div className="relative overflow-hidden" style={{ height: contentH }}>
          {/* gridlines */}
          {ticks.map((tick) => (
            <div
              className="pointer-events-none absolute bottom-0 top-0 w-px bg-border/40"
              key={`grid-${tick}`}
              style={{ left: px(tick) }}
            />
          ))}

          {/* group outlines */}
          {groups
            .filter((group) => group.last >= firstRow && group.first <= lastRow)
            .map((group) => {
              const start = px(rel(group.span.startTimeMs));
              const end = px(rel(spanEndMs(group.span, nowMs)));
              return (
                <div
                  className="pointer-events-none absolute rounded-md border border-trace-group/70 bg-trace-group/5"
                  key={`group-${group.span.spanId}`}
                  style={{
                    height: (group.last - group.first + 1) * ROW_HEIGHT - 2,
                    left: start - 4,
                    top: group.first * ROW_HEIGHT + 1,
                    width: Math.max(8, end - start + 8),
                  }}
                />
              );
            })}

          {/* span bars */}
          {rows.slice(firstRow, lastRow + 1).map((row, sliceIndex) => {
            const index = firstRow + sliceIndex;
            return (
              <WaterfallRow
                colors={colors}
                heat={heat}
                index={index}
                key={row.key}
                maxCost={maxCost}
                nowMs={nowMs}
                onSelect={onSelect}
                px={px}
                recent={recentSpanIds.has(row.key)}
                rel={rel}
                row={row}
                running={running}
                selected={row.key === selectedKey}
                setHover={setHover}
                zoomToSpan={(span) => {
                  const start = rel(span.startTimeMs);
                  const end = rel(spanEndMs(span, nowMs));
                  userSetView(spanZoomView(start, end, durRef.current));
                }}
              />
            );
          })}
        </div>
      </div>

      <div className="flex flex-none select-none gap-4 border-t border-subtle px-4 py-1.5 text-[11px] text-muted-foreground">
        <span>drag to pan</span>
        <span>⌘ + scroll to zoom</span>
        <span>double-click a span to zoom to it</span>
        <span>click background to deselect</span>
      </div>
    </div>
  );
});

function WaterfallRow({
  colors,
  heat,
  index,
  maxCost,
  nowMs,
  onSelect,
  px,
  recent,
  rel,
  row,
  running,
  selected,
  setHover,
  zoomToSpan,
}: {
  colors: TraceColors;
  heat: boolean;
  index: number;
  maxCost: number;
  nowMs: number;
  onSelect: (key: string) => void;
  px: (t: number) => number;
  recent: boolean;
  rel: (ms: number) => number;
  row: TimelineRow;
  running: boolean;
  selected: boolean;
  setHover: (hover: SpanHover | null) => void;
  zoomToSpan: (span: Span) => void;
}) {
  const span = row.span;
  const isGroupHeader = isSessionTraceGroupSpan(span);
  const synthetic = isSyntheticSpan(span) && !isGroupHeader;
  const inFlight = running && span.endTimeMs <= span.startTimeMs;
  const x0 = px(rel(span.startTimeMs));
  const x1 = px(rel(spanEndMs(span, nowMs)));
  const barWidth = Math.max(3, x1 - x0);
  const color = spanColor(span, colors, heat, maxCost);
  const error = span.statusCode === "STATUS_CODE_ERROR";
  const showLabel = barWidth > LABEL_MIN_WIDTH;
  const hasEvents = span.events.length > 0;

  return (
    <div
      className="absolute left-0 right-0"
      style={{ height: ROW_HEIGHT, top: index * ROW_HEIGHT }}
    >
      <div
        className={cn(
          "absolute flex cursor-pointer items-center overflow-hidden rounded transition-shadow duration-100",
          synthetic && "border border-dashed border-current bg-transparent",
          recent && "live-timeline-pulse",
        )}
        data-span={row.key}
        onClick={(event) => {
          event.stopPropagation();
          onSelect(row.key);
        }}
        onDoubleClick={(event) => {
          event.stopPropagation();
          zoomToSpan(span);
        }}
        onMouseEnter={(event) =>
          setHover({ span, x: event.clientX, y: event.clientY })
        }
        onMouseLeave={() => setHover(null)}
        onMouseMove={(event) =>
          setHover({ span, x: event.clientX, y: event.clientY })
        }
        onPointerDown={(event) => event.stopPropagation()}
        style={{
          background: synthetic ? "transparent" : color,
          boxShadow: selected
            ? `0 0 0 1.5px var(--color-foreground), 0 0 14px ${color}`
            : error
              ? "inset 0 0 0 1.5px var(--color-detail-failure)"
              : undefined,
          color: synthetic ? color : undefined,
          height: BAR_HEIGHT,
          left: x0,
          opacity: heat && span.observationKind !== "LLM" ? 0.55 : inFlight ? 0.8 : 1,
          top: (ROW_HEIGHT - BAR_HEIGHT) / 2 - (hasEvents ? 2 : 0),
          width: barWidth,
        }}
      >
        {showLabel ? (
          <span
            className="select-none whitespace-nowrap px-2 text-[11px] font-semibold text-white [text-shadow:0_1px_1px_rgba(0,0,0,0.4)]"
            style={synthetic ? { color, textShadow: "none" } : undefined}
          >
            {span.spanName}
            <span className="ml-1.5 font-geist-mono text-[10px] font-normal opacity-75">
              {inFlight
                ? "running"
                : formatDuration(spanEndMs(span, nowMs) - span.startTimeMs)}
            </span>
          </span>
        ) : null}
      </div>

      {/* OTel event markers under the bar */}
      {span.events.map((event, eventIndex) => {
        const eventMs = Date.parse(event.timestamp);
        if (!Number.isFinite(eventMs)) return null;
        const isErrorEvent = /error|exception/i.test(event.name);
        return (
          <div
            className="absolute rounded-[2.5px]"
            key={`event-${eventIndex}`}
            style={{
              background: isErrorEvent ? colors.error : colors.marker,
              boxShadow: "0 0 0 1.5px var(--color-background)",
              height: 7,
              left: px(rel(eventMs)) - 2,
              top: ROW_HEIGHT - 8.5,
              width: 4.5,
            }}
            title={event.name}
          />
        );
      })}
    </div>
  );
}
