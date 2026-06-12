import type { Span, SpanNode } from "../../../server/telemetry/types";
import { spanKey } from "./spanUtils";

/** Visible time window, in ms relative to the trace domain start. */
export type TimelineView = { t0: number; t1: number };

export type TimelineRow = {
  depth: number;
  hasChildren: boolean;
  key: string;
  span: Span;
};

export type TimelineGroup = {
  /** Row indices of the first/last descendant inside the outline box. */
  first: number;
  last: number;
  span: Span;
};

export type TimelineDomain = {
  /** Total domain length in ms (≥ 1). */
  dur: number;
  /** Absolute ms timestamp the relative scale is anchored to. */
  startMs: number;
};

/** Relative end for a span, drawing in-flight spans out to `nowMs`. */
export function spanEndMs(span: Span, nowMs?: number): number {
  if (span.endTimeMs > span.startTimeMs) return span.endTimeMs;
  return nowMs ?? span.startTimeMs;
}

export function timelineDomain(spans: Span[], nowMs?: number): TimelineDomain {
  if (spans.length === 0) return { dur: 1, startMs: 0 };
  let startMs = Infinity;
  let endMs = -Infinity;
  for (const span of spans) {
    if (span.startTimeMs < startMs) startMs = span.startTimeMs;
    const end = spanEndMs(span, nowMs);
    if (end > endMs) endMs = end;
  }
  return { dur: Math.max(1, endMs - startMs), startMs };
}

/**
 * Clamp a requested view window: keep zoom between 0.2% and 180% of the
 * domain, and panning within a quarter-domain margin either side.
 */
export function clampView(t0: number, t1: number, dur: number): TimelineView {
  const minWidth = Math.max(1, dur * 0.002);
  const maxWidth = dur * 1.8;
  const width = Math.max(minWidth, Math.min(maxWidth, t1 - t0));
  const lo = -dur * 0.25;
  const hi = dur * 1.25;
  const clamped = Math.max(lo, Math.min(hi - width, t0));
  return { t0: clamped, t1: clamped + width };
}

/** Zoom keeping `anchorT` fixed on screen. factor < 1 zooms in. */
export function zoomAtAnchor(
  view: TimelineView,
  factor: number,
  anchorT: number,
  dur: number,
): TimelineView {
  const t0 = anchorT - (anchorT - view.t0) * factor;
  const t1 = anchorT + (view.t1 - anchorT) * factor;
  return clampView(t0, t1, dur);
}

/** The default fit window: a sliver of margin either side of the domain. */
export function fitView(dur: number): TimelineView {
  return clampView(-dur * 0.025, dur * 1.06, dur);
}

/** Window that frames one span with proportional padding. */
export function spanZoomView(
  startT: number,
  endT: number,
  dur: number,
): TimelineView {
  const pad = (endT - startT) * 0.15 + dur * 0.01;
  return clampView(startT - pad, endT + pad, dur);
}

/**
 * Tick positions at a "nice" step (1/2/2.5/5 × 10^k ms) targeting roughly one
 * label per 110px, clamped to 4–10 labels.
 */
export function niceTicks(t0: number, t1: number, width: number): number[] {
  const span = Math.max(1e-6, t1 - t0);
  const target = Math.max(4, Math.min(10, Math.floor(width / 110)));
  let step = Infinity;
  outer: for (let k = 0; k <= 8; k += 1) {
    for (const base of [1, 2, 2.5, 5]) {
      const candidate = base * 10 ** k;
      if (span / candidate <= target) {
        step = candidate;
        break outer;
      }
    }
  }
  if (!Number.isFinite(step)) step = 5 * 10 ** 8;
  const ticks: number[] = [];
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) ticks.push(t);
  return ticks;
}

/** "850ms" / "1.5s" / "2m" / "3.5h" / "2d" axis labels. */
export function fmtTick(ms: number): string {
  const sign = ms < 0 ? "-" : "";
  const abs = Math.abs(ms);
  const unit = (value: number, suffix: string) => {
    const rounded = Math.round(value * 10) / 10;
    return `${sign}${Number.isInteger(rounded) ? rounded : rounded.toFixed(1)}${suffix}`;
  };
  if (abs >= 86_400_000) return unit(abs / 86_400_000, "d");
  if (abs >= 3_600_000) return unit(abs / 3_600_000, "h");
  if (abs >= 60_000) return unit(abs / 60_000, "m");
  if (abs >= 1_000) return unit(abs / 1_000, "s");
  return `${sign}${Math.round(abs)}ms`;
}

/**
 * Flatten the display tree into rows, descending only into expanded nodes.
 */
export function buildRows(
  tree: SpanNode[],
  expanded: ReadonlySet<string>,
): TimelineRow[] {
  const rows: TimelineRow[] = [];
  const walk = (node: SpanNode, depth: number) => {
    const key = spanKey(node.span);
    rows.push({
      depth,
      hasChildren: node.children.length > 0,
      key,
      span: node.span,
    });
    if (node.children.length > 0 && expanded.has(key)) {
      node.children.forEach((child) => walk(child, depth + 1));
    }
  };
  tree.forEach((node) => walk(node, 0));
  return rows;
}

/**
 * Outline boxes around the visible descendants of expanded group rows
 * (sub-agents, chains, session turns). The predicate decides which spans
 * group — callers exclude the trace root, where a box is noise.
 */
export function buildGroups(
  rows: TimelineRow[],
  isGroupSpan: (span: Span) => boolean,
): TimelineGroup[] {
  const groups: TimelineGroup[] = [];
  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i]!;
    if (!row.hasChildren || !isGroupSpan(row.span)) continue;
    let last = i;
    for (let j = i + 1; j < rows.length && rows[j]!.depth > row.depth; j += 1) {
      last = j;
    }
    if (last > i) groups.push({ first: i + 1, last, span: row.span });
  }
  return groups;
}
