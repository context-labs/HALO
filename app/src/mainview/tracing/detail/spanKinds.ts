import { useEffect, useState } from "react";

import type { ObservationKind, Span } from "../../../server/telemetry/types";

/**
 * Concrete RGB strings resolved from the --color-trace-* CSS variables so the
 * heatmap lerp can run on numbers while the source of truth stays in CSS.
 */
export type TraceColors = {
  agent: string;
  error: string;
  group: string;
  heat0: string;
  heat1: string;
  heat2: string;
  heat3: string;
  heatDim: string;
  llm: string;
  marker: string;
  other: string;
  tool: string;
};

const FALLBACK_COLORS: TraceColors = {
  agent: "rgb(23, 178, 106)",
  error: "rgb(240, 68, 56)",
  group: "rgb(83, 177, 253)",
  heat0: "rgb(75, 85, 101)",
  heat1: "rgb(253, 176, 34)",
  heat2: "rgb(247, 144, 9)",
  heat3: "rgb(240, 68, 56)",
  heatDim: "rgb(54, 65, 82)",
  llm: "rgb(46, 144, 250)",
  marker: "rgb(253, 176, 34)",
  other: "rgb(75, 85, 101)",
  tool: "rgb(253, 176, 34)",
};

const TOKEN_BY_KEY: Record<keyof TraceColors, string> = {
  agent: "--color-trace-agent",
  error: "--color-detail-failure",
  group: "--color-trace-group",
  heat0: "--color-trace-heat-0",
  heat1: "--color-trace-heat-1",
  heat2: "--color-trace-heat-2",
  heat3: "--color-trace-heat-3",
  heatDim: "--color-trace-heat-dim",
  llm: "--color-trace-llm",
  marker: "--color-trace-marker",
  other: "--color-trace-other",
  tool: "--color-trace-tool",
};

function resolveTraceColors(): TraceColors {
  if (typeof document === "undefined") return FALLBACK_COLORS;
  // The theme class ("dark") lives on <body>, so resolve there.
  const style = getComputedStyle(document.body);
  const resolved = {} as TraceColors;
  for (const key of Object.keys(TOKEN_BY_KEY) as (keyof TraceColors)[]) {
    const value = style.getPropertyValue(TOKEN_BY_KEY[key]).trim();
    resolved[key] = value || FALLBACK_COLORS[key];
  }
  return resolved;
}

/** Trace colors, re-resolved whenever the theme class flips. */
export function useTraceColors(): TraceColors {
  const [colors, setColors] = useState<TraceColors>(resolveTraceColors);
  useEffect(() => {
    setColors(resolveTraceColors());
    const observer = new MutationObserver(() => setColors(resolveTraceColors()));
    observer.observe(document.body, {
      attributeFilter: ["class"],
      attributes: true,
    });
    observer.observe(document.documentElement, {
      attributeFilter: ["class"],
      attributes: true,
    });
    return () => observer.disconnect();
  }, []);
  return colors;
}

export function colorForKind(kind: ObservationKind, colors: TraceColors): string {
  switch (kind) {
    case "LLM":
      return colors.llm;
    case "TOOL":
      return colors.tool;
    case "AGENT":
    case "CHAIN":
      return colors.agent;
    case "RETRIEVER":
    case "EMBEDDING":
    case "RERANKER":
      return colors.group;
    case "GUARDRAIL":
    case "EVALUATOR":
    case "PROMPT":
      return colors.marker;
    default:
      return colors.other;
  }
}

/**
 * Bar/dot color for a span. With the cost heatmap on, LLM spans get the
 * cool→ember ramp by spend and everything else dims.
 */
export function spanColor(
  span: Pick<Span, "costTotal" | "observationKind">,
  colors: TraceColors,
  heat: boolean,
  maxCost: number,
): string {
  if (heat) {
    if (span.observationKind !== "LLM") return colors.heatDim;
    const cost = numericCost(span.costTotal);
    if (maxCost <= 0 || cost <= 0) return colors.heat0;
    return costColor(Math.sqrt(cost / maxCost), colors);
  }
  return colorForKind(span.observationKind, colors);
}

/** v in 0..1 → 4-stop lerp across the heat ramp. */
export function costColor(v: number, colors: TraceColors): string {
  const stops: Array<[number, string]> = [
    [0, colors.heat0],
    [0.45, colors.heat1],
    [0.75, colors.heat2],
    [1, colors.heat3],
  ];
  const clamped = Math.max(0, Math.min(1, v));
  for (let i = 1; i < stops.length; i += 1) {
    if (clamped <= stops[i]![0]) {
      const [p0, c0] = stops[i - 1]!;
      const [p1, c1] = stops[i]!;
      return rgbLerp(c0, c1, (clamped - p0) / (p1 - p0));
    }
  }
  return stops[stops.length - 1]![1];
}

export function numericCost(costTotal: string | null): number {
  if (costTotal == null) return 0;
  const value = Number(costTotal);
  return Number.isFinite(value) ? value : 0;
}

export function maxLlmCost(spans: Pick<Span, "costTotal" | "observationKind">[]): number {
  let max = 0;
  for (const span of spans) {
    if (span.observationKind !== "LLM") continue;
    const cost = numericCost(span.costTotal);
    if (cost > max) max = cost;
  }
  return max;
}

function rgbLerp(a: string, b: string, t: number): string {
  const pa = parseRgb(a);
  const pb = parseRgb(b);
  if (!pa || !pb) return a;
  const mix = pa.map((v, i) => Math.round(v + (pb[i]! - v) * t));
  return `rgb(${mix[0]}, ${mix[1]}, ${mix[2]})`;
}

function parseRgb(value: string): [number, number, number] | null {
  const match = value.match(/rgba?\(\s*(\d+)[\s,]+(\d+)[\s,]+(\d+)/);
  if (!match) return null;
  return [Number(match[1]), Number(match[2]), Number(match[3])];
}
