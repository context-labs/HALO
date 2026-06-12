import { describe, expect, test } from "bun:test";

import {
  buildGroups,
  buildRows,
  clampView,
  fitView,
  fmtTick,
  niceTicks,
  spanZoomView,
  timelineDomain,
  zoomAtAnchor,
} from "../src/mainview/tracing/detail/timelineMath";
import type { Span, SpanNode } from "../src/server/telemetry/types";

function span(overrides: Partial<Span>): Span {
  const startTimeMs = overrides.startTimeMs ?? 0;
  const endTimeMs = overrides.endTimeMs ?? startTimeMs + 100;
  return {
    agentId: "",
    agentName: "",
    apiKeyId: "",
    cacheReadTokens: null,
    cacheWriteTokens: null,
    chatId: "",
    conversationId: "",
    costCacheRead: null,
    costCacheWrite: null,
    costInput: null,
    costOutput: null,
    costReasoning: null,
    costTotal: null,
    deploymentEnvironment: "",
    durationMs: endTimeMs - startTimeMs,
    durationNs: String((endTimeMs - startTimeMs) * 1_000_000),
    endTime: new Date(endTimeMs).toISOString(),
    endTimeMs,
    events: [],
    id: 1,
    ingestedAt: new Date(0).toISOString(),
    input: null,
    inputMessages: null,
    inputTokens: null,
    links: [],
    llmModelName: "",
    llmProvider: "",
    llmResponseModel: "",
    observationKind: "SPAN",
    output: null,
    outputMessages: null,
    outputTokens: null,
    parentSpanId: "",
    projectId: "p",
    reasoningTokens: null,
    resourceAttributes: {},
    resourceAttributesDouble: {},
    resourceAttributesInt: {},
    retrievalDocuments: null,
    scopeName: "",
    scopeVersion: "",
    serviceName: "",
    serviceVersion: "",
    sessionId: null,
    spanAttributes: {},
    spanAttributesDouble: {},
    spanAttributesInt: {},
    spanId: "s",
    spanKind: "SPAN_KIND_INTERNAL",
    spanName: "span",
    startTime: new Date(startTimeMs).toISOString(),
    startTimeMs,
    statusCode: "STATUS_CODE_OK",
    statusMessage: "",
    teamId: "",
    totalTokens: null,
    traceId: "t",
    traceState: "",
    userId: null,
    ...overrides,
  };
}

describe("clampView / zoomAtAnchor", () => {
  const dur = 10_000;

  test("clamps width and pan bounds", () => {
    const tooWide = clampView(-100_000, 100_000, dur);
    expect(tooWide.t1 - tooWide.t0).toBeCloseTo(dur * 1.8);
    const tooNarrow = clampView(500, 500.0001, dur);
    expect(tooNarrow.t1 - tooNarrow.t0).toBeCloseTo(Math.max(1, dur * 0.002));
    const panned = clampView(50_000, 51_000, dur);
    expect(panned.t1).toBeLessThanOrEqual(dur * 1.25 + 1e-6);
  });

  test("zoom keeps the anchor's screen position fixed", () => {
    const view = { t0: 0, t1: 10_000 };
    const anchor = 2_500; // 25% across
    const zoomed = zoomAtAnchor(view, 0.5, anchor, dur);
    const ratio = (anchor - zoomed.t0) / (zoomed.t1 - zoomed.t0);
    expect(ratio).toBeCloseTo(0.25, 5);
  });

  test("fit + span zoom stay within clamp bounds", () => {
    const fit = fitView(dur);
    expect(fit.t0).toBeLessThan(0);
    expect(fit.t1).toBeGreaterThan(dur);
    const zoomed = spanZoomView(100, 100, dur); // zero-duration span
    expect(zoomed.t1 - zoomed.t0).toBeGreaterThan(0);
  });
});

describe("niceTicks / fmtTick", () => {
  test("keeps tick density near one per ~110px", () => {
    const ticks = niceTicks(0, 10_000, 900);
    expect(ticks.length).toBeGreaterThanOrEqual(4);
    expect(ticks.length).toBeLessThanOrEqual(11);
  });

  test("works on sub-second domains", () => {
    const ticks = niceTicks(0, 50, 800);
    expect(ticks.length).toBeGreaterThanOrEqual(4);
    expect(ticks[1]! - ticks[0]!).toBeLessThanOrEqual(10);
  });

  test("formats units", () => {
    expect(fmtTick(850)).toBe("850ms");
    expect(fmtTick(1_500)).toBe("1.5s");
    expect(fmtTick(120_000)).toBe("2m");
    expect(fmtTick(5_400_000)).toBe("1.5h");
    expect(fmtTick(2 * 86_400_000)).toBe("2d");
    expect(fmtTick(-500)).toBe("-500ms");
  });
});

describe("timelineDomain", () => {
  test("derives span bounds and draws in-flight spans to now", () => {
    const spans = [
      span({ endTimeMs: 1_500, spanId: "a", startTimeMs: 1_000 }),
      span({ endTimeMs: 1_200, spanId: "b", startTimeMs: 1_200 }), // in-flight
    ];
    const domain = timelineDomain(spans, 3_000);
    expect(domain.startMs).toBe(1_000);
    expect(domain.dur).toBe(2_000);
  });

  test("empty input yields a 1ms domain", () => {
    expect(timelineDomain([]).dur).toBe(1);
  });
});

describe("buildRows / buildGroups", () => {
  const root = span({ observationKind: "AGENT", spanId: "run", spanName: "run" });
  const sub = span({
    observationKind: "AGENT",
    parentSpanId: "run",
    spanId: "sub",
    startTimeMs: 10,
  });
  const llm = span({
    observationKind: "LLM",
    parentSpanId: "sub",
    spanId: "llm",
    startTimeMs: 20,
  });
  const tree: SpanNode[] = [
    {
      children: [{ children: [{ children: [], span: llm }], span: sub }],
      span: root,
    },
  ];

  test("collapsed nodes hide descendants", () => {
    const all = buildRows(tree, new Set(["t:run", "t:sub"]));
    expect(all.map((row) => row.span.spanId)).toEqual(["run", "sub", "llm"]);
    const collapsed = buildRows(tree, new Set(["t:run"]));
    expect(collapsed.map((row) => row.span.spanId)).toEqual(["run", "sub"]);
  });

  test("groups outline expanded sub-agents, excluding the root", () => {
    const rows = buildRows(tree, new Set(["t:run", "t:sub"]));
    const groups = buildGroups(
      rows,
      (item) => item.observationKind === "AGENT" && item.parentSpanId !== "",
    );
    expect(groups).toHaveLength(1);
    expect(groups[0]!.span.spanId).toBe("sub");
    expect(groups[0]!.first).toBe(2);
    expect(groups[0]!.last).toBe(2);
  });
});
