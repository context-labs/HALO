import { describe, expect, test } from "bun:test";
import type { Span, Trace } from "../src/server/telemetry/types";
import {
  buildClientSpanTree,
  buildSessionSpanTree,
  findFirstInspectableSpan,
  flattenSpanTree,
  isSessionTraceGroupSpan,
  isSyntheticSpan,
} from "../src/mainview/tracing/spanTree";

describe("client span tree", () => {
  test("groups orphan children under a pending parent span", () => {
    const child = span({
      agentId: "gator-flue-agent",
      agentName: "Gator Flue Agent",
      parentSpanId: "root-span",
      spanId: "llm-span",
      spanName: "pi-ai.anthropic.turn",
    });

    const tree = buildClientSpanTree([child]);
    const root = tree[0]!;

    expect(tree).toHaveLength(1);
    expect(root.span.spanId).toBe("root-span");
    expect(root.span.spanName).toBe("gator.flue_agent.run");
    expect(root.span.observationKind).toBe("AGENT");
    expect(isSyntheticSpan(root.span)).toBe(true);
    expect(root.children.map((node) => node.span.spanId)).toEqual(["llm-span"]);
  });

  test("uses the real parent span once it arrives", () => {
    const root = span({
      observationKind: "AGENT",
      parentSpanId: "",
      spanId: "root-span",
      spanName: "gator.flue_agent.run",
    });
    const child = span({
      parentSpanId: "root-span",
      spanId: "llm-span",
      spanName: "pi-ai.anthropic.turn",
      startTimeMs: 1_100,
    });

    const tree = buildClientSpanTree([child, root]);
    const rootNode = tree[0]!;

    expect(tree).toHaveLength(1);
    expect(rootNode.span).toBe(root);
    expect(isSyntheticSpan(rootNode.span)).toBe(false);
    expect(rootNode.children.map((node) => node.span.spanId)).toEqual(["llm-span"]);
  });

  test("flattens the display tree in preorder", () => {
    const child = span({
      parentSpanId: "root-span",
      spanId: "tool-span",
      spanName: "gator_todos_list.tool",
    });

    const flattened = flattenSpanTree(buildClientSpanTree([child]));

    expect(flattened.map((item) => item.spanId)).toEqual(["root-span", "tool-span"]);
  });

  test("groups session spans by trace without making turn wrappers inspectable", () => {
    const root = span({
      observationKind: "AGENT",
      parentSpanId: "",
      spanId: "root-span",
      spanName: "gator.flue_agent.run",
      traceId: "trace-a",
    });
    const child = span({
      parentSpanId: "root-span",
      spanId: "llm-span",
      spanName: "pi-ai.anthropic.turn",
      startTimeMs: 1_100,
      traceId: "trace-a",
    });

    const tree = buildSessionSpanTree([root, child], [
      trace({ rootSpanName: "gator.flue_agent.run", traceId: "trace-a" }),
    ]);

    expect(tree).toHaveLength(1);
    expect(tree[0]?.span.spanName).toBe("Turn 1");
    expect(isSessionTraceGroupSpan(tree[0]?.span)).toBe(true);
    expect(tree[0]?.span.spanAttributes["halo.synthetic.trace_name"]).toBe(
      "gator.flue_agent.run",
    );
    expect(findFirstInspectableSpan(tree)).toBe(root);
  });
});

function trace(overrides: Partial<Trace> = {}): Trace {
  const startTimeMs = overrides.startTimeMs ?? 1_000;
  const endTimeMs = overrides.endTimeMs ?? startTimeMs + 100;
  return {
    agentId: "",
    agentName: "Gator Flue Agent",
    cacheReadTokens: null,
    deploymentEnvironment: "",
    durationMs: endTimeMs - startTimeMs,
    durationNs: String((endTimeMs - startTimeMs) * 1_000_000),
    endTime: new Date(endTimeMs).toISOString(),
    endTimeMs,
    hasError: false,
    llmSpanCount: 1,
    projectId: "local-project",
    rootObservationKind: "AGENT",
    rootSpanName: "gator.flue_agent.run",
    serviceName: "gator-agent",
    serviceVersion: "0.0.3",
    sessionId: "session",
    source: "local",
    sourceConnectionId: null,
    sourceConnectionName: null,
    sourceImportJobId: null,
    sourceImportedAt: null,
    sourceImportedAtMs: null,
    sourceTags: [],
    sourceTraceId: null,
    sourceUrl: null,
    spanCount: 2,
    startTime: new Date(startTimeMs).toISOString(),
    startTimeMs,
    totalCost: null,
    totalTokens: null,
    traceId: "trace",
    ...overrides,
  };
}

function span(overrides: Partial<Span> = {}): Span {
  const startTimeMs = overrides.startTimeMs ?? 1_000;
  const endTimeMs = overrides.endTimeMs ?? startTimeMs + 100;
  const durationMs = endTimeMs - startTimeMs;

  return {
    agentId: "",
    agentName: "",
    apiKeyId: "local-dev",
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
    durationMs,
    durationNs: String(durationMs * 1_000_000),
    endTime: new Date(endTimeMs).toISOString(),
    endTimeMs,
    events: [],
    id: 1,
    ingestedAt: new Date(endTimeMs).toISOString(),
    input: null,
    inputMessages: null,
    inputTokens: null,
    links: [],
    llmModelName: "",
    llmProvider: "",
    llmResponseModel: "",
    observationKind: "LLM",
    output: null,
    outputMessages: null,
    outputTokens: null,
    parentSpanId: "",
    projectId: "local-project",
    reasoningTokens: null,
    resourceAttributes: {},
    resourceAttributesDouble: {},
    resourceAttributesInt: {},
    retrievalDocuments: null,
    scopeName: "",
    scopeVersion: "",
    serviceName: "gator-agent",
    serviceVersion: "0.0.3",
    sessionId: null,
    spanAttributes: {},
    spanAttributesDouble: {},
    spanAttributesInt: {},
    spanId: "span",
    spanKind: "SPAN_KIND_INTERNAL",
    spanName: "span",
    startTime: new Date(startTimeMs).toISOString(),
    startTimeMs,
    statusCode: "STATUS_CODE_OK",
    statusMessage: "",
    teamId: "local-team",
    totalTokens: null,
    traceId: "trace",
    traceState: "",
    userId: null,
    ...overrides,
  };
}
