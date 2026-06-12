import { describe, expect, test } from "bun:test";

import {
  buildSessionConversation,
  buildTraceConversation,
} from "../src/mainview/tracing/detail/conversation";
import { buildClientSpanTree, buildSessionSpanTree } from "../src/mainview/tracing/spanTree";
import type { Span, Trace } from "../src/server/telemetry/types";

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

const messages = (items: Array<Record<string, unknown>>) => JSON.stringify(items);

describe("buildTraceConversation", () => {
  test("multi-turn agent loop: dedupes history, groups spans by boundary", () => {
    // Mirrors the prototype's auth-refactor run shape: plan → tool → interim
    // answer, then more work → final answer. The second LLM call resends the
    // full history.
    const spans = [
      span({ observationKind: "AGENT", spanId: "run", startTimeMs: 0, endTimeMs: 1_000, spanName: "agent.run" }),
      span({
        endTimeMs: 200,
        inputMessages: messages([
          { content: "Be helpful.", role: "system" },
          { content: "Refactor the auth module.", role: "user" },
        ]),
        observationKind: "LLM",
        outputMessages: messages([
          { content: "I looked through the module. Working on it.", role: "assistant" },
        ]),
        parentSpanId: "run",
        spanId: "L1",
        startTimeMs: 100,
        costTotal: "0.10",
      }),
      span({
        endTimeMs: 400,
        observationKind: "TOOL",
        parentSpanId: "run",
        spanId: "T1",
        startTimeMs: 300,
      }),
      span({
        endTimeMs: 900,
        inputMessages: messages([
          { content: "Be helpful.", role: "system" },
          { content: "Refactor the auth module.", role: "user" },
          { content: "I looked through the module. Working on it.", role: "assistant" },
        ]),
        observationKind: "LLM",
        outputMessages: messages([
          { content: "Done — suite is green.", role: "assistant" },
        ]),
        parentSpanId: "run",
        spanId: "L2",
        startTimeMs: 500,
        costTotal: "0.25",
      }),
    ];
    const tree = buildClientSpanTree(spans);
    const model = buildTraceConversation(spans, tree);

    expect(model.messages.map((m) => m.role)).toEqual(["user", "agent", "agent"]);
    expect(model.messages[0]!.text).toBe("Refactor the auth module.");
    expect(model.messages[1]!.text).toBe(
      "I looked through the module. Working on it.",
    );
    expect(model.messages[2]!.text).toBe("Done — suite is green.");

    // L1 ends at the first boundary; T1 + L2 fall to the second.
    const firstGroup = model.messages[1]!.group!;
    const secondGroup = model.messages[2]!.group!;
    expect(firstGroup.rows.map((row) => row.span.spanId)).toEqual(["L1"]);
    expect(firstGroup.toolCallCount).toBe(0);
    expect(secondGroup.rows.map((row) => row.span.spanId)).toEqual(["T1", "L2"]);
    expect(secondGroup.toolCallCount).toBe(1);
    expect(secondGroup.costTotal).toBeCloseTo(0.25);
    expect(model.outcome.ok).toBe(true);
    expect(model.outcome.costTotal).toBeCloseTo(0.35);
  });

  test("single LLM call trace", () => {
    const only = span({
      endTimeMs: 300,
      inputMessages: messages([{ content: "Hi", role: "user" }]),
      observationKind: "LLM",
      outputMessages: messages([{ content: "Hello!", role: "assistant" }]),
      spanId: "L1",
      startTimeMs: 100,
    });
    const tree = buildClientSpanTree([only]);
    const model = buildTraceConversation([only], tree);
    expect(model.messages.map((m) => m.role)).toEqual(["user", "agent"]);
    // Sole root with no children keeps its own row in the chip.
    expect(model.messages[1]!.group?.rows.map((r) => r.span.spanId)).toEqual([
      "L1",
    ]);
  });

  test("tool-only trace falls back to a single agent message with all spans", () => {
    const spans = [
      span({ endTimeMs: 500, output: "Copied 3 files.", spanId: "root", startTimeMs: 0 }),
      span({
        endTimeMs: 300,
        observationKind: "TOOL",
        parentSpanId: "root",
        spanId: "T1",
        startTimeMs: 100,
        statusCode: "STATUS_CODE_ERROR",
      }),
    ];
    const tree = buildClientSpanTree(spans);
    const model = buildTraceConversation(spans, tree);
    const agent = model.messages.find((m) => m.role === "agent")!;
    expect(agent.text).toBe("Copied 3 files.");
    expect(agent.group?.rows.map((r) => r.span.spanId)).toEqual(["T1"]);
    expect(agent.group?.failedCount).toBe(1);
    expect(model.outcome.ok).toBe(false);
  });

  test("missing content yields chip-only agent message", () => {
    const spans = [
      span({ spanId: "root", startTimeMs: 0, endTimeMs: 100 }),
      span({ observationKind: "TOOL", parentSpanId: "root", spanId: "T1", startTimeMs: 10, endTimeMs: 90 }),
    ];
    const tree = buildClientSpanTree(spans);
    const model = buildTraceConversation(spans, tree);
    const agent = model.messages.find((m) => m.role === "agent")!;
    expect(agent.text).toBe("");
    expect(agent.group?.rows.length).toBe(1);
  });

  test("nested sub-agent spans get relative depth inside their group", () => {
    const spans = [
      span({ observationKind: "AGENT", spanId: "run", startTimeMs: 0, endTimeMs: 1_000 }),
      span({ observationKind: "AGENT", parentSpanId: "run", spanId: "A1", startTimeMs: 50, endTimeMs: 600 }),
      span({ observationKind: "TOOL", parentSpanId: "A1", spanId: "T1", startTimeMs: 100, endTimeMs: 200 }),
      span({
        endTimeMs: 900,
        inputMessages: messages([{ content: "Go", role: "user" }]),
        observationKind: "LLM",
        outputMessages: messages([{ content: "All done.", role: "assistant" }]),
        parentSpanId: "run",
        spanId: "L1",
        startTimeMs: 700,
      }),
    ];
    const tree = buildClientSpanTree(spans);
    const model = buildTraceConversation(spans, tree);
    const agent = model.messages.find((m) => m.role === "agent")!;
    const depths = Object.fromEntries(
      agent.group!.rows.map((row) => [row.span.spanId, row.depth]),
    );
    expect(depths.A1).toBe(0);
    expect(depths.T1).toBe(1);
    expect(depths.L1).toBe(0);
  });
});

describe("buildSessionConversation", () => {
  function trace(overrides: Partial<Trace>): Trace {
    const startTimeMs = overrides.startTimeMs ?? 0;
    const endTimeMs = overrides.endTimeMs ?? startTimeMs + 100;
    return {
      agentId: "",
      agentName: "",
      cacheReadTokens: null,
      deploymentEnvironment: "",
      durationMs: endTimeMs - startTimeMs,
      durationNs: "0",
      endTime: new Date(endTimeMs).toISOString(),
      endTimeMs,
      hasError: false,
      inputPreview: null,
      llmSpanCount: 1,
      outputPreview: null,
      projectId: "p",
      rootObservationKind: "AGENT",
      rootSpanName: "run",
      serviceName: "",
      serviceVersion: "",
      sessionId: "sess",
      source: "local",
      sourceConnectionId: null,
      sourceConnectionName: null,
      sourceImportJobId: null,
      sourceImportedAt: null,
      sourceImportedAtMs: null,
      sourceTags: [],
      sourceTraceId: null,
      sourceUrl: null,
      spanCount: 1,
      startTime: new Date(startTimeMs).toISOString(),
      startTimeMs,
      totalCost: null,
      totalTokens: null,
      traceId: overrides.traceId ?? "t",
      ...overrides,
    };
  }

  test("each trace becomes a turn in time order", () => {
    const turnSpan = (traceId: string, startTimeMs: number, q: string, a: string) =>
      span({
        endTimeMs: startTimeMs + 100,
        inputMessages: messages([{ content: q, role: "user" }]),
        observationKind: "LLM",
        outputMessages: messages([{ content: a, role: "assistant" }]),
        spanId: `${traceId}-L1`,
        startTimeMs,
        traceId,
      });
    const spans = [
      turnSpan("t1", 0, "First question", "First answer"),
      turnSpan("t2", 1_000, "Second question", "Second answer"),
    ];
    const tree = buildSessionSpanTree(spans, [
      trace({ startTimeMs: 0, traceId: "t1" }),
      trace({ startTimeMs: 1_000, traceId: "t2" }),
    ]);
    const model = buildSessionConversation(tree);
    expect(model.messages.map((m) => [m.role, m.text])).toEqual([
      ["user", "First question"],
      ["agent", "First answer"],
      ["user", "Second question"],
      ["agent", "Second answer"],
    ]);
  });
});
