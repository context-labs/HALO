import type { Span, SpanNode, Trace } from "../../server/telemetry/types";

const SYNTHETIC_KIND_KEY = "halo.synthetic";
const SYNTHETIC_PARENT_KIND = "pending_parent";
export const SYNTHETIC_SESSION_TRACE_KIND = "session_trace";

export function buildClientSpanTree(spans: Span[]): SpanNode[] {
  const nodes = new Map<string, SpanNode>();
  const missingParentChildren = new Map<string, Span[]>();
  const roots: SpanNode[] = [];

  for (const span of spans) {
    nodes.set(span.spanId, { children: [], span });
  }

  for (const span of spans) {
    if (span.parentSpanId && !nodes.has(span.parentSpanId)) {
      const children = missingParentChildren.get(span.parentSpanId) ?? [];
      children.push(span);
      missingParentChildren.set(span.parentSpanId, children);
    }
  }

  for (const [parentSpanId, children] of missingParentChildren) {
    nodes.set(parentSpanId, {
      children: [],
      span: makePendingParentSpan(parentSpanId, children),
    });
  }

  for (const node of nodes.values()) {
    const parent = node.span.parentSpanId ? nodes.get(node.span.parentSpanId) : null;
    if (parent && parent !== node) parent.children.push(node);
    else roots.push(node);
  }

  sortSpanNodes(roots);
  return roots;
}

export function flattenSpanTree(nodes: SpanNode[]): Span[] {
  const spans: Span[] = [];
  const visit = (node: SpanNode) => {
    spans.push(node.span);
    node.children.forEach(visit);
  };
  nodes.forEach(visit);
  return spans;
}

export function isSyntheticSpan(span: Span | null | undefined) {
  return Boolean(span?.spanAttributes[SYNTHETIC_KIND_KEY]);
}

export function isSessionTraceGroupSpan(span: Span | null | undefined) {
  return span?.spanAttributes[SYNTHETIC_KIND_KEY] === SYNTHETIC_SESSION_TRACE_KIND;
}

export function findFirstInspectableSpan(nodes: SpanNode[]): Span | null {
  for (const node of nodes) {
    if (!isSessionTraceGroupSpan(node.span)) return node.span;
    const child = findFirstInspectableSpan(node.children);
    if (child) return child;
  }
  return null;
}

export function buildSessionSpanTree(spans: Span[], traces: Trace[]): SpanNode[] {
  const spansByTrace = new Map<string, Span[]>();
  for (const span of spans) {
    const group = spansByTrace.get(span.traceId) ?? [];
    group.push(span);
    spansByTrace.set(span.traceId, group);
  }
  const traceMap = new Map(traces.map((trace) => [trace.traceId, trace]));
  return [...spansByTrace.entries()]
    .map(([traceId, group], index) => {
      const trace = traceMap.get(traceId);
      return {
        children: buildClientSpanTree(group),
        span: makeSessionTraceSpan(traceId, trace, group, index),
      };
    })
    .sort((a, b) =>
      a.span.startTimeMs === b.span.startTimeMs
        ? a.span.traceId.localeCompare(b.span.traceId)
        : a.span.startTimeMs - b.span.startTimeMs,
    );
}

function makeSessionTraceSpan(
  traceId: string,
  trace: Trace | undefined,
  spans: Span[],
  index: number,
): Span {
  const first = [...spans].sort((a, b) => a.startTimeMs - b.startTimeMs)[0];
  if (!first) {
    throw new Error("Cannot create a session trace group without spans.");
  }
  const startTimeMs = trace?.startTimeMs ?? Math.min(...spans.map((span) => span.startTimeMs));
  const endTimeMs = trace?.endTimeMs ?? Math.max(...spans.map((span) => span.endTimeMs));
  const durationMs = Math.max(0, endTimeMs - startTimeMs);
  const traceName = trace?.rootSpanName || `Trace ${traceId.slice(0, 8)}`;
  return {
    ...first,
    cacheReadTokens: trace?.cacheReadTokens ?? null,
    costTotal: trace?.totalCost ?? null,
    durationMs,
    durationNs: String(Math.round(durationMs * 1_000_000)),
    endTime: new Date(endTimeMs).toISOString(),
    endTimeMs,
    events: [],
    input: null,
    inputMessages: null,
    inputTokens: null,
    links: [],
    observationKind: trace?.rootObservationKind ?? "SPAN",
    output: null,
    outputMessages: null,
    outputTokens: null,
    parentSpanId: "",
    spanAttributes: {
      [SYNTHETIC_KIND_KEY]: SYNTHETIC_SESSION_TRACE_KIND,
      "halo.synthetic.trace_id": traceId,
      "halo.synthetic.trace_name": traceName,
    },
    spanAttributesDouble: {},
    spanAttributesInt: {},
    spanId: `session:${traceId}`,
    spanKind: "SPAN_KIND_INTERNAL",
    spanName: `Turn ${index + 1}`,
    startTime: new Date(startTimeMs).toISOString(),
    startTimeMs,
    statusCode: trace?.hasError ? "STATUS_CODE_ERROR" : "STATUS_CODE_OK",
    statusMessage: "",
    totalTokens: trace?.totalTokens ?? null,
    traceId,
  };
}

function makePendingParentSpan(parentSpanId: string, children: Span[]): Span {
  const firstChild = [...children].sort((a, b) =>
    a.startTimeMs === b.startTimeMs
      ? a.spanId.localeCompare(b.spanId)
      : a.startTimeMs - b.startTimeMs,
  )[0];
  if (!firstChild) {
    throw new Error("Cannot create a pending parent span without children.");
  }
  const startTimeMs = Math.min(...children.map((span) => span.startTimeMs));
  const endTimeMs = Math.max(...children.map((span) => span.endTimeMs));
  const durationMs = Math.max(0, endTimeMs - startTimeMs);

  return {
    agentId: firstNonEmpty(children, (span) => span.agentId),
    agentName: firstNonEmpty(children, (span) => span.agentName),
    apiKeyId: firstChild.apiKeyId,
    cacheReadTokens: null,
    cacheWriteTokens: null,
    chatId: firstChild.chatId,
    conversationId: firstChild.conversationId,
    costCacheRead: null,
    costCacheWrite: null,
    costInput: null,
    costOutput: null,
    costReasoning: null,
    costTotal: null,
    deploymentEnvironment: firstChild.deploymentEnvironment,
    durationMs,
    durationNs: String(Math.round(durationMs * 1_000_000)),
    endTime: new Date(endTimeMs).toISOString(),
    endTimeMs,
    events: [],
    id: -1,
    ingestedAt: firstChild.ingestedAt,
    input: null,
    inputMessages: null,
    inputTokens: null,
    links: [],
    llmModelName: "",
    llmProvider: "",
    llmResponseModel: "",
    observationKind: firstChild.agentId || firstChild.agentName ? "AGENT" : "SPAN",
    output: null,
    outputMessages: null,
    outputTokens: null,
    parentSpanId: "",
    projectId: firstChild.projectId,
    reasoningTokens: null,
    resourceAttributes: firstChild.resourceAttributes,
    resourceAttributesDouble: {},
    resourceAttributesInt: {},
    retrievalDocuments: null,
    scopeName: firstChild.scopeName,
    scopeVersion: firstChild.scopeVersion,
    serviceName: firstChild.serviceName,
    serviceVersion: firstChild.serviceVersion,
    sessionId: firstChild.sessionId,
    spanAttributes: {
      [SYNTHETIC_KIND_KEY]: SYNTHETIC_PARENT_KIND,
      "halo.synthetic.parent_span_id": parentSpanId,
    },
    spanAttributesDouble: {},
    spanAttributesInt: {},
    spanId: parentSpanId,
    spanKind: "SPAN_KIND_INTERNAL",
    spanName: inferPendingParentName(parentSpanId, children),
    startTime: new Date(startTimeMs).toISOString(),
    startTimeMs,
    statusCode: "STATUS_CODE_UNSET",
    statusMessage: "Waiting for the parent span to be exported.",
    teamId: firstChild.teamId,
    totalTokens: null,
    traceId: firstChild.traceId,
    traceState: firstChild.traceState,
    userId: firstChild.userId,
  };
}

function inferPendingParentName(parentSpanId: string, children: Span[]) {
  const agentId = firstNonEmpty(children, (span) => span.agentId);
  const agentRunName = inferAgentRunName(agentId);
  if (agentRunName) return agentRunName;

  const agentName = firstNonEmpty(children, (span) => span.agentName);
  if (agentName) return `${agentName} run`;

  const serviceName = firstNonEmpty(children, (span) => span.serviceName);
  if (serviceName) return `${serviceName} parent span`;

  return `Pending parent ${parentSpanId.slice(0, 8)}`;
}

function inferAgentRunName(agentId: string) {
  const parts = agentId
    .trim()
    .toLowerCase()
    .split(/[-_.]+/)
    .filter(Boolean);
  if (parts.length < 2) return "";
  return `${parts[0]}.${parts.slice(1).join("_")}.run`;
}

function firstNonEmpty(spans: Span[], read: (span: Span) => string | null | undefined) {
  for (const span of spans) {
    const value = read(span)?.trim();
    if (value) return value;
  }
  return "";
}

function sortSpanNodes(items: SpanNode[]) {
  items.sort((a, b) =>
    a.span.startTimeMs === b.span.startTimeMs
      ? a.span.spanId.localeCompare(b.span.spanId)
      : a.span.startTimeMs - b.span.startTimeMs,
  );
  items.forEach((item) => sortSpanNodes(item.children));
}
