import type { Span, SpanNode } from "../../../server/telemetry/types";
import { parseSpanConversation, type ParsedMessage } from "../llmMessages";
import { isSessionTraceGroupSpan, isSyntheticSpan } from "../spanTree";
import { spanKey } from "./spanUtils";
import { numericCost } from "./spanKinds";

export type ConversationSpanRow = {
  /** Indent level relative to the group (ancestors inside the same group). */
  depth: number;
  key: string;
  span: Span;
};

export type ConversationGroup = {
  /** Sum of LLM span costs, null when no span carried a cost. */
  costTotal: number | null;
  endMs: number;
  failedCount: number;
  rows: ConversationSpanRow[];
  startMs: number;
  /** TOOL-kind spans in the group (a subset of rows). */
  toolCallCount: number;
};

export type ConversationMessage = {
  atMs: number;
  /** Span groups only hang off agent messages. */
  group: ConversationGroup | null;
  role: "agent" | "user";
  text: string;
};

export type ConversationModel = {
  messages: ConversationMessage[];
  outcome: {
    costTotal: number | null;
    durationMs: number;
    endMs: number;
    ok: boolean;
    startMs: number;
  };
};

/**
 * Rebuild the user↔agent conversation from a trace's spans.
 *
 * Agent loops resend the whole history on every LLM call, so messages are
 * deduplicated by signature in start-time order (the first occurrence wins —
 * an assistant turn first appears in the producing span's output). Surviving
 * user messages become user bubbles; surviving assistant output text becomes
 * agent message boundaries, and every span is grouped under the earliest
 * boundary that finishes at or after it.
 */
export function buildTraceConversation(
  spans: Span[],
  tree: SpanNode[],
): ConversationModel {
  const realSpans = spans.filter((span) => !isSessionTraceGroupSpan(span));
  // Sub-agent LLM calls see the orchestrator's instructions as "user"
  // messages — only top-level LLM spans speak in the conversation; sub-agent
  // work stays inside the disclosure chips.
  const topLevel = topLevelLlmSpans(tree);
  const survivors = dedupedMessages(topLevel.length > 0 ? topLevel : realSpans);

  const userMessages: ConversationMessage[] = [];
  const boundaries: Array<{ atMs: number; span: Span; text: string }> = [];
  for (const { message, span } of survivors) {
    const text = message.content.trim();
    if (!text) continue;
    if (message.role === "user") {
      userMessages.push({
        atMs: span.startTimeMs,
        group: null,
        role: "user",
        text,
      });
    } else if (message.role === "assistant" && message.source === "output") {
      boundaries.push({ atMs: span.endTimeMs, span, text });
    }
  }
  boundaries.sort((a, b) => a.atMs - b.atMs);

  const { parentKeyByKey, rows } = groupableRows(tree);

  if (userMessages.length === 0) {
    const fallback = fallbackUserText(tree);
    if (fallback) {
      const startMs = rows[0]?.span.startTimeMs ?? 0;
      userMessages.push({ atMs: startMs - 1, group: null, role: "user", text: fallback });
    }
  }

  if (boundaries.length === 0) {
    const text = fallbackAgentText(tree) ?? "";
    const endMs = rows.reduce((max, row) => Math.max(max, row.span.endTimeMs), 0);
    const group = rows.length > 0 ? makeGroup(rows, parentKeyByKey) : null;
    const agentMessage: ConversationMessage = {
      atMs: endMs,
      group,
      role: "agent",
      text,
    };
    return finishModel(realSpans, [...userMessages, agentMessage]);
  }

  // Assign every span to the earliest boundary finishing at or after it;
  // stragglers after the last boundary attach to the final message.
  const rowsByBoundary = new Map<number, RawRow[]>();
  for (const row of rows) {
    let index = boundaries.findIndex((b) => b.atMs >= row.span.endTimeMs);
    if (index === -1) index = boundaries.length - 1;
    const bucket = rowsByBoundary.get(index) ?? [];
    bucket.push(row);
    rowsByBoundary.set(index, bucket);
  }

  const agentMessages: ConversationMessage[] = boundaries.map((boundary, index) => {
    const bucket = rowsByBoundary.get(index) ?? [];
    return {
      atMs: boundary.atMs,
      group: bucket.length > 0 ? makeGroup(bucket, parentKeyByKey) : null,
      role: "agent",
      text: boundary.text,
    };
  });

  return finishModel(
    realSpans,
    [...userMessages, ...agentMessages].sort((a, b) => a.atMs - b.atMs),
  );
}

/**
 * Session conversation: each trace is a turn, reconstructed independently and
 * concatenated in time order. The synthetic "Turn N" group nodes wrap each
 * trace's subtree in the session display tree.
 */
export function buildSessionConversation(tree: SpanNode[]): ConversationModel {
  const turnNodes = tree.filter((node) => isSessionTraceGroupSpan(node.span));
  if (turnNodes.length === 0) {
    // Not a synthetic-grouped session tree; treat it as one trace.
    return buildTraceConversation(flattenNodes(tree), tree);
  }
  const messages: ConversationMessage[] = [];
  const allSpans: Span[] = [];
  for (const turn of turnNodes) {
    const spans = flattenNodes(turn.children);
    allSpans.push(...spans);
    messages.push(...buildTraceConversation(spans, turn.children).messages);
  }
  messages.sort((a, b) => a.atMs - b.atMs);
  return finishModel(allSpans, messages);
}

type RawRow = { depth: number; key: string; span: Span };

type SpanMessage = { message: ParsedMessage; span: Span };

/** LLM spans with no AGENT ancestor besides the trace root. */
function topLevelLlmSpans(tree: SpanNode[]): Span[] {
  const out: Span[] = [];
  const walk = (node: SpanNode, agentDepth: number, isRoot: boolean) => {
    const kind = node.span.observationKind;
    if (kind === "LLM" && agentDepth === 0) out.push(node.span);
    const nextDepth = agentDepth + (!isRoot && kind === "AGENT" ? 1 : 0);
    node.children.forEach((child) => walk(child, nextDepth, false));
  };
  tree.forEach((root) => {
    if (isSessionTraceGroupSpan(root.span)) {
      root.children.forEach((child) => walk(child, 0, true));
    } else {
      walk(root, 0, true);
    }
  });
  return out;
}

function dedupedMessages(spans: Span[]): SpanMessage[] {
  const llmSpans = spans
    .filter((span) => span.observationKind === "LLM")
    .sort((a, b) =>
      a.startTimeMs === b.startTimeMs
        ? a.spanId.localeCompare(b.spanId)
        : a.startTimeMs - b.startTimeMs,
    );

  const seen = new Set<string>();
  const survivors: SpanMessage[] = [];
  for (const span of llmSpans) {
    const conversation = parseSpanConversation(span);
    if (!conversation) continue;
    for (const message of conversation.messages) {
      const signature = `${message.role}:${message.content.slice(0, 120)}:${message.toolCalls
        .map((call) => call.name)
        .join(",")}`;
      if (seen.has(signature)) continue;
      seen.add(signature);
      survivors.push({ message, span });
    }
  }
  return survivors;
}

type GroupableRows = {
  parentKeyByKey: Map<string, string | null>;
  rows: RawRow[];
};

/**
 * Spans that belong in disclosure chips: the display tree in order, minus a
 * sole structural root (a chip listing the whole-trace span is noise) and
 * synthetic session groups. Depth is relative to chip nesting and fixed up
 * per-group in makeGroup.
 */
function groupableRows(tree: SpanNode[]): GroupableRows {
  const rows: RawRow[] = [];
  const parentKeyByKey = new Map<string, string | null>();
  const soleRoot = tree.length === 1 && tree[0]!.children.length > 0 ? tree[0] : null;
  const walk = (node: SpanNode, depth: number, parentKey: string | null) => {
    const key = spanKey(node.span);
    rows.push({ depth, key, span: node.span });
    parentKeyByKey.set(key, parentKey);
    node.children.forEach((child) => walk(child, depth + 1, key));
  };
  if (soleRoot) {
    soleRoot.children.forEach((child) => walk(child, 0, null));
  } else {
    tree.forEach((node) => {
      if (isSessionTraceGroupSpan(node.span)) {
        node.children.forEach((child) => walk(child, 0, null));
      } else {
        walk(node, 0, null);
      }
    });
  }
  return { parentKeyByKey, rows };
}

function makeGroup(
  rows: RawRow[],
  parentKeyByKey: Map<string, string | null>,
): ConversationGroup {
  const memberKeys = new Set(rows.map((row) => row.key));
  const groupRows: ConversationSpanRow[] = rows.map((row) => {
    let depth = 0;
    let parent = parentKeyByKey.get(row.key) ?? null;
    while (parent) {
      if (memberKeys.has(parent)) depth += 1;
      parent = parentKeyByKey.get(parent) ?? null;
    }
    return { depth, key: row.key, span: row.span };
  });

  let costTotal: number | null = null;
  let failedCount = 0;
  let toolCallCount = 0;
  let startMs = Infinity;
  let endMs = -Infinity;
  for (const row of groupRows) {
    if (row.span.observationKind === "LLM" && row.span.costTotal != null) {
      costTotal = (costTotal ?? 0) + numericCost(row.span.costTotal);
    }
    if (row.span.observationKind === "TOOL") toolCallCount += 1;
    if (row.span.statusCode === "STATUS_CODE_ERROR") failedCount += 1;
    if (row.span.startTimeMs < startMs) startMs = row.span.startTimeMs;
    if (row.span.endTimeMs > endMs) endMs = row.span.endTimeMs;
  }

  return {
    costTotal,
    endMs: Number.isFinite(endMs) ? endMs : 0,
    failedCount,
    rows: groupRows,
    startMs: Number.isFinite(startMs) ? startMs : 0,
    toolCallCount,
  };
}

/** Root → earliest content-bearing span, mirroring the web app's fallback. */
function fallbackUserText(tree: SpanNode[]): string | null {
  for (const span of fallbackCandidates(tree)) {
    const conversation = parseSpanConversation(span);
    const user = conversation?.messages.find(
      (message) => message.role === "user" && message.content.trim(),
    );
    if (user) return user.content.trim();
    if (span.input?.trim() && !span.input.trim().startsWith("{")) {
      return span.input.trim();
    }
  }
  return null;
}

function fallbackAgentText(tree: SpanNode[]): string | null {
  const candidates = fallbackCandidates(tree);
  for (const span of candidates) {
    const conversation = parseSpanConversation(span);
    const assistant = [...(conversation?.messages ?? [])]
      .reverse()
      .find(
        (message) =>
          message.role === "assistant" &&
          message.source === "output" &&
          message.content.trim(),
      );
    if (assistant) return assistant.content.trim();
    if (span.output?.trim() && !span.output.trim().startsWith("{")) {
      return span.output.trim();
    }
  }
  return null;
}

function fallbackCandidates(tree: SpanNode[]): Span[] {
  const spans = flattenNodes(tree).filter((span) => !isSyntheticSpan(span));
  const structuralRoots = spans.filter((span) => !span.parentSpanId);
  const byKindThenTime = [...spans].sort((a, b) => {
    const rank = (span: Span) =>
      span.observationKind === "AGENT" ? 0 : span.observationKind === "LLM" ? 1 : 2;
    return rank(a) === rank(b) ? a.startTimeMs - b.startTimeMs : rank(a) - rank(b);
  });
  return [...structuralRoots, ...byKindThenTime];
}

function flattenNodes(nodes: SpanNode[]): Span[] {
  const out: Span[] = [];
  const visit = (node: SpanNode) => {
    out.push(node.span);
    node.children.forEach(visit);
  };
  nodes.forEach(visit);
  return out;
}

function finishModel(
  spans: Span[],
  messages: ConversationMessage[],
): ConversationModel {
  let startMs = Infinity;
  let endMs = -Infinity;
  let costTotal: number | null = null;
  let ok = true;
  for (const span of spans) {
    if (isSyntheticSpan(span)) continue;
    if (span.startTimeMs < startMs) startMs = span.startTimeMs;
    if (span.endTimeMs > endMs) endMs = span.endTimeMs;
    if (span.statusCode === "STATUS_CODE_ERROR") ok = false;
    if (span.observationKind === "LLM" && span.costTotal != null) {
      costTotal = (costTotal ?? 0) + numericCost(span.costTotal);
    }
  }
  if (!Number.isFinite(startMs)) startMs = 0;
  if (!Number.isFinite(endMs)) endMs = startMs;
  return {
    messages,
    outcome: {
      costTotal,
      durationMs: Math.max(0, endMs - startMs),
      endMs,
      ok,
      startMs,
    },
  };
}
