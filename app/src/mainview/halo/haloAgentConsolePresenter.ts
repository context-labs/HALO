import type { HaloRunEvent } from "../../server/halo/types";

export type HaloAgentConsoleKind = "call" | "result" | "message" | "raw";

export type HaloAgentConsoleBody =
  | {
      kind: "json";
      text: string;
      value: unknown;
    }
  | {
      kind: "text";
      text: string;
    };

export type HaloAgentConsoleSummary = {
  label: string;
  value: string;
};

export type HaloAgentConsoleRow = {
  agentId: string | null;
  agentName: string;
  body: HaloAgentConsoleBody | null;
  command: string | null;
  copyText: string;
  createdAt: string;
  depth: number | null;
  eventId: number;
  kind: HaloAgentConsoleKind;
  key: string;
  rawPayload: Record<string, unknown>;
  role: string | null;
  sequence: number;
  stepSequence: number | null;
  subtitle: string | null;
  summaries: HaloAgentConsoleSummary[];
  title: string;
  toolCallId: string | null;
  toolName: string | null;
};

export function presentHaloAgentStep(event: HaloRunEvent): HaloAgentConsoleRow[] {
  const payload = recordOrEmpty(event.payload);
  const item = isRecord(payload.item) ? payload.item : null;
  const role = stringValue(item?.role) ?? stringValue(payload.role);
  const agentName =
    stringValue(payload.agent_name) ??
    stringValue(payload.agentName) ??
    stringValue(item?.name) ??
    "root";
  const agentId = stringValue(payload.agent_id) ?? stringValue(payload.agentId);
  const depth = numberValue(payload.depth);
  const stepSequence = numberValue(payload.sequence);
  const common = {
    agentId,
    agentName,
    createdAt: event.createdAt,
    depth,
    eventId: event.id,
    rawPayload: payload,
    role,
    sequence: event.sequence,
    stepSequence,
  };

  if (item && role === "assistant") {
    const rows: HaloAgentConsoleRow[] = [];
    const content = valueAsText(item.content);
    if (content.trim().length > 0) {
      rows.push({
        ...common,
        body: { kind: "text", text: content },
        command: null,
        copyText: content,
        key: `${event.id}:message`,
        kind: "message",
        subtitle: "assistant message",
        summaries: [],
        title: "assistant replied",
        toolCallId: null,
        toolName: null,
      });
    }

    const toolCalls = Array.isArray(item.tool_calls) ? item.tool_calls : [];
    for (const [index, toolCall] of toolCalls.entries()) {
      if (!isRecord(toolCall)) continue;
      rows.push(presentToolCall({ ...common, index, toolCall }));
    }

    if (rows.length > 0) return rows;
  }

  if (item && role === "tool") {
    return [presentToolResult({ ...common, item })];
  }

  if (item && role) {
    const content = valueAsText(item.content);
    if (content.trim().length > 0) {
      return [
        {
          ...common,
          body: { kind: "text", text: content },
          command: null,
          copyText: content,
          key: `${event.id}:${role}`,
          kind: "message",
          subtitle: `${role} message`,
          summaries: [],
          title: `${role} event`,
          toolCallId: stringValue(item.tool_call_id),
          toolName: null,
        },
      ];
    }
  }

  const rawText = JSON.stringify(payload, null, 2);
  return [
    {
      ...common,
      body: { kind: "json", text: rawText, value: payload },
      command: null,
      copyText: rawText,
      key: `${event.id}:raw`,
      kind: "raw",
      subtitle: event.eventType,
      summaries: [],
      title: stringValue(payload.type) ?? "agent event",
      toolCallId: null,
      toolName: null,
    },
  ];
}

function presentToolCall(input: {
  agentId: string | null;
  agentName: string;
  createdAt: string;
  depth: number | null;
  eventId: number;
  index: number;
  rawPayload: Record<string, unknown>;
  role: string | null;
  sequence: number;
  stepSequence: number | null;
  toolCall: Record<string, unknown>;
}): HaloAgentConsoleRow {
  const fn = isRecord(input.toolCall.function) ? input.toolCall.function : {};
  const toolName = stringValue(fn.name) ?? "tool";
  const toolCallId = stringValue(input.toolCall.id);
  const parsedArguments = parseStructuredValue(fn.arguments);
  const command = formatToolCommand(
    toolName,
    parsedArguments.kind === "json" ? parsedArguments.value : null,
  );
  const argumentSummaries =
    parsedArguments.kind === "json" ? summarizeValue(parsedArguments.value) : [];

  return {
    agentId: input.agentId,
    agentName: input.agentName,
    body: parsedArguments.text ? parsedArguments : null,
    command,
    copyText: command,
    createdAt: input.createdAt,
    depth: input.depth,
    eventId: input.eventId,
    key: `${input.eventId}:call:${toolCallId ?? input.index}`,
    kind: "call",
    rawPayload: input.rawPayload,
    role: input.role,
    sequence: input.sequence,
    stepSequence: input.stepSequence,
    subtitle: "tool call",
    summaries: argumentSummaries,
    title: toolName,
    toolCallId,
    toolName,
  };
}

function presentToolResult(input: {
  agentId: string | null;
  agentName: string;
  createdAt: string;
  depth: number | null;
  eventId: number;
  item: Record<string, unknown>;
  rawPayload: Record<string, unknown>;
  role: string | null;
  sequence: number;
  stepSequence: number | null;
}): HaloAgentConsoleRow {
  const parsedContent = parseStructuredValue(input.item.content);
  const toolCallId = stringValue(input.item.tool_call_id);
  const toolName = stringValue(input.item.name);
  const summaries =
    parsedContent.kind === "json" ? summarizeValue(parsedContent.value) : [];

  return {
    agentId: input.agentId,
    agentName: input.agentName,
    body: parsedContent,
    command: null,
    copyText: parsedContent.text,
    createdAt: input.createdAt,
    depth: input.depth,
    eventId: input.eventId,
    key: `${input.eventId}:result:${toolCallId ?? "tool"}`,
    kind: "result",
    rawPayload: input.rawPayload,
    role: input.role,
    sequence: input.sequence,
    stepSequence: input.stepSequence,
    subtitle: "tool result",
    summaries,
    title: toolName ?? "tool returned",
    toolCallId,
    toolName,
  };
}

function parseStructuredValue(value: unknown): HaloAgentConsoleBody {
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return { kind: "text", text: "" };
    try {
      const parsed = JSON.parse(trimmed) as unknown;
      return {
        kind: "json",
        text: JSON.stringify(parsed, null, 2),
        value: parsed,
      };
    } catch {
      return { kind: "text", text: value };
    }
  }

  if (value == null) return { kind: "text", text: "" };

  if (typeof value === "object") {
    return {
      kind: "json",
      text: JSON.stringify(value, null, 2),
      value,
    };
  }

  return { kind: "text", text: String(value) };
}

function formatToolCommand(toolName: string, args: unknown): string {
  if (!isRecord(args)) return `halo tool ${toolName}`;

  const flags = Object.entries(args)
    .filter(([, value]) => value != null)
    .map(([key, value]) => {
      const flag = `--${key.replace(/[A-Z]/g, (match) => `-${match.toLowerCase()}`)}`;
      if (typeof value === "boolean") return value ? flag : `${flag}=false`;
      return `${flag} ${formatFlagValue(value)}`;
    });

  return ["halo", "tool", toolName, ...flags].join(" ");
}

function formatFlagValue(value: unknown): string {
  if (typeof value === "number") return String(value);
  if (typeof value === "boolean") return String(value);
  if (typeof value === "string") {
    if (/^[A-Za-z0-9._:@/-]+$/.test(value)) return value;
    return JSON.stringify(value);
  }
  return `'${JSON.stringify(value)}'`;
}

function summarizeValue(value: unknown): HaloAgentConsoleSummary[] {
  const target = isRecord(value) && isRecord(value.result) ? value.result : value;
  if (!isRecord(target)) return [];

  return Object.entries(target)
    .filter(([, entry]) => entry != null)
    .slice(0, 8)
    .map(([key, entry]) => ({
      label: titleizeKey(key),
      value: summarizeEntry(entry),
    }));
}

function summarizeEntry(value: unknown): string {
  if (Array.isArray(value)) return `${value.length} item${value.length === 1 ? "" : "s"}`;
  if (isRecord(value)) return `${Object.keys(value).length} key${Object.keys(value).length === 1 ? "" : "s"}`;
  if (typeof value === "string") return value.length > 72 ? `${value.slice(0, 69)}...` : value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return String(value);
}

function titleizeKey(value: string): string {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function valueAsText(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

function recordOrEmpty(value: unknown): Record<string, unknown> {
  return isRecord(value) ? value : {};
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value != null && !Array.isArray(value);
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
