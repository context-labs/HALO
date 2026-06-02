import type { SpanDbRow, ObservationKind, ProducerShape } from "./types";
import { LOCAL_TELEMETRY_AUTH } from "./types";

export interface OtlpAnyValue {
  stringValue?: string;
  boolValue?: boolean;
  intValue?: string | number;
  doubleValue?: number;
  arrayValue?: { values?: OtlpAnyValue[] };
  kvlistValue?: { values?: OtlpKeyValue[] };
  bytesValue?: string;
}

export interface OtlpKeyValue {
  key: string;
  value?: OtlpAnyValue;
}

interface OtlpEvent {
  timeUnixNano?: string | number;
  name?: string;
  attributes?: OtlpKeyValue[];
}

interface OtlpLink {
  traceId?: string;
  spanId?: string;
  traceState?: string;
  attributes?: OtlpKeyValue[];
}

interface OtlpSpan {
  traceId?: string;
  spanId?: string;
  parentSpanId?: string;
  traceState?: string;
  name?: string;
  kind?: number | string;
  startTimeUnixNano?: string | number;
  endTimeUnixNano?: string | number;
  attributes?: OtlpKeyValue[];
  events?: OtlpEvent[];
  links?: OtlpLink[];
  status?: {
    code?: number | string;
    message?: string;
  };
}

interface OtlpScopeSpans {
  scope?: { name?: string; version?: string };
  instrumentationScope?: { name?: string; version?: string };
  spans?: OtlpSpan[];
}

interface OtlpResourceSpans {
  resource?: { attributes?: OtlpKeyValue[] };
  scopeSpans?: OtlpScopeSpans[];
  instrumentationLibrarySpans?: OtlpScopeSpans[];
}

export interface OtlpExportTraceServiceRequest {
  resourceSpans?: OtlpResourceSpans[];
}

type DecodedAttribute =
  | { kind: "string"; value: string }
  | { kind: "int"; value: number }
  | { kind: "double"; value: number }
  | { kind: "bool"; value: boolean };

type AttributeMaps = {
  strings: Record<string, string>;
  ints: Record<string, number>;
  doubles: Record<string, number>;
};

type CanonicalColumns = {
  llm_provider?: string;
  llm_model_name?: string;
  llm_response_model?: string;
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  cache_read_tokens?: number;
  cache_write_tokens?: number;
  reasoning_tokens?: number;
  cost_total?: number;
  cost_input?: number;
  cost_output?: number;
  cost_cache_read?: number;
  cost_cache_write?: number;
  cost_reasoning?: number;
  user_id?: string;
  session_id?: string;
  conversation_id?: string;
  chat_id?: string;
  agent_name?: string;
  agent_id?: string;
  input_messages?: string;
  output_messages?: string;
  input?: string;
  output?: string;
  retrieval_documents?: string;
};

const SPAN_KIND_STRINGS: Record<string, string> = {
  "0": "SPAN_KIND_UNSPECIFIED",
  "1": "SPAN_KIND_INTERNAL",
  "2": "SPAN_KIND_SERVER",
  "3": "SPAN_KIND_CLIENT",
  "4": "SPAN_KIND_PRODUCER",
  "5": "SPAN_KIND_CONSUMER",
};

const STATUS_CODE_STRINGS: Record<string, string> = {
  "0": "STATUS_CODE_UNSET",
  "1": "STATUS_CODE_OK",
  "2": "STATUS_CODE_ERROR",
};

export function decodeOtlpJsonBody(body: string): OtlpExportTraceServiceRequest {
  if (!body.trim()) return { resourceSpans: [] };
  const parsed = JSON.parse(body) as unknown;
  if (parsed == null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("invalid OTLP JSON payload: not an object");
  }
  return parsed as OtlpExportTraceServiceRequest;
}

export function buildSpanRowsFromOtlp(
  request: OtlpExportTraceServiceRequest,
  now = Date.now(),
): SpanDbRow[] {
  const rows: SpanDbRow[] = [];
  for (const resourceSpans of request.resourceSpans ?? []) {
    const resourceAttrs = resourceSpans.resource?.attributes ?? [];
    const resourceMap = toAttributeMap(resourceAttrs);
    const resourceSplit = splitAttributesByType(resourceAttrs);
    const serviceName = getString(resourceMap, "service.name") ?? "";
    const serviceVersion = getString(resourceMap, "service.version") ?? "";
    const deploymentEnvironment =
      getString(resourceMap, "deployment.environment") ?? "";

    for (const scopeSpans of [
      ...(resourceSpans.scopeSpans ?? []),
      ...(resourceSpans.instrumentationLibrarySpans ?? []),
    ]) {
      const scope = scopeSpans.scope ?? scopeSpans.instrumentationScope;
      const scopeName = scope?.name ?? "";
      const scopeVersion = scope?.version ?? "";
      for (const span of scopeSpans.spans ?? []) {
        rows.push(
          oneSpanToRow({
            deploymentEnvironment,
            ingestedAt: now,
            resourceSplit,
            scopeName,
            scopeVersion,
            serviceName,
            serviceVersion,
            span,
          }),
        );
      }
    }
  }
  return rows;
}

function oneSpanToRow(args: {
  span: OtlpSpan;
  scopeName: string;
  scopeVersion: string;
  resourceSplit: AttributeMaps;
  serviceName: string;
  serviceVersion: string;
  deploymentEnvironment: string;
  ingestedAt: number;
}): SpanDbRow {
  const attrMap = toAttributeMap(args.span.attributes ?? []);
  const attrSplit = splitAttributesByType(args.span.attributes ?? []);
  const canonical = extractCanonicalLlmColumns(attrMap);
  const observationKind = classifyObservationKind(attrMap, args.scopeName);

  const startNano = normalizeNano(args.span.startTimeUnixNano);
  const endNano = normalizeNano(args.span.endTimeUnixNano);
  const startTime = nanosToMs(startNano);
  const endTime = nanosToMs(endNano);
  const duration = durationNs(startNano, endNano);

  return {
    api_key_id: LOCAL_TELEMETRY_AUTH.apiKeyId,
    project_id: LOCAL_TELEMETRY_AUTH.projectId,
    team_id: LOCAL_TELEMETRY_AUTH.teamId,
    trace_id: normaliseId(args.span.traceId),
    span_id: normaliseId(args.span.spanId),
    parent_span_id: normaliseId(args.span.parentSpanId),
    trace_state: args.span.traceState ?? "",
    start_time: startTime,
    end_time: endTime,
    start_time_unix_nano: startNano,
    end_time_unix_nano: endNano,
    duration_ns: duration,
    duration_ms: Number(duration) / 1_000_000,
    ingested_at: args.ingestedAt,
    span_name: args.span.name ?? "",
    span_kind: spanKindString(args.span.kind),
    service_name: args.serviceName,
    service_version: args.serviceVersion,
    deployment_environment: args.deploymentEnvironment,
    scope_name: args.scopeName,
    scope_version: args.scopeVersion,
    status_code: statusCodeString(args.span.status?.code),
    status_message: args.span.status?.message ?? "",
    observation_kind: observationKind,
    llm_provider: canonical.llm_provider ?? "",
    llm_model_name: canonical.llm_model_name ?? "",
    llm_response_model: canonical.llm_response_model ?? "",
    input_tokens: canonical.input_tokens ?? null,
    output_tokens: canonical.output_tokens ?? null,
    total_tokens: canonical.total_tokens ?? null,
    cache_read_tokens: canonical.cache_read_tokens ?? null,
    cache_write_tokens: canonical.cache_write_tokens ?? null,
    reasoning_tokens: canonical.reasoning_tokens ?? null,
    cost_total: canonical.cost_total ?? null,
    cost_input: canonical.cost_input ?? null,
    cost_output: canonical.cost_output ?? null,
    cost_cache_read: canonical.cost_cache_read ?? null,
    cost_cache_write: canonical.cost_cache_write ?? null,
    cost_reasoning: canonical.cost_reasoning ?? null,
    user_id: canonical.user_id ?? null,
    session_id: canonical.session_id ?? null,
    conversation_id: canonical.conversation_id ?? "",
    chat_id: canonical.chat_id ?? "",
    agent_name: canonical.agent_name ?? "",
    agent_id: canonical.agent_id ?? "",
    input_messages: canonical.input_messages ?? null,
    output_messages: canonical.output_messages ?? null,
    input: canonical.input ?? null,
    output: canonical.output ?? null,
    retrieval_documents: canonical.retrieval_documents ?? null,
    resource_attributes: JSON.stringify(args.resourceSplit.strings),
    resource_attributes_int: JSON.stringify(args.resourceSplit.ints),
    resource_attributes_double: JSON.stringify(args.resourceSplit.doubles),
    span_attributes: JSON.stringify(attrSplit.strings),
    span_attributes_int: JSON.stringify(attrSplit.ints),
    span_attributes_double: JSON.stringify(attrSplit.doubles),
    events_json: JSON.stringify(
      (args.span.events ?? []).map((event) => ({
        attributes: stringifyAttributes(event.attributes ?? []),
        name: event.name ?? "",
        timestamp: isoFromMs(nanosToMs(normalizeNano(event.timeUnixNano))),
      })),
    ),
    links_json: JSON.stringify(
      (args.span.links ?? []).map((link) => ({
        attributes: stringifyAttributes(link.attributes ?? []),
        spanId: normaliseId(link.spanId),
        traceId: normaliseId(link.traceId),
        traceState: link.traceState ?? "",
      })),
    ),
  };
}

export function decodeAttributeValue(value: OtlpAnyValue | undefined): DecodedAttribute | null {
  if (value == null) return null;
  if (value.stringValue != null) return { kind: "string", value: value.stringValue };
  if (value.boolValue != null) return { kind: "bool", value: value.boolValue };
  if (value.intValue != null) return { kind: "int", value: Number(value.intValue) };
  if (value.doubleValue != null) return { kind: "double", value: Number(value.doubleValue) };
  if (value.arrayValue != null) {
    return {
      kind: "string",
      value: JSON.stringify((value.arrayValue.values ?? []).map(anyValueToJson)),
    };
  }
  if (value.kvlistValue != null) {
    return {
      kind: "string",
      value: JSON.stringify(
        Object.fromEntries(
          (value.kvlistValue.values ?? []).map((kv) => [
            kv.key,
            anyValueToJson(kv.value),
          ]),
        ),
      ),
    };
  }
  if (value.bytesValue != null) return { kind: "string", value: value.bytesValue };
  return null;
}

function anyValueToJson(value: OtlpAnyValue | undefined): unknown {
  const decoded = decodeAttributeValue(value);
  return decoded?.value ?? null;
}

function toAttributeMap(attrs: OtlpKeyValue[]): Map<string, DecodedAttribute> {
  const map = new Map<string, DecodedAttribute>();
  for (const kv of attrs) {
    const decoded = decodeAttributeValue(kv.value);
    if (decoded != null) map.set(kv.key, decoded);
  }
  return map;
}

function splitAttributesByType(attrs: OtlpKeyValue[]): AttributeMaps {
  const out: AttributeMaps = { doubles: {}, ints: {}, strings: {} };
  for (const kv of attrs) {
    const decoded = decodeAttributeValue(kv.value);
    if (decoded == null) continue;
    if (decoded.kind === "string") out.strings[kv.key] = decoded.value;
    if (decoded.kind === "int") out.ints[kv.key] = decoded.value;
    if (decoded.kind === "double") out.doubles[kv.key] = decoded.value;
    if (decoded.kind === "bool") {
      out.ints[kv.key] = decoded.value ? 1 : 0;
      out.strings[kv.key] = decoded.value ? "true" : "false";
    }
  }
  return out;
}

function stringifyAttributes(attrs: OtlpKeyValue[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const kv of attrs) {
    const decoded = decodeAttributeValue(kv.value);
    if (decoded == null) continue;
    out[kv.key] = String(decoded.value);
  }
  return out;
}

function getString(map: Map<string, DecodedAttribute>, key: string): string | undefined {
  const value = map.get(key);
  if (value == null) return undefined;
  return String(value.value);
}

function getNumber(map: Map<string, DecodedAttribute>, key: string): number | undefined {
  const value = map.get(key);
  if (value == null) return undefined;
  const n = Number(value.value);
  return Number.isFinite(n) ? n : undefined;
}

function extractCanonicalLlmColumns(attrs: Map<string, DecodedAttribute>): CanonicalColumns {
  const merged: CanonicalColumns = {
    ...extractCodex(attrs),
    ...extractVercelAi(attrs),
    ...extractUpstreamGenAi(attrs),
    ...extractOpenInference(attrs),
  };

  const sessionId = getString(attrs, "session.id");
  const conversationId =
    getString(attrs, "conversation.id") ?? getString(attrs, "conversation_id");
  const chatId = getString(attrs, "chat.id") ?? getString(attrs, "chat_id");
  const cursorRunId = getString(attrs, "cursor.run_id");
  const cursorAgentId = getString(attrs, "cursor.agent_id");
  const elevenConversationId = getString(attrs, "elevenlabs.conversation_id");
  const elevenUserId = getString(attrs, "elevenlabs.user_id");

  if (sessionId != null) merged.session_id = sessionId;
  else if (cursorRunId != null) merged.session_id = cursorRunId;
  else if (elevenConversationId != null) merged.session_id = elevenConversationId;
  if (conversationId != null) merged.conversation_id = conversationId;
  if (chatId != null) merged.chat_id = chatId;
  if (cursorAgentId != null && !merged.agent_id) merged.agent_id = cursorAgentId;
  if (elevenUserId != null && !merged.user_id) merged.user_id = elevenUserId;

  return merged;
}

function extractOpenInference(attrs: Map<string, DecodedAttribute>): CanonicalColumns {
  const out: CanonicalColumns = {};
  setString(out, "llm_provider", attrs, "llm.provider");
  setString(out, "llm_model_name", attrs, "llm.model_name");
  setNumber(out, "input_tokens", attrs, "llm.token_count.prompt");
  setNumber(out, "output_tokens", attrs, "llm.token_count.completion");
  setNumber(out, "total_tokens", attrs, "llm.token_count.total");
  setNumber(out, "cache_read_tokens", attrs, "llm.token_count.prompt_details.cache_read");
  setNumber(out, "cache_write_tokens", attrs, "llm.token_count.prompt_details.cache_write");
  setNumber(out, "reasoning_tokens", attrs, "llm.token_count.completion_details.reasoning");
  setNumber(out, "cost_total", attrs, "llm.cost.total");
  setNumber(out, "cost_input", attrs, "llm.cost.prompt_details.input");
  setNumber(out, "cost_output", attrs, "llm.cost.completion_details.output");
  setNumber(out, "cost_cache_read", attrs, "llm.cost.prompt_details.cache_read");
  setNumber(out, "cost_cache_write", attrs, "llm.cost.prompt_details.cache_write");
  setNumber(out, "cost_reasoning", attrs, "llm.cost.completion_details.reasoning");
  setString(out, "user_id", attrs, "user.id");
  setString(out, "session_id", attrs, "session.id");
  setString(out, "agent_name", attrs, "agent.name");
  setString(out, "agent_id", attrs, "agent.id");
  setString(out, "input_messages", attrs, "llm.input_messages");
  setString(out, "output_messages", attrs, "llm.output_messages");
  setString(out, "input", attrs, "input.value");
  setString(out, "output", attrs, "output.value");
  setString(out, "retrieval_documents", attrs, "retrieval.documents");
  out.input_messages ??= reassembleIndexedMessages(attrs, "llm.input_messages");
  out.output_messages ??= reassembleIndexedMessages(attrs, "llm.output_messages");
  return out;
}

function extractUpstreamGenAi(attrs: Map<string, DecodedAttribute>): CanonicalColumns {
  const out: CanonicalColumns = {};
  out.llm_provider = getString(attrs, "gen_ai.provider.name") ?? getString(attrs, "gen_ai.system");
  setString(out, "llm_model_name", attrs, "gen_ai.request.model");
  setString(out, "llm_response_model", attrs, "gen_ai.response.model");
  out.input_tokens =
    getNumber(attrs, "gen_ai.usage.input_tokens") ??
    getNumber(attrs, "gen_ai.usage.prompt_tokens");
  out.output_tokens =
    getNumber(attrs, "gen_ai.usage.output_tokens") ??
    getNumber(attrs, "gen_ai.usage.completion_tokens");
  setNumber(out, "total_tokens", attrs, "gen_ai.usage.total_tokens");
  setNumber(out, "cache_read_tokens", attrs, "gen_ai.usage.cache_read.input_tokens");
  setNumber(out, "cache_write_tokens", attrs, "gen_ai.usage.cache_creation.input_tokens");
  out.input_messages =
    getString(attrs, "gen_ai.input.messages") ?? reassembleIndexedMessages(attrs, "gen_ai.prompt");
  out.output_messages =
    getString(attrs, "gen_ai.output.messages") ??
    reassembleIndexedMessages(attrs, "gen_ai.completion");
  out.user_id =
    getString(attrs, "traceloop.association.properties.user_id") ??
    getString(attrs, "user.id");
  out.session_id =
    getString(attrs, "traceloop.association.properties.session_id") ??
    getString(attrs, "session.id");
  setString(out, "agent_name", attrs, "gen_ai.agent.name");
  setString(out, "agent_id", attrs, "gen_ai.agent.id");
  return out;
}

function extractVercelAi(attrs: Map<string, DecodedAttribute>): CanonicalColumns {
  const out: CanonicalColumns = {};
  setString(out, "llm_provider", attrs, "ai.model.provider");
  setString(out, "llm_model_name", attrs, "ai.model.id");
  out.input_tokens =
    getNumber(attrs, "ai.usage.inputTokens") ??
    getNumber(attrs, "ai.usage.promptTokens");
  out.output_tokens =
    getNumber(attrs, "ai.usage.outputTokens") ??
    getNumber(attrs, "ai.usage.completionTokens");
  out.total_tokens =
    getNumber(attrs, "ai.usage.totalTokens") ?? getNumber(attrs, "ai.usage.tokens");
  setNumber(out, "cache_read_tokens", attrs, "ai.usage.cachedInputTokens");
  setNumber(out, "reasoning_tokens", attrs, "ai.usage.reasoningTokens");
  setString(out, "input_messages", attrs, "ai.prompt.messages");
  setString(out, "output_messages", attrs, "ai.response.toolCalls");
  setString(out, "output", attrs, "ai.response.text");
  setString(out, "agent_name", attrs, "ai.telemetry.functionId");
  out.user_id =
    getString(attrs, "ai.telemetry.metadata.userId") ??
    getString(attrs, "ai.telemetry.metadata.user_id");
  out.session_id =
    getString(attrs, "ai.telemetry.metadata.sessionId") ??
    getString(attrs, "ai.telemetry.metadata.session_id");
  return out;
}

function extractCodex(attrs: Map<string, DecodedAttribute>): CanonicalColumns {
  const out: CanonicalColumns = {};
  setString(out, "llm_model_name", attrs, "model");
  setString(out, "conversation_id", attrs, "conversation_id");
  setString(out, "user_id", attrs, "account_id");
  return out;
}

function setString<T extends keyof CanonicalColumns>(
  out: CanonicalColumns,
  field: T,
  attrs: Map<string, DecodedAttribute>,
  key: string,
) {
  const value = getString(attrs, key);
  if (value != null) out[field] = value as CanonicalColumns[T];
}

function setNumber<T extends keyof CanonicalColumns>(
  out: CanonicalColumns,
  field: T,
  attrs: Map<string, DecodedAttribute>,
  key: string,
) {
  const value = getNumber(attrs, key);
  if (value != null) out[field] = value as CanonicalColumns[T];
}

function classifyObservationKind(
  attrs: Map<string, DecodedAttribute>,
  scopeName: string,
): ObservationKind {
  const explicit =
    getString(attrs, "inference.observation.kind") ??
    getString(attrs, "openinference.span.kind");
  const normalized = normalizeKind(explicit);
  if (normalized) return normalized;

  const genAiOp = getString(attrs, "gen_ai.operation.name")?.toLowerCase();
  if (genAiOp) {
    if (["chat", "completion", "generate_content", "text_completion"].includes(genAiOp)) {
      return "LLM";
    }
    if (genAiOp.includes("embed")) return "EMBEDDING";
    if (genAiOp.includes("tool")) return "TOOL";
  }

  const runType = getString(attrs, "langchain.run_type")?.toLowerCase();
  if (runType === "llm" || runType === "chat_model") return "LLM";
  if (runType === "tool") return "TOOL";
  if (runType === "chain") return "CHAIN";
  if (runType === "retriever") return "RETRIEVER";

  const braintrust = getString(attrs, "braintrust.span_type")?.toLowerCase();
  if (braintrust === "llm") return "LLM";
  if (braintrust === "tool") return "TOOL";
  if (braintrust === "function") return "CHAIN";

  if (getString(attrs, "tool.name") || getString(attrs, "gen_ai.tool.name")) return "TOOL";
  const inferred = scopeInferredKind(scopeName);
  return inferred ?? "SPAN";
}

function normalizeKind(value: string | undefined): ObservationKind | undefined {
  if (!value) return undefined;
  const upper = value.toUpperCase();
  const known: ObservationKind[] = [
    "LLM",
    "TOOL",
    "CHAIN",
    "RETRIEVER",
    "EMBEDDING",
    "AGENT",
    "RERANKER",
    "GUARDRAIL",
    "EVALUATOR",
    "PROMPT",
    "UNKNOWN",
    "SPAN",
  ];
  return known.includes(upper as ObservationKind)
    ? (upper as ObservationKind)
    : undefined;
}

function identifyProducerShape(scopeName: string): ProducerShape {
  const bridge = inferenceBridgeSuffix(scopeName);
  if (bridge != null) {
    if (bridge === "ai-sdk") return "vercel-ai";
    if (bridge === "cursor-sdk") return "cursor";
    if (bridge === "elevenlabs") return "elevenlabs";
    if (bridge === "unknown-future") return "unknown";
    return "openinference";
  }
  if (
    scopeName.startsWith("openinference.instrumentation.") ||
    scopeName.startsWith("@arizeai/openinference-instrumentation")
  ) {
    return "openinference";
  }
  if (scopeName === "traceloop" || scopeName.startsWith("traceloop.")) return "openllmetry";
  if (scopeName === "ai") return "vercel-ai";
  if (scopeName === "langsmith") return "langsmith";
  if (scopeName === "pydantic-ai") return "pydantic-ai";
  if (scopeName === "cursor-sdk") return "cursor";
  if (scopeName === "braintrust" || scopeName.startsWith("braintrust-")) return "braintrust";
  if (scopeName === "codex" || scopeName.startsWith("codex") || scopeName.startsWith("codex-rs")) {
    return "codex";
  }
  return "unknown";
}

function scopeInferredKind(scopeName: string): ObservationKind | undefined {
  return identifyProducerShape(scopeName) === "unknown" ? undefined : "LLM";
}

function inferenceBridgeSuffix(scopeName: string): string | undefined {
  const prefixes = [
    "@inference/tracing",
    "catalyst_tracing",
    "inference_catalyst_tracing",
  ];
  for (const prefix of prefixes) {
    if (scopeName === prefix) return "";
    if (scopeName.startsWith(`${prefix}.`)) return scopeName.slice(prefix.length + 1);
  }
  return undefined;
}

function reassembleIndexedMessages(
  attrs: Map<string, DecodedAttribute>,
  prefix: string,
): string | undefined {
  const parts = Array.from(attrs.entries())
    .filter(([key]) => key.startsWith(`${prefix}.`))
    .sort(([a], [b]) => a.localeCompare(b));
  if (parts.length === 0) return undefined;
  return JSON.stringify(
    Object.fromEntries(parts.map(([key, value]) => [key, String(value.value)])),
  );
}

function normalizeNano(raw: string | number | undefined): string {
  if (raw == null || raw === "") return "0";
  return String(raw);
}

function nanosToMs(nanos: string): number {
  try {
    return Number(BigInt(nanos) / 1_000_000n);
  } catch {
    return 0;
  }
}

function durationNs(start: string, end: string): string {
  try {
    const duration = BigInt(end) - BigInt(start);
    return duration > 0n ? duration.toString() : "0";
  } catch {
    return "0";
  }
}

function spanKindString(kind: number | string | undefined): string {
  if (kind == null) return "SPAN_KIND_UNSPECIFIED";
  if (typeof kind === "string" && kind.startsWith("SPAN_KIND_")) return kind;
  return SPAN_KIND_STRINGS[String(kind)] ?? "SPAN_KIND_UNSPECIFIED";
}

function statusCodeString(code: number | string | undefined): string {
  if (code == null) return "STATUS_CODE_UNSET";
  if (typeof code === "string" && code.startsWith("STATUS_CODE_")) return code;
  return STATUS_CODE_STRINGS[String(code)] ?? "STATUS_CODE_UNSET";
}

function normaliseId(id: string | undefined): string {
  if (!id) return "";
  const trimmed = id.trim();
  if (/^[0-9a-fA-F]+$/.test(trimmed)) return trimmed.toLowerCase();
  try {
    const hex = Buffer.from(trimmed, "base64").toString("hex");
    return hex || trimmed;
  } catch {
    return trimmed;
  }
}

function isoFromMs(ms: number): string {
  return new Date(ms).toISOString();
}
