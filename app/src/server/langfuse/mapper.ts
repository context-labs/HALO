import { createHash } from "node:crypto";
import type {
  OtlpAnyValue,
  OtlpExportTraceServiceRequest,
  OtlpKeyValue,
} from "../telemetry/otlp";
import type {
  LangfuseObservation,
  LangfuseTraceWithDetails,
} from "./types";

type OtlpSpanInput = {
  attributes: OtlpKeyValue[];
  endTimeUnixNano: string;
  kind: string;
  name: string;
  parentSpanId?: string;
  spanId: string;
  startTimeUnixNano: string;
  status: { code: string; message?: string };
  traceId: string;
};

export type LangfuseImportContext = {
  baseUrl?: string;
  connectionId?: string;
  connectionName?: string;
  importedAt?: Date | number | string;
  importJobId?: string;
};

export function langfuseTraceToOtlp(
  trace: LangfuseTraceWithDetails,
  context: LangfuseImportContext = {},
): OtlpExportTraceServiceRequest {
  const traceId = toOtelTraceId(trace.id);
  const resource = traceResourceAttributes(trace);
  const observations = [...(trace.observations ?? [])].sort((a, b) => {
    const at = Date.parse(a.startTime ?? trace.timestamp ?? "");
    const bt = Date.parse(b.startTime ?? trace.timestamp ?? "");
    return (Number.isFinite(at) ? at : 0) - (Number.isFinite(bt) ? bt : 0);
  });
  const spanIdByObservationId = new Map<string, string>();
  const spans: OtlpSpanInput[] = [];

  if (observations.length === 0) {
    const rootSpanId = toOtelSpanId(`halo-root:${trace.id}`);
    const rootStart = dateToNano(trace.timestamp);
    spans.push({
      attributes: rootTraceAttributes(trace, context),
      endTimeUnixNano: rootStart,
      kind: "SPAN_KIND_INTERNAL",
      name: trace.name || "Langfuse trace",
      spanId: rootSpanId,
      startTimeUnixNano: rootStart,
      status: traceStatus(trace, observations),
      traceId,
    });
  }

  for (const observation of observations) {
    if (!observation.id) continue;
    const spanId = toOtelSpanId(observation.id);
    spanIdByObservationId.set(observation.id, spanId);
  }

  for (const observation of observations) {
    if (!observation.id) continue;
    const kind = observationKind(observation);
    const parentSpanId = observation.parentObservationId
      ? spanIdByObservationId.get(observation.parentObservationId)
      : undefined;
    const startTimeUnixNano = dateToNano(observation.startTime);
    spans.push({
      attributes: observationAttributes(observation, trace, kind, context),
      endTimeUnixNano: dateToNano(observation.endTime ?? observation.startTime),
      kind: kind === "LLM" ? "SPAN_KIND_CLIENT" : "SPAN_KIND_INTERNAL",
      name: observation.name || observation.type?.toLowerCase() || "observation",
      parentSpanId,
      spanId: spanIdByObservationId.get(observation.id) ?? toOtelSpanId(observation.id),
      startTimeUnixNano,
      status: observationStatus(observation),
      traceId,
    });
  }

  return {
    resourceSpans: [
      {
        resource: {
          attributes: compactAttributes({
            "deployment.environment": resource.deploymentEnvironment,
            "service.name": resource.serviceName,
            "service.version": resource.serviceVersion,
          }),
        },
        scopeSpans: [
          {
            scope: {
              name: "langfuse-import",
              version: "1",
            },
            spans: spans.map((span) => ({
              attributes: span.attributes,
              endTimeUnixNano: span.endTimeUnixNano,
              kind: span.kind,
              name: span.name,
              parentSpanId: span.parentSpanId,
              spanId: span.spanId,
              startTimeUnixNano: span.startTimeUnixNano,
              status: span.status,
              traceId: span.traceId,
            })),
          },
        ],
      },
    ],
  };
}

export function toOtelTraceId(id: string): string {
  const normalized = id.trim().toLowerCase();
  if (/^[0-9a-f]{32}$/.test(normalized)) return normalized;
  return sha256(normalized).slice(0, 32);
}

export function toOtelSpanId(id: string): string {
  const normalized = id.trim().toLowerCase();
  if (/^[0-9a-f]{16}$/.test(normalized)) return normalized;
  return sha256(normalized).slice(0, 16);
}

function rootTraceAttributes(
  trace: LangfuseTraceWithDetails,
  context: LangfuseImportContext,
): OtlpKeyValue[] {
  return [
    ...traceMetadataAttributes(trace, context),
    ...compactAttributes({
      "input.value": valueAsString(trace.input),
      "openinference.span.kind": "AGENT",
      "output.value": valueAsString(trace.output),
    }),
  ];
}

function traceMetadataAttributes(
  trace: LangfuseTraceWithDetails,
  context: LangfuseImportContext,
): OtlpKeyValue[] {
  const sourceUrl = langfuseTraceUrl(context.baseUrl, trace.htmlPath);
  const attrs = compactAttributes({
    "agent.name": traceAgentName(trace),
    "halo.source": "langfuse",
    "halo.source.connection_id": context.connectionId,
    "halo.source.connection_name": context.connectionName,
    "halo.source.import_job_id": context.importJobId,
    "halo.source.imported_at": importedAtIso(context.importedAt),
    "halo.source.tags": trace.tags ?? [],
    "halo.source.trace_id": trace.id,
    "halo.source.url": sourceUrl,
    "langfuse.environment": trace.environment,
    "langfuse.html_path": trace.htmlPath,
    "langfuse.project.trace_id": trace.id,
    "langfuse.release": trace.release,
    "langfuse.trace.id": trace.id,
    "langfuse.trace.input": valueAsString(trace.input),
    "langfuse.trace.metadata": valueAsString(trace.metadata),
    "langfuse.trace.name": trace.name,
    "langfuse.trace.output": valueAsString(trace.output),
    "langfuse.trace.public": trace.public,
    "langfuse.trace.tags": trace.tags ?? [],
    "langfuse.trace.url": sourceUrl,
    "langfuse.version": trace.version,
    "session.id": trace.sessionId,
    "user.id": trace.userId,
  });
  addMetadata(attrs, "langfuse.trace.metadata", trace.metadata);
  return attrs;
}

function observationAttributes(
  observation: LangfuseObservation,
  trace: LangfuseTraceWithDetails,
  kind: string,
  context: LangfuseImportContext,
): OtlpKeyValue[] {
  const model = observation.providedModelName ?? observation.model;
  const provider = inferProvider(model, observation.metadata);
  const inputTokens = usageValue(observation, "input");
  const outputTokens = usageValue(observation, "output");
  const totalTokens = usageValue(observation, "total");
  const costTotal = costValue(observation, "total") ?? observation.totalCost;

  const attrs = [
    ...traceMetadataAttributes(trace, context),
    ...compactAttributes({
      "agent.name": traceAgentName(trace),
      "input.value": valueAsString(observation.input),
      "langfuse.environment": observation.environment ?? trace.environment,
      "langfuse.observation.cost_details": valueAsString(observation.costDetails),
      "langfuse.observation.id": observation.id,
      "langfuse.observation.input": valueAsString(observation.input),
      "langfuse.observation.latency": observation.latency,
      "langfuse.observation.level": observation.level,
      "langfuse.observation.metadata": valueAsString(observation.metadata),
      "langfuse.observation.model.parameters": valueAsString(
        observation.modelParameters,
      ),
      "langfuse.observation.output": valueAsString(observation.output),
      "langfuse.observation.parent_id": observation.parentObservationId,
      "langfuse.observation.prompt.id": observation.promptId,
      "langfuse.observation.prompt.name": observation.promptName,
      "langfuse.observation.prompt.version": observation.promptVersion,
      "langfuse.observation.status_message": observation.statusMessage,
      "langfuse.observation.time_to_first_token": observation.timeToFirstToken,
      "langfuse.observation.type": observation.type,
      "langfuse.observation.usage_details": valueAsString(
        observation.usageDetails,
      ),
      "langfuse.trace.id": trace.id,
      "langfuse.trace.name": trace.name,
      "langfuse.trace.tags": trace.tags ?? [],
      "langfuse.version": observation.version ?? trace.version,
      "llm.cost.completion_details.output": costValue(observation, "output"),
      "llm.cost.prompt_details.input": costValue(observation, "input"),
      "llm.cost.total": costTotal,
      "llm.model_name": model,
      "llm.provider": provider,
      "llm.token_count.completion": outputTokens,
      "llm.token_count.prompt": inputTokens,
      "llm.token_count.total": totalTokens,
      "openinference.span.kind": kind,
      "output.value": valueAsString(observation.output),
      "session.id": trace.sessionId,
      "user.id": trace.userId,
    }),
  ];

  if (kind === "LLM") {
    attrs.push(...compactAttributes({
      "llm.input_messages": valueAsString(observation.input),
      "llm.output_messages": valueAsString(observation.output),
    }));
  }
  if (kind === "TOOL") {
    attrs.push(...compactAttributes({
      "tool.name": inferToolName(observation),
    }));
  }
  addMetadata(attrs, "langfuse.observation.metadata", observation.metadata);
  return attrs;
}

function observationKind(observation: LangfuseObservation): string {
  const type = observation.type?.toUpperCase() ?? "";
  if (type === "GENERATION" || type === "LLM" || type === "EMBEDDING") return "LLM";
  if (observation.name?.startsWith("tool_call_")) return "TOOL";
  if (
    [
      "AGENT",
      "TOOL",
      "CHAIN",
      "RETRIEVER",
      "RERANKER",
      "GUARDRAIL",
      "EVALUATOR",
      "PROMPT",
      "SPAN",
    ].includes(type)
  ) {
    return type;
  }
  if (type === "EVENT") return "SPAN";
  return "SPAN";
}

function traceStatus(
  trace: LangfuseTraceWithDetails,
  observations: LangfuseObservation[],
) {
  const failed = observations.some((observation) =>
    observationStatus(observation).code.includes("ERROR"),
  );
  return {
    code: failed ? "STATUS_CODE_ERROR" : "STATUS_CODE_OK",
    message: failed ? "One or more Langfuse observations failed" : "",
  };
}

function observationStatus(observation: LangfuseObservation) {
  const level = observation.level?.toUpperCase();
  const message = observation.statusMessage ?? "";
  const metadata = asRecord(observation.metadata);
  const success = metadata.success;
  const failed =
    level === "ERROR" ||
    level === "WARNING" ||
    Boolean(message) ||
    success === false;
  return {
    code: failed ? "STATUS_CODE_ERROR" : "STATUS_CODE_OK",
    message,
  };
}

function usageValue(observation: LangfuseObservation, key: string): number | undefined {
  const usage = asRecord(observation.usageDetails);
  const deprecated = asRecord(observation.usage);
  const aliases: Record<string, string[]> = {
    input: ["input", "prompt", "inputTokens"],
    output: ["output", "completion", "outputTokens"],
    total: ["total", "totalTokens"],
  };
  for (const source of [usage, deprecated]) {
    for (const alias of aliases[key] ?? []) {
      const found = asNumber(source[alias]);
      if (found != null) return found;
    }
  }
  return undefined;
}

function costValue(observation: LangfuseObservation, key: string): number | undefined {
  const costs = asRecord(observation.costDetails);
  const deprecated = asRecord(observation.usage);
  const aliases: Record<string, string[]> = {
    input: ["input", "prompt", "inputCost"],
    output: ["output", "completion", "outputCost"],
    total: ["total", "totalCost"],
  };
  for (const source of [costs, deprecated]) {
    for (const alias of aliases[key] ?? []) {
      const found = asNumber(source[alias]);
      if (found != null) return found;
    }
  }
  return undefined;
}

function inferProvider(model: string | null | undefined, metadata: unknown): string | undefined {
  const hidden = asRecord(asRecord(metadata).hidden_params);
  const value =
    hidden.litellm_model_name ?? asRecord(metadata).litellm_model_name ?? model;
  const text = String(value ?? "").toLowerCase();
  if (!text) return undefined;
  if (text.includes("anthropic") || text.includes("claude") || text.includes("bedrock/")) {
    return "anthropic";
  }
  if (text.includes("openai") || text.includes("gpt")) return "openai";
  if (text.includes("gemini") || text.includes("google")) return "google";
  return undefined;
}

function inferToolName(observation: LangfuseObservation): string | undefined {
  const metadata = asRecord(observation.metadata);
  if (typeof metadata.function_name === "string") return metadata.function_name;
  const name = observation.name ?? "";
  return name.startsWith("tool_call_") ? name.slice("tool_call_".length) : name;
}

function traceResourceAttributes(trace: LangfuseTraceWithDetails) {
  const resourceAttributes = asRecord(asRecord(trace.metadata).resourceAttributes);
  return {
    deploymentEnvironment:
      firstString(resourceAttributes["deployment.environment"], trace.environment) ?? "",
    serviceName:
      firstString(resourceAttributes["service.name"], trace.name) ?? "langfuse-import",
    serviceVersion:
      firstString(resourceAttributes["service.version"], trace.version, trace.release) ??
      "",
  };
}

function traceAgentName(trace: LangfuseTraceWithDetails): string {
  const traceAttributes = asRecord(asRecord(trace.metadata).attributes);
  return firstString(traceAttributes["agent.name"], trace.name) ?? "Langfuse trace";
}

function compactAttributes(values: Record<string, unknown>): OtlpKeyValue[] {
  return Object.entries(values)
    .map(([key, value]) => attribute(key, value))
    .filter((value): value is OtlpKeyValue => value != null);
}

function attribute(key: string, value: unknown): OtlpKeyValue | null {
  const encoded = anyValue(value);
  return encoded ? { key, value: encoded } : null;
}

function anyValue(value: unknown): OtlpAnyValue | null {
  if (value == null || value === "") return null;
  if (typeof value === "boolean") return { boolValue: value };
  if (typeof value === "number") {
    return Number.isInteger(value) ? { intValue: value } : { doubleValue: value };
  }
  if (typeof value === "bigint") return { intValue: value.toString() };
  if (Array.isArray(value)) {
    return { arrayValue: { values: value.map(anyValue).filter(Boolean) as OtlpAnyValue[] } };
  }
  if (typeof value === "string") return { stringValue: value };
  return { stringValue: valueAsString(value) };
}

function valueAsString(value: unknown): string | undefined {
  if (value == null || value === "") return undefined;
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function importedAtIso(value: LangfuseImportContext["importedAt"]): string {
  if (value instanceof Date) return value.toISOString();
  if (typeof value === "number") return new Date(value).toISOString();
  if (typeof value === "string" && value.trim()) {
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? new Date(parsed).toISOString() : value;
  }
  return new Date().toISOString();
}

function langfuseTraceUrl(
  baseUrl: string | undefined,
  htmlPath: string | null | undefined,
): string | undefined {
  if (!htmlPath) return undefined;
  try {
    return new URL(htmlPath, baseUrl ? `${baseUrl.replace(/\/+$/, "")}/` : undefined)
      .toString();
  } catch {
    return htmlPath;
  }
}

function addMetadata(attrs: OtlpKeyValue[], prefix: string, metadata: unknown) {
  const record = asRecord(metadata);
  for (const [key, value] of Object.entries(record)) {
    if (value == null) continue;
    attrs.push(...compactAttributes({ [`${prefix}.${key}`]: value }));
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function firstString(...values: unknown[]): string | undefined {
  for (const value of values) {
    if (typeof value !== "string") continue;
    const trimmed = value.trim();
    if (trimmed) return trimmed;
  }
  return undefined;
}

function asNumber(value: unknown): number | undefined {
  if (value == null || value === "") return undefined;
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function dateToNano(value: string | null | undefined): string {
  const parsed = Date.parse(value ?? "");
  const ms = Number.isFinite(parsed) ? parsed : Date.now();
  return (BigInt(ms) * 1_000_000n).toString();
}

function minNano(values: string[]): string {
  return values.reduce((min, value) => (BigInt(value) < BigInt(min) ? value : min));
}

function maxNano(values: string[]): string {
  return values.reduce((max, value) => (BigInt(value) > BigInt(max) ? value : max));
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}
