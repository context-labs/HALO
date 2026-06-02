import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import type { Database } from "bun:sqlite";
import {
  getSpansForSession,
  getTracesForSession,
  listSessions,
  listTraces,
} from "../telemetry/storage";
import type { Span, SpanDbRow, TelemetryFilters } from "../telemetry/types";
import { LOCAL_TELEMETRY_AUTH } from "../telemetry/types";
import type { HaloRunPreview, HaloRunTargetType } from "./types";

const MAX_EXPORT_TRACES = 500;
const MAX_EXPORT_SESSIONS = 200;
const MAX_EXPORT_SPANS = 25_000;

export type HaloTraceExport = HaloRunPreview & {
  path: string;
};

export function previewHaloRunExport(
  sqlite: Database,
  input: {
    filters: TelemetryFilters;
    targetType: HaloRunTargetType;
  },
): HaloRunPreview {
  const selected = selectTraceIds(sqlite, input);
  return {
    sessionCount: selected.sessionCount,
    spanCount: selected.spanCount,
    targetType: input.targetType,
    traceCount: selected.traceIds.length,
    warnings: selected.warnings,
  };
}

export function exportHaloTraceJsonl(
  sqlite: Database,
  input: {
    filters: TelemetryFilters;
    outputDir: string;
    runId: string;
    targetType: HaloRunTargetType;
  },
): HaloTraceExport {
  const selected = selectTraceIds(sqlite, input);
  mkdirSync(input.outputDir, { recursive: true });
  const path = join(input.outputDir, "traces.jsonl");
  const rows = getSpanRowsForTraces(sqlite, selected.traceIds, MAX_EXPORT_SPANS);
  const jsonl = rows.map((row) => JSON.stringify(spanRowToHaloSpan(row))).join("\n");
  writeFileSync(path, jsonl ? `${jsonl}\n` : "", "utf8");
  return {
    path,
    sessionCount: selected.sessionCount,
    spanCount: rows.length,
    targetType: input.targetType,
    traceCount: selected.traceIds.length,
    warnings: selected.warnings,
  };
}

function selectTraceIds(
  sqlite: Database,
  input: {
    filters: TelemetryFilters;
    targetType: HaloRunTargetType;
  },
) {
  const warnings: string[] = [];
  if (input.targetType === "trace_group") {
    const result = listTraces(sqlite, {
      filters: input.filters,
      limit: MAX_EXPORT_TRACES,
      sortBy: "start_time",
      sortOrder: "desc",
    });
    if (result.totalCount > result.traces.length) {
      warnings.push(`Export capped at ${result.traces.length} of ${result.totalCount} traces.`);
    }
    return {
      sessionCount: new Set(result.traces.map((trace) => trace.sessionId).filter(Boolean))
        .size,
      spanCount: result.traces.reduce((sum, trace) => sum + trace.spanCount, 0),
      traceIds: result.traces.map((trace) => trace.traceId),
      warnings,
    };
  }

  const sessions = listSessions(sqlite, {
    filters: input.filters,
    limit: MAX_EXPORT_SESSIONS,
    sortBy: "last_activity",
    sortOrder: "desc",
  });
  if (sessions.totalCount > sessions.sessions.length) {
    warnings.push(
      `Session export capped at ${sessions.sessions.length} of ${sessions.totalCount} sessions.`,
    );
  }
  const traceIds = new Set<string>();
  let spanCount = 0;
  for (const session of sessions.sessions) {
    spanCount += session.spanCount;
    const traces = getTracesForSession(sqlite, {
      limit: MAX_EXPORT_TRACES,
      sessionId: session.sessionId,
    });
    for (const trace of traces.traces) traceIds.add(trace.traceId);
  }
  return {
    sessionCount: sessions.sessions.length,
    spanCount,
    traceIds: [...traceIds],
    warnings,
  };
}

function getSpanRowsForTraces(sqlite: Database, traceIds: string[], limit: number) {
  if (traceIds.length === 0) return [];
  const rows: SpanDbRow[] = [];
  for (const traceId of traceIds) {
    const traceRows = sqlite
      .query<Record<string, unknown>, [string, string]>(
        `SELECT *
         FROM spans
         WHERE project_id = ? AND trace_id = ?
         ORDER BY start_time ASC, span_id ASC`,
      )
      .all(LOCAL_TELEMETRY_AUTH.projectId, traceId)
      .map(rowToSpanDbRow);
    rows.push(...traceRows);
    if (rows.length >= limit) return rows.slice(0, limit);
  }
  return rows;
}

function spanRowToHaloSpan(row: SpanDbRow) {
  const resourceAttributes = parseJson<Record<string, unknown>>(
    row.resource_attributes,
    {},
  );
  const spanAttributes = parseJson<Record<string, unknown>>(row.span_attributes, {});
  const spanInts = parseJson<Record<string, unknown>>(row.span_attributes_int, {});
  const spanDoubles = parseJson<Record<string, unknown>>(
    row.span_attributes_double,
    {},
  );
  return {
    attributes: {
      ...spanAttributes,
      ...prefixMap("int", spanInts),
      ...prefixMap("double", spanDoubles),
      "halo.canvas.agent_id": row.agent_id,
      "halo.canvas.agent_name": row.agent_name,
      "halo.canvas.cache_read_tokens": row.cache_read_tokens,
      "halo.canvas.cost_total": row.cost_total,
      "halo.canvas.input": row.input,
      "halo.canvas.input_messages": row.input_messages,
      "halo.canvas.llm_model_name": row.llm_model_name,
      "halo.canvas.llm_provider": row.llm_provider,
      "halo.canvas.observation_kind": row.observation_kind,
      "halo.canvas.output": row.output,
      "halo.canvas.output_messages": row.output_messages,
      "halo.canvas.total_tokens": row.total_tokens,
      "input.value": row.input ?? row.input_messages,
      "llm.model_name": row.llm_model_name,
      "llm.provider": row.llm_provider,
      "openinference.span.kind": row.observation_kind,
      "output.value": row.output ?? row.output_messages,
    },
    end_time: new Date(row.end_time).toISOString(),
    events: parseJson(row.events_json, []),
    kind: row.span_kind,
    links: parseJson(row.links_json, []),
    name: row.span_name,
    parent_span_id: row.parent_span_id || "",
    resource: {
      attributes: {
        ...resourceAttributes,
        "deployment.environment": row.deployment_environment,
        "service.name": row.service_name,
        "service.version": row.service_version,
      },
    },
    scope: {
      name: row.scope_name,
      version: row.scope_version,
    },
    span_id: row.span_id,
    start_time: new Date(row.start_time).toISOString(),
    status: {
      code: row.status_code,
      message: row.status_message,
    },
    trace_id: row.trace_id,
    trace_state: row.trace_state,
  };
}

function rowToSpanDbRow(row: Record<string, unknown>): SpanDbRow {
  return {
    agent_id: String(row.agent_id ?? ""),
    agent_name: String(row.agent_name ?? ""),
    api_key_id: String(row.api_key_id ?? ""),
    cache_read_tokens: nullableNumber(row.cache_read_tokens),
    cache_write_tokens: nullableNumber(row.cache_write_tokens),
    chat_id: String(row.chat_id ?? ""),
    conversation_id: String(row.conversation_id ?? ""),
    cost_cache_read: nullableNumber(row.cost_cache_read),
    cost_cache_write: nullableNumber(row.cost_cache_write),
    cost_input: nullableNumber(row.cost_input),
    cost_output: nullableNumber(row.cost_output),
    cost_reasoning: nullableNumber(row.cost_reasoning),
    cost_total: nullableNumber(row.cost_total),
    deployment_environment: String(row.deployment_environment ?? ""),
    duration_ms: Number(row.duration_ms ?? 0),
    duration_ns: String(row.duration_ns ?? "0"),
    end_time: Number(row.end_time ?? 0),
    end_time_unix_nano: String(row.end_time_unix_nano ?? "0"),
    events_json: String(row.events_json ?? "[]"),
    input: nullableString(row.input),
    input_messages: nullableString(row.input_messages),
    input_tokens: nullableNumber(row.input_tokens),
    ingested_at: Number(row.ingested_at ?? 0),
    links_json: String(row.links_json ?? "[]"),
    llm_model_name: String(row.llm_model_name ?? ""),
    llm_provider: String(row.llm_provider ?? ""),
    llm_response_model: String(row.llm_response_model ?? ""),
    observation_kind: String(row.observation_kind ?? "SPAN") as Span["observationKind"],
    output: nullableString(row.output),
    output_messages: nullableString(row.output_messages),
    output_tokens: nullableNumber(row.output_tokens),
    parent_span_id: String(row.parent_span_id ?? ""),
    project_id: String(row.project_id ?? ""),
    reasoning_tokens: nullableNumber(row.reasoning_tokens),
    resource_attributes: String(row.resource_attributes ?? "{}"),
    resource_attributes_double: String(row.resource_attributes_double ?? "{}"),
    resource_attributes_int: String(row.resource_attributes_int ?? "{}"),
    retrieval_documents: nullableString(row.retrieval_documents),
    scope_name: String(row.scope_name ?? ""),
    scope_version: String(row.scope_version ?? ""),
    service_name: String(row.service_name ?? ""),
    service_version: String(row.service_version ?? ""),
    session_id: nullableString(row.session_id),
    span_attributes: String(row.span_attributes ?? "{}"),
    span_attributes_double: String(row.span_attributes_double ?? "{}"),
    span_attributes_int: String(row.span_attributes_int ?? "{}"),
    span_id: String(row.span_id ?? ""),
    span_kind: String(row.span_kind ?? "SPAN_KIND_UNSPECIFIED"),
    span_name: String(row.span_name ?? ""),
    start_time: Number(row.start_time ?? 0),
    start_time_unix_nano: String(row.start_time_unix_nano ?? "0"),
    status_code: String(row.status_code ?? "STATUS_CODE_UNSET"),
    status_message: String(row.status_message ?? ""),
    team_id: String(row.team_id ?? ""),
    total_tokens: nullableNumber(row.total_tokens),
    trace_id: String(row.trace_id ?? ""),
    trace_state: String(row.trace_state ?? ""),
    user_id: nullableString(row.user_id),
  };
}

function prefixMap(prefix: string, values: Record<string, unknown>) {
  return Object.fromEntries(Object.entries(values).map(([key, value]) => [`${prefix}.${key}`, value]));
}

function nullableString(value: unknown) {
  return value == null ? null : String(value);
}

function nullableNumber(value: unknown) {
  if (value == null) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function parseJson<T>(raw: string, fallback: T): T {
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}
