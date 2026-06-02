import type { Database } from "bun:sqlite";
import type { LiveEventStore } from "../live/events";
import { decodeOtlpJsonBody, buildSpanRowsFromOtlp } from "./otlp";
import type {
  FacetId,
  FacetOption,
  FilterFacets,
  NumericFacetSummary,
  ObservationKind,
  SessionSortKey,
  SessionSummary,
  Span,
  SpanDbRow,
  SpanNode,
  SpanSortKey,
  TelemetryFilters,
  Trace,
  TraceSource,
  TraceSortKey,
} from "./types";
import { LIVE_WS_URL, LOCAL_TELEMETRY_AUTH, TRACE_INGEST_URL } from "./types";

const SPAN_COLUMNS = [
  "project_id",
  "team_id",
  "api_key_id",
  "trace_id",
  "span_id",
  "parent_span_id",
  "trace_state",
  "start_time",
  "end_time",
  "start_time_unix_nano",
  "end_time_unix_nano",
  "duration_ns",
  "duration_ms",
  "ingested_at",
  "span_name",
  "span_kind",
  "service_name",
  "service_version",
  "deployment_environment",
  "scope_name",
  "scope_version",
  "status_code",
  "status_message",
  "observation_kind",
  "llm_provider",
  "llm_model_name",
  "llm_response_model",
  "input_tokens",
  "output_tokens",
  "total_tokens",
  "cache_read_tokens",
  "cache_write_tokens",
  "reasoning_tokens",
  "cost_total",
  "cost_input",
  "cost_output",
  "cost_cache_read",
  "cost_cache_write",
  "cost_reasoning",
  "user_id",
  "session_id",
  "conversation_id",
  "chat_id",
  "agent_name",
  "agent_id",
  "input_messages",
  "output_messages",
  "input",
  "output",
  "retrieval_documents",
  "resource_attributes",
  "resource_attributes_int",
  "resource_attributes_double",
  "span_attributes",
  "span_attributes_int",
  "span_attributes_double",
  "events_json",
  "links_json",
] as const;

type QueryParams = Record<string, string | number | null>;

export interface IngestTelemetryInput {
  body: string;
  contentEncoding: string;
  sizeBytes: number;
}

export interface IngestTelemetryResult {
  batchId: string;
  acceptedSpanCount: number;
  traceCount: number;
}

export type ClearTelemetryDataResult = {
  ingestBatchCount: number;
  liveEventCount: number;
  searchRowCount: number;
  spanCount: number;
  traceCount: number;
};

export function clearTelemetryData(sqlite: Database): ClearTelemetryDataResult {
  const counts: ClearTelemetryDataResult = {
    ingestBatchCount: tableCount(sqlite, "ingest_batches"),
    liveEventCount: tableCount(sqlite, "live_events"),
    searchRowCount: tableCount(sqlite, "span_search_fts"),
    spanCount: tableCount(sqlite, "spans"),
    traceCount: tableCount(sqlite, "trace_summaries"),
  };

  const transaction = sqlite.transaction(() => {
    sqlite.run("DELETE FROM span_search_fts");
    sqlite.run("DELETE FROM trace_summaries");
    sqlite.run("DELETE FROM spans");
    sqlite.run("DELETE FROM ingest_batches");
    sqlite.run("DELETE FROM live_events");
    resetAutoincrement(sqlite, [
      "ingest_batches",
      "live_events",
      "spans",
      "trace_summaries",
    ]);
  });
  transaction();
  return counts;
}

export function ingestTelemetry(
  sqlite: Database,
  input: IngestTelemetryInput,
  live?: LiveEventStore,
): IngestTelemetryResult {
  const batchId = crypto.randomUUID();
  const receivedAt = Date.now();
  const decoded = decodeOtlpJsonBody(input.body);
  const rows = buildSpanRowsFromOtlp(decoded, receivedAt);
  const traceIds = new Set(rows.map((row) => row.trace_id).filter(Boolean));

  const transaction = sqlite.transaction(() => {
    for (const row of rows) {
      upsertSpan(sqlite, row);
      upsertSearch(sqlite, row);
    }

    for (const traceId of traceIds) {
      refreshTraceSummary(sqlite, traceId, receivedAt);
    }

    sqlite
      .query(
        `INSERT INTO ingest_batches (
          id, received_at, size_bytes, accepted_span_count, trace_count,
          content_encoding, status, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, 'accepted', NULL)`,
      )
      .run(
        batchId,
        receivedAt,
        input.sizeBytes,
        rows.length,
        traceIds.size,
        input.contentEncoding,
      );
  });

  transaction();

  if (live) {
    publishLiveIngestEvents(sqlite, live, {
      acceptedSpanCount: rows.length,
      batchId,
      receivedAt,
      rows,
      sizeBytes: input.sizeBytes,
      traceIds: [...traceIds],
    });
  }

  return {
    acceptedSpanCount: rows.length,
    batchId,
    traceCount: traceIds.size,
  };
}

function publishLiveIngestEvents(
  sqlite: Database,
  live: LiveEventStore,
  input: {
    acceptedSpanCount: number;
    batchId: string;
    receivedAt: number;
    rows: Pick<SpanDbRow, "span_id" | "trace_id">[];
    sizeBytes: number;
    traceIds: string[];
  },
) {
  const uniqueRows = new Map<string, { span_id: string; trace_id: string }>();
  for (const row of input.rows) {
    uniqueRows.set(`${row.trace_id}:${row.span_id}`, row);
  }

  for (const row of uniqueRows.values()) {
    const span = getSpan(sqlite, {
      spanId: row.span_id,
      traceId: row.trace_id,
    });
    if (!span) continue;
    live.publish({
      eventType: "span.upserted",
      payload: {
        span,
        type: "span.upserted",
      },
      traceId: span.traceId,
    });
  }

  for (const traceId of input.traceIds) {
    const trace = getTrace(sqlite, traceId);
    if (!trace) continue;
    live.publish({
      eventType: "trace.upserted",
      payload: {
        trace,
        type: "trace.upserted",
      },
      traceId: trace.traceId,
    });
  }

  live.publish({
    eventType: "ingest.accepted",
    payload: {
      acceptedSpanCount: input.acceptedSpanCount,
      batchId: input.batchId,
      receivedAt: new Date(input.receivedAt).toISOString(),
      sizeBytes: input.sizeBytes,
      traceCount: input.traceIds.length,
      traceIds: input.traceIds,
      type: "ingest.accepted",
    },
  });

  live.publish({
    eventType: "telemetry.changed",
    payload: {
      acceptedSpanCount: input.acceptedSpanCount,
      batchId: input.batchId,
      traceCount: input.traceIds.length,
      traceIds: input.traceIds,
      type: "telemetry.changed",
    },
  });
}

function upsertSpan(sqlite: Database, row: SpanDbRow) {
  const placeholders = SPAN_COLUMNS.map(() => "?").join(", ");
  const updateAssignments = SPAN_COLUMNS.filter(
    (column) => !["project_id", "trace_id", "span_id"].includes(column),
  )
    .map((column) => `${column} = excluded.${column}`)
    .join(", ");
  const values = SPAN_COLUMNS.map((column) => row[column]);

  sqlite
    .query(
      `INSERT INTO spans (${SPAN_COLUMNS.join(", ")})
       VALUES (${placeholders})
       ON CONFLICT(project_id, trace_id, span_id)
       DO UPDATE SET ${updateAssignments}`,
    )
    .run(...values);
}

function upsertSearch(sqlite: Database, row: SpanDbRow) {
  sqlite
    .query(
      `DELETE FROM span_search_fts
       WHERE rowid IN (
         SELECT rowid FROM span_search_fts
         WHERE project_id = ? AND trace_id = ? AND span_id = ?
       )`,
    )
    .run(row.project_id, row.trace_id, row.span_id);

  sqlite
    .query(
      `INSERT INTO span_search_fts (project_id, trace_id, span_id, content)
       VALUES (?, ?, ?, ?)`,
    )
    .run(row.project_id, row.trace_id, row.span_id, buildSearchContent(row));
}

function buildSearchContent(row: SpanDbRow): string {
  return [
    row.trace_id,
    row.span_id,
    row.parent_span_id,
    row.span_name,
    row.service_name,
    row.scope_name,
    row.status_message,
    row.observation_kind,
    row.llm_provider,
    row.llm_model_name,
    row.llm_response_model,
    row.user_id,
    row.session_id,
    row.agent_name,
    row.agent_id,
    row.input_messages,
    row.output_messages,
    row.input,
    row.output,
    row.retrieval_documents,
    row.resource_attributes,
    row.span_attributes,
    row.events_json,
  ]
    .filter((value) => value != null && value !== "")
    .join("\n");
}

function refreshTraceSummary(sqlite: Database, traceId: string, updatedAt: number) {
  const projectId = LOCAL_TELEMETRY_AUTH.projectId;
  const aggregate = sqlite
    .query<{
      start_time: number;
      end_time: number;
      span_count: number;
      llm_span_count: number;
      total_tokens: number | null;
      cache_read_tokens: number | null;
      total_cost: number | null;
      has_error: number;
    }, [string, string]>(
      `SELECT
        min(start_time) AS start_time,
        max(end_time) AS end_time,
        count(*) AS span_count,
        sum(CASE WHEN observation_kind = 'LLM' THEN 1 ELSE 0 END) AS llm_span_count,
        sum(total_tokens) AS total_tokens,
        sum(cache_read_tokens) AS cache_read_tokens,
        sum(cost_total) AS total_cost,
        max(CASE WHEN status_code = 'STATUS_CODE_ERROR' THEN 1 ELSE 0 END) AS has_error
       FROM spans
       WHERE project_id = ? AND trace_id = ?`,
    )
    .get(projectId, traceId);

  if (!aggregate || aggregate.span_count === 0) return;

  const rootCandidates = sqlite
    .query<Record<string, string | number | null>, [string, string]>(
      `SELECT *
       FROM spans
       WHERE project_id = ? AND trace_id = ? AND parent_span_id = ''
       ORDER BY start_time ASC, span_id ASC`,
    )
    .all(projectId, traceId);
  const root =
    chooseTraceRoot(rootCandidates) ??
    sqlite
      .query<Record<string, string | number | null>, [string, string]>(
        `SELECT *
         FROM spans
         WHERE project_id = ? AND trace_id = ?
         ORDER BY CASE WHEN parent_span_id = '' THEN 0 ELSE 1 END, start_time ASC, span_id ASC
         LIMIT 1`,
      )
      .get(projectId, traceId) ?? {};

  const durationMs = Math.max(0, aggregate.end_time - aggregate.start_time);
  const durationNs = String(Math.round(durationMs * 1_000_000));
  const existingSource = sqlite
    .query<{ source: string }, [string, string]>(
      `SELECT source
       FROM trace_summaries
       WHERE project_id = ? AND trace_id = ?
       LIMIT 1`,
    )
    .get(projectId, traceId)?.source;
  const source =
    existingSource === "local"
      ? localTraceSourceMetadata()
      : traceSourceFromRows(rootCandidates.length > 0 ? rootCandidates : [root]);

  sqlite
    .query(
      `INSERT INTO trace_summaries (
        project_id, trace_id, session_id, start_time, end_time, duration_ns,
        duration_ms, root_span_name, root_observation_kind, span_count,
        llm_span_count, total_tokens, cache_read_tokens, total_cost, has_error,
        service_name, service_version, deployment_environment, agent_name,
        agent_id, source, source_trace_id, source_connection_id,
        source_connection_name, source_import_job_id, source_imported_at,
        source_url, source_tags_json, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(project_id, trace_id) DO UPDATE SET
        session_id = excluded.session_id,
        start_time = excluded.start_time,
        end_time = excluded.end_time,
        duration_ns = excluded.duration_ns,
        duration_ms = excluded.duration_ms,
        root_span_name = excluded.root_span_name,
        root_observation_kind = excluded.root_observation_kind,
        span_count = excluded.span_count,
        llm_span_count = excluded.llm_span_count,
        total_tokens = excluded.total_tokens,
        cache_read_tokens = excluded.cache_read_tokens,
        total_cost = excluded.total_cost,
        has_error = excluded.has_error,
        service_name = excluded.service_name,
        service_version = excluded.service_version,
        deployment_environment = excluded.deployment_environment,
        agent_name = excluded.agent_name,
        agent_id = excluded.agent_id,
        source = excluded.source,
        source_trace_id = excluded.source_trace_id,
        source_connection_id = excluded.source_connection_id,
        source_connection_name = excluded.source_connection_name,
        source_import_job_id = excluded.source_import_job_id,
        source_imported_at = excluded.source_imported_at,
        source_url = excluded.source_url,
        source_tags_json = excluded.source_tags_json,
        updated_at = excluded.updated_at`,
    )
    .run(
      projectId,
      traceId,
      emptyToNull(String(root.session_id ?? "")),
      aggregate.start_time,
      aggregate.end_time,
      durationNs,
      durationMs,
      String(root.span_name ?? ""),
      String(root.observation_kind ?? "SPAN"),
      aggregate.span_count,
      aggregate.llm_span_count,
      aggregate.total_tokens,
      aggregate.cache_read_tokens,
      aggregate.total_cost,
      aggregate.has_error > 0 ? 1 : 0,
      String(root.service_name ?? ""),
      String(root.service_version ?? ""),
      String(root.deployment_environment ?? ""),
      String(root.agent_name ?? ""),
      String(root.agent_id ?? ""),
      source.source,
      source.sourceTraceId,
      source.sourceConnectionId,
      source.sourceConnectionName,
      source.sourceImportJobId,
      source.sourceImportedAt,
      source.sourceUrl,
      JSON.stringify(source.sourceTags),
      updatedAt,
    );
}

type TraceSourceMetadata = {
  source: TraceSource;
  sourceTraceId: string | null;
  sourceConnectionId: string | null;
  sourceConnectionName: string | null;
  sourceImportJobId: string | null;
  sourceImportedAt: number | null;
  sourceUrl: string | null;
  sourceTags: string[];
};

function chooseTraceRoot(
  rows: Array<Record<string, string | number | null>>,
): Record<string, string | number | null> | null {
  return (
    rows.find((row) => sourceMetadataFromRow(row).source !== "langfuse") ??
    rows[0] ??
    null
  );
}

function traceSourceFromRows(
  rows: Array<Record<string, string | number | null>>,
): TraceSourceMetadata {
  const importMetadata = rows
    .map(sourceMetadataFromRow)
    .find((metadata) => metadata.source === "langfuse");
  const hasLocalRoot = rows.some(
    (row) =>
      String(row.parent_span_id ?? "") === "" &&
      sourceMetadataFromRow(row).source !== "langfuse",
  );
  if (importMetadata && !hasLocalRoot) return importMetadata;
  return localTraceSourceMetadata();
}

function sourceMetadataFromRow(
  row: Record<string, string | number | null>,
): TraceSourceMetadata {
  const spanAttrs = parseJson<Record<string, unknown>>(row.span_attributes, {});
  const resourceAttrs = parseJson<Record<string, unknown>>(
    row.resource_attributes,
    {},
  );
  const source = asTraceSource(
    firstString(spanAttrs["halo.source"], resourceAttrs["halo.source"]) ?? "local",
  );
  if (source !== "langfuse") return localTraceSourceMetadata();

  return {
    source,
    sourceConnectionId: firstString(
      spanAttrs["halo.source.connection_id"],
      resourceAttrs["halo.source.connection_id"],
    ),
    sourceConnectionName: firstString(
      spanAttrs["halo.source.connection_name"],
      resourceAttrs["halo.source.connection_name"],
    ),
    sourceImportJobId: firstString(
      spanAttrs["halo.source.import_job_id"],
      resourceAttrs["halo.source.import_job_id"],
    ),
    sourceImportedAt: timestampMsFromValue(
      firstString(spanAttrs["halo.source.imported_at"]),
    ),
    sourceTags: firstStringArray(
      spanAttrs["halo.source.tags"],
      spanAttrs["langfuse.trace.tags"],
    ),
    sourceTraceId: firstString(
      spanAttrs["halo.source.trace_id"],
      spanAttrs["langfuse.trace.id"],
      spanAttrs["langfuse.project.trace_id"],
    ),
    sourceUrl: firstString(spanAttrs["halo.source.url"], spanAttrs["langfuse.trace.url"]),
  };
}

function localTraceSourceMetadata(): TraceSourceMetadata {
  return {
    source: "local",
    sourceConnectionId: null,
    sourceConnectionName: null,
    sourceImportJobId: null,
    sourceImportedAt: null,
    sourceTags: [],
    sourceTraceId: null,
    sourceUrl: null,
  };
}

function firstString(...values: unknown[]): string | null {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value.trim();
    if (typeof value === "number" && Number.isFinite(value)) return String(value);
  }
  return null;
}

function firstStringArray(...values: unknown[]): string[] {
  for (const value of values) {
    const array = parseStringArray(value);
    if (array.length > 0) return array;
  }
  return [];
}

function timestampMsFromValue(value: string | null): number | null {
  if (!value) return null;
  const asNumber = Number(value);
  if (Number.isFinite(asNumber)) return asNumber;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function getTelemetryInfo(
  sqlite: Database,
  dbPath: string,
  liveUrl = LIVE_WS_URL,
) {
  const counts = sqlite
    .query<{ trace_count: number; span_count: number }, []>(
      `SELECT
        (SELECT count(*) FROM trace_summaries) AS trace_count,
        (SELECT count(*) FROM spans) AS span_count`,
    )
    .get();
  const lastBatch =
    sqlite
      .query<
        {
          id: string;
          received_at: number;
          size_bytes: number;
          accepted_span_count: number;
          trace_count: number;
          status: string;
        },
        []
      >(
        `SELECT id, received_at, size_bytes, accepted_span_count, trace_count, status
         FROM ingest_batches
         ORDER BY received_at DESC
         LIMIT 1`,
      )
      .get() ?? null;

  return {
    dbPath,
    ingestUrl: TRACE_INGEST_URL,
    lastBatch:
      lastBatch == null
        ? null
        : {
            acceptedSpanCount: lastBatch.accepted_span_count,
            id: lastBatch.id,
            receivedAt: isoFromMs(lastBatch.received_at),
            sizeBytes: lastBatch.size_bytes,
            status: lastBatch.status,
            traceCount: lastBatch.trace_count,
          },
    projectId: LOCAL_TELEMETRY_AUTH.projectId,
    liveEnabled: true,
    liveUrl,
    spanCount: counts?.span_count ?? 0,
    traceCount: counts?.trace_count ?? 0,
  };
}

export function listTraces(
  sqlite: Database,
  input: {
    filters?: TelemetryFilters;
    sortBy?: TraceSortKey;
    sortOrder?: "asc" | "desc";
    cursor?: string | null;
    limit?: number;
  },
) {
  const limit = clampLimit(input.limit);
  const sortColumn = traceSortColumn(input.sortBy);
  const order = input.sortOrder === "asc" ? "ASC" : "DESC";
  const { conditions, params } = buildTraceWhere(input.filters);
  const cursor = decodeCursor(input.cursor);
  if (cursor) {
    conditions.push(
      input.sortOrder === "asc"
        ? `(${sortColumn}, trace_id) > (:cursorValue, :cursorId)`
        : `(${sortColumn}, trace_id) < (:cursorValue, :cursorId)`,
    );
    params.cursorValue = cursor.value;
    params.cursorId = cursor.id;
  }
  const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";
  const rows = sqlite
    .query<Record<string, unknown>, QueryParams>(
      `SELECT * FROM trace_summaries
       ${where}
       ORDER BY ${sortColumn} ${order}, trace_id ${order}
       LIMIT :limitPlusOne`,
    )
    .all({ ...params, limitPlusOne: limit + 1 });
  const count = sqlite
    .query<{ c: number }, QueryParams>(
      `SELECT count(*) AS c FROM trace_summaries ${where}`,
    )
    .get(params);
  const hasMore = rows.length > limit;
  const trimmed = hasMore ? rows.slice(0, limit) : rows;
  return {
    nextCursor: hasMore
      ? encodeCursor({
          id: String(trimmed.at(-1)?.trace_id ?? ""),
          value: trimmed.at(-1)?.[sortColumn] as string | number,
        })
      : null,
    totalCount: count?.c ?? 0,
    traces: trimmed.map(mapTrace),
  };
}

export function getTrace(sqlite: Database, traceId: string): Trace | null {
  const row = sqlite
    .query<Record<string, unknown>, [string, string]>(
      `SELECT * FROM trace_summaries
       WHERE project_id = ? AND trace_id = ?
       LIMIT 1`,
    )
    .get(LOCAL_TELEMETRY_AUTH.projectId, traceId);
  return row ? mapTrace(row) : null;
}

export function getSpansForTrace(
  sqlite: Database,
  input: { traceId: string; cursor?: string | null; limit?: number },
) {
  const limit = clampLimit(input.limit, 250);
  const cursor = decodeCursor(input.cursor);
  const params: QueryParams = {
    limitPlusOne: limit + 1,
    projectId: LOCAL_TELEMETRY_AUTH.projectId,
    traceId: input.traceId,
  };
  const cursorSql = cursor
    ? "AND (start_time, span_id) > (:cursorValue, :cursorId)"
    : "";
  if (cursor) {
    params.cursorValue = cursor.value;
    params.cursorId = cursor.id;
  }
  const rows = sqlite
    .query<Record<string, unknown>, QueryParams>(
      `SELECT * FROM spans
       WHERE project_id = :projectId AND trace_id = :traceId
       ${cursorSql}
       ORDER BY start_time ASC, span_id ASC
       LIMIT :limitPlusOne`,
    )
    .all(params);
  const hasMore = rows.length > limit;
  const trimmed = hasMore ? rows.slice(0, limit) : rows;
  return {
    nextCursor: hasMore
      ? encodeCursor({
          id: String(trimmed.at(-1)?.span_id ?? ""),
          value: trimmed.at(-1)?.start_time as string | number,
        })
      : null,
    spans: trimmed.map(mapSpan),
  };
}

export function getSpan(
  sqlite: Database,
  input: { traceId: string; spanId: string },
): Span | null {
  const row = sqlite
    .query<Record<string, unknown>, [string, string, string]>(
      `SELECT * FROM spans
       WHERE project_id = ? AND trace_id = ? AND span_id = ?
       LIMIT 1`,
    )
    .get(LOCAL_TELEMETRY_AUTH.projectId, input.traceId, input.spanId);
  return row ? mapSpan(row) : null;
}

export function listSpans(
  sqlite: Database,
  input: {
    filters?: TelemetryFilters;
    sortBy?: SpanSortKey;
    sortOrder?: "asc" | "desc";
    cursor?: string | null;
    limit?: number;
  },
) {
  const limit = clampLimit(input.limit);
  const sortColumn = spanSortColumn(input.sortBy);
  const order = input.sortOrder === "asc" ? "ASC" : "DESC";
  const { conditions, params } = buildSpanWhere(input.filters);
  const cursor = decodeCursor(input.cursor);
  if (cursor) {
    conditions.push(
      input.sortOrder === "asc"
        ? `(${sortColumn}, span_id) > (:cursorValue, :cursorId)`
        : `(${sortColumn}, span_id) < (:cursorValue, :cursorId)`,
    );
    params.cursorValue = cursor.value;
    params.cursorId = cursor.id;
  }
  const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";
  const rows = sqlite
    .query<Record<string, unknown>, QueryParams>(
      `SELECT * FROM spans
       ${where}
       ORDER BY ${sortColumn} ${order}, span_id ${order}
       LIMIT :limitPlusOne`,
    )
    .all({ ...params, limitPlusOne: limit + 1 });
  const count = sqlite
    .query<{ c: number }, QueryParams>(`SELECT count(*) AS c FROM spans ${where}`)
    .get(params);
  const hasMore = rows.length > limit;
  const trimmed = hasMore ? rows.slice(0, limit) : rows;
  return {
    nextCursor: hasMore
      ? encodeCursor({
          id: String(trimmed.at(-1)?.span_id ?? ""),
          value: trimmed.at(-1)?.[sortColumn] as string | number,
        })
      : null,
    spans: trimmed.map(mapSpan),
    totalCount: count?.c ?? 0,
  };
}

export function searchTraces(
  sqlite: Database,
  input: { query: string; filters?: TelemetryFilters; limit?: number; cursor?: string | null },
) {
  const q = input.query.trim();
  if (!q) {
    return { nextCursor: null, results: [], totalCount: 0, warnings: [] };
  }

  const limit = clampLimit(input.limit);
  const matchedTraceIds = searchTraceIds(sqlite, q, limit * 10);
  if (matchedTraceIds.length === 0) {
    return { nextCursor: null, results: [], totalCount: 0, warnings: [] };
  }

  const { conditions, params } = buildTraceWhere(input.filters);
  conditions.push(`trace_id IN (${matchedTraceIds.map((_, i) => `:trace${i}`).join(", ")})`);
  matchedTraceIds.forEach((id, i) => {
    params[`trace${i}`] = id;
  });
  const rows = sqlite
    .query<Record<string, unknown>, QueryParams>(
      `SELECT * FROM trace_summaries
       WHERE ${conditions.join(" AND ")}
       ORDER BY start_time DESC, trace_id DESC
       LIMIT :limit`,
    )
    .all({ ...params, limit });
  return {
    nextCursor: null,
    results: rows.map((row) => ({
      matchedSpanCount: 1,
      score: 100,
      topMatches: [],
      trace: mapTrace(row),
    })),
    totalCount: rows.length,
    warnings: [],
  };
}

export function listSessions(
  sqlite: Database,
  input: {
    filters?: TelemetryFilters;
    sortBy?: SessionSortKey;
    sortOrder?: "asc" | "desc";
    cursor?: string | null;
    limit?: number;
  },
) {
  const limit = clampLimit(input.limit);
  const sortColumn = sessionSortColumn(input.sortBy);
  const order = input.sortOrder === "asc" ? "ASC" : "DESC";
  const { conditions, params } = buildSessionWhere(input.filters);
  const cursor = decodeCursor(input.cursor);
  const cursorSql = cursor
    ? input.sortOrder === "asc"
      ? `WHERE (${sortColumn}, session_id) > (:cursorValue, :cursorId)`
      : `WHERE (${sortColumn}, session_id) < (:cursorValue, :cursorId)`
    : "";
  if (cursor) {
    params.cursorValue = cursor.value;
    params.cursorId = cursor.id;
  }
  const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";
  const rows = sqlite
    .query<Record<string, unknown>, QueryParams>(
      `${sessionAggregateSql(where)}
       SELECT *
       FROM session_rows
       ${cursorSql}
       ORDER BY ${sortColumn} ${order}, session_id ${order}
       LIMIT :limitPlusOne`,
    )
    .all({ ...params, limitPlusOne: limit + 1 });
  const count = sqlite
    .query<{ c: number }, QueryParams>(
      `SELECT count(DISTINCT session_id) AS c
       FROM trace_summaries
       ${where}`,
    )
    .get(params);
  const hasMore = rows.length > limit;
  const trimmed = hasMore ? rows.slice(0, limit) : rows;
  return {
    nextCursor: hasMore
      ? encodeCursor({
          id: String(trimmed.at(-1)?.session_id ?? ""),
          value: trimmed.at(-1)?.[sortColumn] as string | number,
        })
      : null,
    sessions: trimmed.map(mapSessionSummary),
    totalCount: count?.c ?? 0,
  };
}

export function searchSessions(
  sqlite: Database,
  input: { query: string; filters?: TelemetryFilters; limit?: number; cursor?: string | null },
) {
  const q = input.query.trim();
  if (!q) {
    return { nextCursor: null, results: [], totalCount: 0, warnings: [] };
  }

  const matchedSessionIds = searchSessionIds(sqlite, q, clampLimit(input.limit) * 10);
  if (matchedSessionIds.length === 0) {
    return { nextCursor: null, results: [], totalCount: 0, warnings: [] };
  }

  const sessionIds = intersectValues(input.filters?.sessionIds, matchedSessionIds);
  if (sessionIds.length === 0) {
    return { nextCursor: null, results: [], totalCount: 0, warnings: [] };
  }
  const filters = {
    ...input.filters,
    sessionIds,
  };
  const result = listSessions(sqlite, {
    cursor: input.cursor,
    filters,
    limit: input.limit,
    sortBy: "last_activity",
    sortOrder: "desc",
  });
  return {
    nextCursor: result.nextCursor,
    results: result.sessions.map((session) => ({
      matchedTraceCount: session.traceCount,
      score: 100,
      session,
    })),
    totalCount: result.totalCount,
    warnings: [],
  };
}

export function getSession(
  sqlite: Database,
  sessionId: string,
): SessionSummary | null {
  const { conditions, params } = buildSessionWhere({ sessionIds: [sessionId] });
  const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";
  const row = sqlite
    .query<Record<string, unknown>, QueryParams>(
      `${sessionAggregateSql(where)}
       SELECT *
       FROM session_rows
       LIMIT 1`,
    )
    .get(params);
  return row ? mapSessionSummary(row) : null;
}

export function getTracesForSession(
  sqlite: Database,
  input: { sessionId: string; cursor?: string | null; limit?: number },
) {
  const limit = clampLimit(input.limit, 500);
  const cursor = decodeCursor(input.cursor);
  const params: QueryParams = {
    limitPlusOne: limit + 1,
    projectId: LOCAL_TELEMETRY_AUTH.projectId,
    sessionId: input.sessionId,
  };
  const cursorSql = cursor
    ? "AND (start_time, trace_id) > (:cursorValue, :cursorId)"
    : "";
  if (cursor) {
    params.cursorValue = cursor.value;
    params.cursorId = cursor.id;
  }
  const rows = sqlite
    .query<Record<string, unknown>, QueryParams>(
      `SELECT *
       FROM trace_summaries
       WHERE project_id = :projectId
         AND session_id = :sessionId
       ${cursorSql}
       ORDER BY start_time ASC, trace_id ASC
       LIMIT :limitPlusOne`,
    )
    .all(params);
  const hasMore = rows.length > limit;
  const trimmed = hasMore ? rows.slice(0, limit) : rows;
  return {
    nextCursor: hasMore
      ? encodeCursor({
          id: String(trimmed.at(-1)?.trace_id ?? ""),
          value: trimmed.at(-1)?.start_time as string | number,
        })
      : null,
    totalCount:
      sqlite
        .query<{ c: number }, [string, string]>(
          `SELECT count(*) AS c
           FROM trace_summaries
           WHERE project_id = ? AND session_id = ?`,
        )
        .get(LOCAL_TELEMETRY_AUTH.projectId, input.sessionId)?.c ?? 0,
    traces: trimmed.map(mapTrace),
  };
}

export function getSpansForSession(
  sqlite: Database,
  input: { sessionId: string; cursor?: string | null; limit?: number },
) {
  const limit = clampLimit(input.limit, 1000);
  const cursor = decodeSessionSpanCursor(input.cursor);
  const params: QueryParams = {
    limitPlusOne: limit + 1,
    projectId: LOCAL_TELEMETRY_AUTH.projectId,
    sessionId: input.sessionId,
  };
  const cursorSql = cursor
    ? `AND (s.start_time, s.trace_id, s.span_id) >
       (:cursorValue, :cursorTraceId, :cursorSpanId)`
    : "";
  if (cursor) {
    params.cursorValue = cursor.value;
    params.cursorTraceId = cursor.traceId;
    params.cursorSpanId = cursor.spanId;
  }
  const rows = sqlite
    .query<Record<string, unknown>, QueryParams>(
      `SELECT s.*
       FROM spans s
       JOIN trace_summaries ts
        ON ts.project_id = s.project_id AND ts.trace_id = s.trace_id
       WHERE s.project_id = :projectId
         AND ts.session_id = :sessionId
       ${cursorSql}
       ORDER BY s.start_time ASC, s.trace_id ASC, s.span_id ASC
       LIMIT :limitPlusOne`,
    )
    .all(params);
  const hasMore = rows.length > limit;
  const trimmed = hasMore ? rows.slice(0, limit) : rows;
  return {
    nextCursor: hasMore
      ? encodeSessionSpanCursor({
          spanId: String(trimmed.at(-1)?.span_id ?? ""),
          traceId: String(trimmed.at(-1)?.trace_id ?? ""),
          value: trimmed.at(-1)?.start_time as string | number,
        })
      : null,
    spans: trimmed.map(mapSpan),
  };
}

export function getSessionFacets(
  sqlite: Database,
  facetIds: FacetId[],
): FilterFacets {
  const out: FilterFacets = { attributeKeys: {}, categorical: {}, numeric: {} };
  for (const facet of facetIds) {
    if (facet === "status") {
      out.categorical[facet] = sessionStatusFacet(sqlite);
      continue;
    }
    const column = facetColumn(facet);
    if (column) {
      out.categorical[facet] = isTraceSummaryCategoricalColumn(column)
        ? sessionCategoricalFacet(sqlite, column)
        : sessionSpanBackedFacet(sqlite, column);
      continue;
    }
    const numeric = numericFacetColumn(facet, "trace");
    if (numeric) {
      out.numeric[facet] = sessionNumericFacet(sqlite, numeric);
    }
  }
  return out;
}

function searchTraceIds(sqlite: Database, q: string, limit: number): string[] {
  const ftsQuery = q
    .split(/\s+/)
    .map((term) => term.replace(/[^a-zA-Z0-9_./:-]/g, ""))
    .filter(Boolean)
    .map((term) => `${term}*`)
    .join(" ");

  if (ftsQuery) {
    try {
      return sqlite
        .query<{ trace_id: string }, [string, string, number]>(
          `SELECT DISTINCT trace_id
           FROM span_search_fts
           WHERE span_search_fts MATCH ? AND project_id = ?
           LIMIT ?`,
        )
        .all(ftsQuery, LOCAL_TELEMETRY_AUTH.projectId, limit)
        .map((row) => row.trace_id);
    } catch {
      // Fall through to LIKE search if the user typed FTS syntax.
    }
  }

  return sqlite
    .query<{ trace_id: string }, [string, string, number]>(
      `SELECT DISTINCT trace_id
       FROM span_search_fts
       WHERE content LIKE ? AND project_id = ?
       LIMIT ?`,
    )
    .all(`%${q}%`, LOCAL_TELEMETRY_AUTH.projectId, limit)
    .map((row) => row.trace_id);
}

function searchSessionIds(sqlite: Database, q: string, limit: number): string[] {
  const ftsQuery = q
    .split(/\s+/)
    .map((term) => term.replace(/[^a-zA-Z0-9_./:-]/g, ""))
    .filter(Boolean)
    .map((term) => `${term}*`)
    .join(" ");
  const found = new Set<string>();

  if (ftsQuery) {
    try {
      for (const row of sqlite
        .query<{ session_id: string }, [string, string, number]>(
          `SELECT DISTINCT ts.session_id
           FROM span_search_fts
           JOIN trace_summaries ts
            ON ts.project_id = span_search_fts.project_id
            AND ts.trace_id = span_search_fts.trace_id
           WHERE span_search_fts MATCH ?
             AND ts.project_id = ?
             AND ts.session_id IS NOT NULL
             AND ts.session_id != ''
           LIMIT ?`,
        )
        .all(ftsQuery, LOCAL_TELEMETRY_AUTH.projectId, limit)) {
        found.add(row.session_id);
      }
    } catch {
      // Fall through to LIKE search if the user typed FTS syntax.
    }
  }

  for (const row of sqlite
    .query<
      { session_id: string },
      [string, string, string, string, string, string, string, string, number]
    >(
      `SELECT DISTINCT ts.session_id
       FROM trace_summaries ts
       LEFT JOIN spans s
        ON s.project_id = ts.project_id AND s.trace_id = ts.trace_id
       WHERE ts.project_id = ?
         AND ts.session_id IS NOT NULL
         AND ts.session_id != ''
         AND (
          ts.session_id LIKE ?
          OR ts.root_span_name LIKE ?
          OR ts.service_name LIKE ?
          OR ts.agent_name LIKE ?
          OR s.llm_model_name LIKE ?
          OR s.input LIKE ?
          OR s.output LIKE ?
         )
       LIMIT ?`,
    )
    .all(
      LOCAL_TELEMETRY_AUTH.projectId,
      `%${q}%`,
      `%${q}%`,
      `%${q}%`,
      `%${q}%`,
      `%${q}%`,
      `%${q}%`,
      `%${q}%`,
      limit,
    )) {
    found.add(row.session_id);
    if (found.size >= limit) break;
  }

  return [...found].slice(0, limit);
}

export function getTraceFacets(sqlite: Database, facetIds: FacetId[]): FilterFacets {
  return getFacets(sqlite, "trace", facetIds);
}

export function getSpanFacets(sqlite: Database, facetIds: FacetId[]): FilterFacets {
  return getFacets(sqlite, "span", facetIds);
}

function getFacets(
  sqlite: Database,
  mode: "trace" | "span",
  facetIds: FacetId[],
): FilterFacets {
  const table = mode === "trace" ? "trace_summaries" : "spans";
  const out: FilterFacets = { attributeKeys: {}, categorical: {}, numeric: {} };
  for (const facet of facetIds) {
    if (facet === "status") {
      out.categorical[facet] =
        mode === "trace" ? traceStatusFacet(sqlite) : spanStatusFacet(sqlite);
      continue;
    }
    if (facet === "source" && mode === "span") {
      out.categorical[facet] = spanSourceFacet(sqlite);
      continue;
    }

    const column = facetColumn(facet);
    if (column) {
      out.categorical[facet] =
        mode === "trace" && !isTraceSummaryCategoricalColumn(column)
          ? spanBackedTraceFacet(sqlite, column)
          : categoricalFacet(sqlite, table, column);
      continue;
    }
    const numeric = numericFacetColumn(facet, mode);
    if (numeric) {
      out.numeric[facet] = numericFacet(sqlite, table, numeric);
      continue;
    }
    if (facet === "span_attributes" && mode === "span") {
      out.attributeKeys.span = attributeKeyFacet(sqlite, "span_attributes");
    }
    if (facet === "resource_attributes" && mode === "span") {
      out.attributeKeys.resource = attributeKeyFacet(sqlite, "resource_attributes");
    }
  }
  return out;
}

function isTraceSummaryCategoricalColumn(column: string): boolean {
  return new Set([
    "agent_id",
    "agent_name",
    "deployment_environment",
    "service_name",
    "session_id",
    "source",
  ]).has(column);
}

function categoricalFacet(sqlite: Database, table: string, column: string): FacetOption[] {
  return sqlite
    .query<{ value: string; count: number }, [string]>(
      `SELECT ${column} AS value, count(*) AS count
       FROM ${table}
       WHERE project_id = ? AND ${column} IS NOT NULL AND ${column} != ''
       GROUP BY ${column}
       ORDER BY count DESC, value ASC
       LIMIT 50`,
    )
    .all(LOCAL_TELEMETRY_AUTH.projectId)
    .map((row) => ({ count: row.count, label: row.value, value: row.value }));
}

function spanBackedTraceFacet(sqlite: Database, column: string): FacetOption[] {
  return sqlite
    .query<{ value: string; count: number }, [string]>(
      `SELECT ${column} AS value, count(DISTINCT trace_id) AS count
       FROM spans
       WHERE project_id = ? AND ${column} IS NOT NULL AND ${column} != ''
       GROUP BY ${column}
       ORDER BY count DESC, value ASC
       LIMIT 50`,
    )
    .all(LOCAL_TELEMETRY_AUTH.projectId)
    .map((row) => ({ count: row.count, label: row.value, value: row.value }));
}

function traceStatusFacet(sqlite: Database): FacetOption[] {
  return sqlite
    .query<{ value: string; count: number }, [string]>(
      `SELECT CASE WHEN has_error = 1 THEN 'error' ELSE 'ok' END AS value,
        count(*) AS count
       FROM trace_summaries
       WHERE project_id = ?
       GROUP BY value
       ORDER BY count DESC, value ASC`,
    )
    .all(LOCAL_TELEMETRY_AUTH.projectId)
    .map((row) => ({ count: row.count, label: row.value, value: row.value }));
}

function spanStatusFacet(sqlite: Database): FacetOption[] {
  return sqlite
    .query<{ value: string; count: number }, [string]>(
      `SELECT status_code AS value, count(*) AS count
       FROM spans
       WHERE project_id = ? AND status_code IS NOT NULL AND status_code != ''
       GROUP BY status_code
       ORDER BY count DESC, value ASC
       LIMIT 50`,
    )
    .all(LOCAL_TELEMETRY_AUTH.projectId)
    .map((row) => ({ count: row.count, label: row.value, value: row.value }));
}

function spanSourceFacet(sqlite: Database): FacetOption[] {
  return sqlite
    .query<{ value: string; count: number }, [string]>(
      `SELECT ts.source AS value, count(*) AS count
       FROM spans s
       JOIN trace_summaries ts
        ON ts.project_id = s.project_id AND ts.trace_id = s.trace_id
       WHERE s.project_id = ? AND ts.source IS NOT NULL AND ts.source != ''
       GROUP BY ts.source
       ORDER BY count DESC, value ASC
       LIMIT 50`,
    )
    .all(LOCAL_TELEMETRY_AUTH.projectId)
    .map((row) => ({ count: row.count, label: row.value, value: row.value }));
}

function numericFacet(sqlite: Database, table: string, column: string): NumericFacetSummary {
  const row = sqlite
    .query<{ min: number | null; max: number | null }, [string]>(
      `SELECT min(${column}) AS min, max(${column}) AS max
       FROM ${table}
       WHERE project_id = ? AND ${column} IS NOT NULL`,
    )
    .get(LOCAL_TELEMETRY_AUTH.projectId);
  return { max: row?.max ?? null, min: row?.min ?? null };
}

function attributeKeyFacet(sqlite: Database, column: string): FacetOption[] {
  return sqlite
    .query<{ value: string; count: number }, [string]>(
      `SELECT json_each.key AS value, count(*) AS count
       FROM spans, json_each(spans.${column})
       WHERE spans.project_id = ?
       GROUP BY json_each.key
       ORDER BY count DESC, value ASC
       LIMIT 50`,
    )
    .all(LOCAL_TELEMETRY_AUTH.projectId)
    .map((row) => ({ count: row.count, label: row.value, value: row.value }));
}

export function buildSpanTree(spans: Span[]): SpanNode[] {
  const nodes = new Map<string, SpanNode>();
  const roots: SpanNode[] = [];
  for (const span of spans) nodes.set(span.spanId, { children: [], span });
  for (const span of spans) {
    const node = nodes.get(span.spanId);
    if (!node) continue;
    const parent = span.parentSpanId ? nodes.get(span.parentSpanId) : null;
    if (parent) parent.children.push(node);
    else roots.push(node);
  }
  const sort = (items: SpanNode[]) => {
    items.sort((a, b) => a.span.startTimeMs - b.span.startTimeMs);
    items.forEach((item) => sort(item.children));
  };
  sort(roots);
  return roots;
}

function sessionAggregateSql(where: string): string {
  return `WITH session_rows AS (
    SELECT
      project_id,
      session_id,
      min(start_time) AS start_time,
      max(end_time) AS end_time,
      max(end_time) - min(start_time) AS duration_ms,
      CAST(round((max(end_time) - min(start_time)) * 1000000) AS TEXT) AS duration_ns,
      count(*) AS trace_count,
      sum(span_count) AS span_count,
      sum(llm_span_count) AS llm_span_count,
      sum(total_tokens) AS total_tokens,
      sum(cache_read_tokens) AS cache_read_tokens,
      sum(total_cost) AS total_cost,
      max(has_error) AS has_error,
      group_concat(DISTINCT NULLIF(service_name, '')) AS service_names,
      group_concat(DISTINCT NULLIF(agent_name, '')) AS agent_names,
      group_concat(DISTINCT NULLIF(source, '')) AS sources,
      group_concat(DISTINCT NULLIF(source_connection_name, '')) AS source_connection_names,
      (
        SELECT latest.trace_id
        FROM trace_summaries latest
        WHERE latest.project_id = trace_summaries.project_id
          AND latest.session_id = trace_summaries.session_id
        ORDER BY latest.start_time DESC, latest.trace_id DESC
        LIMIT 1
      ) AS latest_trace_id,
      (
        SELECT latest.root_span_name
        FROM trace_summaries latest
        WHERE latest.project_id = trace_summaries.project_id
          AND latest.session_id = trace_summaries.session_id
        ORDER BY latest.start_time DESC, latest.trace_id DESC
        LIMIT 1
      ) AS latest_trace_name,
      (
        SELECT group_concat(DISTINCT NULLIF(s.llm_model_name, ''))
        FROM spans s
        JOIN trace_summaries ts
          ON ts.project_id = s.project_id AND ts.trace_id = s.trace_id
        WHERE ts.project_id = trace_summaries.project_id
          AND ts.session_id = trace_summaries.session_id
      ) AS llm_model_names
    FROM trace_summaries
    ${where}
    GROUP BY project_id, session_id
  )`;
}

function buildSessionWhere(filters?: TelemetryFilters) {
  const { conditions, params } = buildTraceWhere(filters);
  conditions.push("session_id IS NOT NULL", "session_id != ''");
  return { conditions, params };
}

function sessionStatusFacet(sqlite: Database): FacetOption[] {
  return sqlite
    .query<{ value: string; count: number }, [string]>(
      `SELECT value, count(*) AS count
       FROM (
        SELECT CASE WHEN max(has_error) = 1 THEN 'error' ELSE 'ok' END AS value
        FROM trace_summaries
        WHERE project_id = ?
          AND session_id IS NOT NULL
          AND session_id != ''
        GROUP BY session_id
       )
       GROUP BY value
       ORDER BY count DESC, value ASC`,
    )
    .all(LOCAL_TELEMETRY_AUTH.projectId)
    .map((row) => ({ count: row.count, label: row.value, value: row.value }));
}

function sessionCategoricalFacet(
  sqlite: Database,
  column: string,
): FacetOption[] {
  return sqlite
    .query<{ value: string; count: number }, [string]>(
      `SELECT ${column} AS value, count(DISTINCT session_id) AS count
       FROM trace_summaries
       WHERE project_id = ?
         AND session_id IS NOT NULL
         AND session_id != ''
         AND ${column} IS NOT NULL
         AND ${column} != ''
       GROUP BY ${column}
       ORDER BY count DESC, value ASC
       LIMIT 50`,
    )
    .all(LOCAL_TELEMETRY_AUTH.projectId)
    .map((row) => ({ count: row.count, label: row.value, value: row.value }));
}

function sessionSpanBackedFacet(sqlite: Database, column: string): FacetOption[] {
  return sqlite
    .query<{ value: string; count: number }, [string]>(
      `SELECT s.${column} AS value, count(DISTINCT ts.session_id) AS count
       FROM spans s
       JOIN trace_summaries ts
        ON ts.project_id = s.project_id AND ts.trace_id = s.trace_id
       WHERE ts.project_id = ?
         AND ts.session_id IS NOT NULL
         AND ts.session_id != ''
         AND s.${column} IS NOT NULL
         AND s.${column} != ''
       GROUP BY s.${column}
       ORDER BY count DESC, value ASC
       LIMIT 50`,
    )
    .all(LOCAL_TELEMETRY_AUTH.projectId)
    .map((row) => ({ count: row.count, label: row.value, value: row.value }));
}

function sessionNumericFacet(
  sqlite: Database,
  column: string,
): NumericFacetSummary {
  const row = sqlite
    .query<{ min: number | null; max: number | null }, [string]>(
      `SELECT min(value) AS min, max(value) AS max
       FROM (
        SELECT sum(${column}) AS value
        FROM trace_summaries
        WHERE project_id = ?
          AND session_id IS NOT NULL
          AND session_id != ''
        GROUP BY session_id
       )
       WHERE value IS NOT NULL`,
    )
    .get(LOCAL_TELEMETRY_AUTH.projectId);
  return { max: row?.max ?? null, min: row?.min ?? null };
}

function buildTraceWhere(filters?: TelemetryFilters) {
  const params: QueryParams = { projectId: LOCAL_TELEMETRY_AUTH.projectId };
  const conditions = ["project_id = :projectId"];
  if (!filters) return { conditions, params };
  pushTimeFilters(filters, conditions, params);
  pushInFilter(conditions, params, "service_name", "service", filters.serviceNames);
  pushInFilter(
    conditions,
    params,
    "deployment_environment",
    "deployment",
    filters.deploymentEnvironments,
  );
  pushInFilter(conditions, params, "session_id", "session", filters.sessionIds);
  pushInFilter(conditions, params, "agent_name", "agent", filters.agents);
  pushInFilter(conditions, params, "source", "source", filters.sources);
  if (filters.traceId) {
    conditions.push("trace_id = :traceId");
    params.traceId = filters.traceId;
  }
  if (filters.status === "error") conditions.push("has_error = 1");
  if (filters.status === "ok") conditions.push("has_error = 0");
  if (filters.observationKinds?.length) {
    pushExistsFilter(conditions, params, "observation_kind", "kind", filters.observationKinds);
  }
  if (filters.llmProviders?.length) {
    pushExistsFilter(conditions, params, "llm_provider", "provider", filters.llmProviders);
  }
  if (filters.llmModelNames?.length) {
    pushExistsFilter(conditions, params, "llm_model_name", "model", filters.llmModelNames);
  }
  return { conditions, params };
}

function buildSpanWhere(filters?: TelemetryFilters) {
  const params: QueryParams = { projectId: LOCAL_TELEMETRY_AUTH.projectId };
  const conditions = ["project_id = :projectId"];
  if (!filters) return { conditions, params };
  pushTimeFilters(filters, conditions, params);
  pushInFilter(conditions, params, "observation_kind", "kind", filters.observationKinds);
  pushInFilter(conditions, params, "llm_provider", "provider", filters.llmProviders);
  pushInFilter(conditions, params, "llm_model_name", "model", filters.llmModelNames);
  pushInFilter(conditions, params, "service_name", "service", filters.serviceNames);
  pushInFilter(
    conditions,
    params,
    "deployment_environment",
    "deployment",
    filters.deploymentEnvironments,
  );
  pushSpanSourceFilter(conditions, params, filters.sources);
  pushInFilter(conditions, params, "user_id", "user", filters.userIds);
  pushInFilter(conditions, params, "session_id", "session", filters.sessionIds);
  pushInFilter(conditions, params, "agent_name", "agent", filters.agents);
  if (filters.status === "error") conditions.push("status_code = 'STATUS_CODE_ERROR'");
  if (filters.status === "ok") {
    conditions.push("status_code IN ('STATUS_CODE_OK', 'STATUS_CODE_UNSET')");
  }
  if (filters.traceId) {
    conditions.push("trace_id = :traceId");
    params.traceId = filters.traceId;
  }
  if (filters.scope === "root") conditions.push("parent_span_id = ''");
  if (filters.scope === "entrypoint") {
    conditions.push(
      "parent_span_id = '' AND (observation_kind IN ('AGENT', 'CHAIN') OR span_kind IN ('SPAN_KIND_SERVER', 'SPAN_KIND_CONSUMER'))",
    );
  }
  if (filters.freeText?.trim()) {
    conditions.push(
      "(span_name LIKE :freeText OR input_messages LIKE :freeText OR output_messages LIKE :freeText OR input LIKE :freeText OR output LIKE :freeText)",
    );
    params.freeText = `%${filters.freeText.trim()}%`;
  }
  return { conditions, params };
}

function pushTimeFilters(
  filters: TelemetryFilters,
  conditions: string[],
  params: QueryParams,
) {
  const startDate = timeFilterMs(filters.startDate);
  if (startDate != null) {
    conditions.push("start_time >= :startDate");
    params.startDate = startDate;
  }
  const endDate = timeFilterMs(filters.endDate);
  if (endDate != null) {
    conditions.push("start_time <= :endDate");
    params.endDate = endDate;
  }
}

function timeFilterMs(value: Date | string | number | undefined): number | null {
  if (value == null) return null;
  if (value instanceof Date) {
    const time = value.getTime();
    return Number.isFinite(time) ? time : null;
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Date.parse(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

function pushInFilter(
  conditions: string[],
  params: QueryParams,
  column: string,
  prefix: string,
  values?: readonly string[],
) {
  const filtered = values?.filter(Boolean) ?? [];
  if (filtered.length === 0) return;
  const placeholders = filtered.map((_, i) => `:${prefix}${i}`);
  filtered.forEach((value, i) => {
    params[`${prefix}${i}`] = value;
  });
  conditions.push(`${column} IN (${placeholders.join(", ")})`);
}

function pushSpanSourceFilter(
  conditions: string[],
  params: QueryParams,
  values?: readonly string[],
) {
  const filtered = values?.filter(Boolean) ?? [];
  if (filtered.length === 0) return;
  const placeholders = filtered.map((_, i) => `:spanSource${i}`);
  filtered.forEach((value, i) => {
    params[`spanSource${i}`] = value;
  });
  conditions.push(
    `EXISTS (
      SELECT 1 FROM trace_summaries ts
      WHERE ts.project_id = spans.project_id
        AND ts.trace_id = spans.trace_id
        AND ts.source IN (${placeholders.join(", ")})
    )`,
  );
}

function pushExistsFilter(
  conditions: string[],
  params: QueryParams,
  column: string,
  prefix: string,
  values: readonly string[],
) {
  const filtered = values.filter(Boolean);
  if (filtered.length === 0) return;
  const placeholders = filtered.map((_, i) => `:${prefix}${i}`);
  filtered.forEach((value, i) => {
    params[`${prefix}${i}`] = value;
  });
  conditions.push(
    `EXISTS (
      SELECT 1 FROM spans s
      WHERE s.project_id = trace_summaries.project_id
        AND s.trace_id = trace_summaries.trace_id
        AND s.${column} IN (${placeholders.join(", ")})
    )`,
  );
}

function facetColumn(facet: FacetId): string | null {
  const map: Partial<Record<FacetId, string>> = {
    agent_id: "agent_id",
    agent_name: "agent_name",
    deployment_environment: "deployment_environment",
    llm_model_name: "llm_model_name",
    llm_provider: "llm_provider",
    observation_kind: "observation_kind",
    service_name: "service_name",
    session_id: "session_id",
    source: "source",
    user_id: "user_id",
  };
  if (facet === "status") return "status_code";
  return map[facet] ?? null;
}

function numericFacetColumn(facet: FacetId, mode: "trace" | "span"): string | null {
  const traceMap: Partial<Record<FacetId, string>> = {
    cost_total: "total_cost",
    duration_ns: "duration_ms",
    llm_span_count: "llm_span_count",
    span_count: "span_count",
    total_tokens: "total_tokens",
  };
  const spanMap: Partial<Record<FacetId, string>> = {
    cache_read_tokens: "cache_read_tokens",
    cost_total: "cost_total",
    duration_ns: "duration_ms",
    input_tokens: "input_tokens",
    output_tokens: "output_tokens",
    total_tokens: "total_tokens",
  };
  return mode === "trace" ? (traceMap[facet] ?? null) : (spanMap[facet] ?? null);
}

function traceSortColumn(sortBy: TraceSortKey | undefined): string {
  const map: Record<TraceSortKey, string> = {
    duration: "duration_ms",
    llm_span_count: "llm_span_count",
    span_count: "span_count",
    start_time: "start_time",
    total_cost: "total_cost",
    total_tokens: "total_tokens",
  };
  return map[sortBy ?? "start_time"];
}

function spanSortColumn(sortBy: SpanSortKey | undefined): string {
  const map: Record<SpanSortKey, string> = {
    cost_total: "cost_total",
    duration_ns: "duration_ms",
    start_time: "start_time",
    total_tokens: "total_tokens",
  };
  return map[sortBy ?? "start_time"];
}

function sessionSortColumn(sortBy: SessionSortKey | undefined): string {
  const map: Record<SessionSortKey, string> = {
    duration: "duration_ms",
    last_activity: "end_time",
    llm_span_count: "llm_span_count",
    span_count: "span_count",
    start_time: "start_time",
    total_cost: "total_cost",
    total_tokens: "total_tokens",
    trace_count: "trace_count",
  };
  return map[sortBy ?? "last_activity"];
}

function mapSessionSummary(row: Record<string, unknown>): SessionSummary {
  const startTimeMs = Number(row.start_time ?? 0);
  const endTimeMs = Number(row.end_time ?? 0);
  return {
    agentNames: csvToStrings(row.agent_names),
    cacheReadTokens: nullableNumber(row.cache_read_tokens),
    durationMs: Number(row.duration_ms ?? 0),
    durationNs: String(row.duration_ns ?? "0"),
    endTime: isoFromMs(endTimeMs),
    endTimeMs,
    hasError: Number(row.has_error ?? 0) > 0,
    latestTraceId: String(row.latest_trace_id ?? ""),
    latestTraceName: String(row.latest_trace_name ?? ""),
    llmModelNames: csvToStrings(row.llm_model_names),
    llmSpanCount: Number(row.llm_span_count ?? 0),
    projectId: String(row.project_id ?? ""),
    serviceNames: csvToStrings(row.service_names),
    sessionId: String(row.session_id ?? ""),
    sourceConnectionNames: csvToStrings(row.source_connection_names),
    sources: csvToStrings(row.sources).map(asTraceSource),
    spanCount: Number(row.span_count ?? 0),
    startTime: isoFromMs(startTimeMs),
    startTimeMs,
    totalCost: nullableMoney(row.total_cost),
    totalTokens: nullableNumber(row.total_tokens),
    traceCount: Number(row.trace_count ?? 0),
  };
}

function mapTrace(row: Record<string, unknown>): Trace {
  const sourceImportedAtMs = nullableNumber(row.source_imported_at);
  return {
    agentId: String(row.agent_id ?? ""),
    agentName: String(row.agent_name ?? ""),
    cacheReadTokens: nullableNumber(row.cache_read_tokens),
    deploymentEnvironment: String(row.deployment_environment ?? ""),
    durationMs: Number(row.duration_ms ?? 0),
    durationNs: String(row.duration_ns ?? "0"),
    endTime: isoFromMs(Number(row.end_time ?? 0)),
    endTimeMs: Number(row.end_time ?? 0),
    hasError: Number(row.has_error ?? 0) > 0,
    llmSpanCount: Number(row.llm_span_count ?? 0),
    projectId: String(row.project_id ?? ""),
    rootObservationKind: asObservationKind(String(row.root_observation_kind ?? "SPAN")),
    rootSpanName: String(row.root_span_name ?? ""),
    serviceName: String(row.service_name ?? ""),
    serviceVersion: String(row.service_version ?? ""),
    sessionId: emptyToNull(String(row.session_id ?? "")),
    source: asTraceSource(String(row.source ?? "local")),
    sourceConnectionId: nullableString(row.source_connection_id),
    sourceConnectionName: nullableString(row.source_connection_name),
    sourceImportJobId: nullableString(row.source_import_job_id),
    sourceImportedAt: sourceImportedAtMs == null ? null : isoFromMs(sourceImportedAtMs),
    sourceImportedAtMs,
    sourceTags: parseStringArray(row.source_tags_json),
    sourceTraceId: nullableString(row.source_trace_id),
    sourceUrl: nullableString(row.source_url),
    spanCount: Number(row.span_count ?? 0),
    startTime: isoFromMs(Number(row.start_time ?? 0)),
    startTimeMs: Number(row.start_time ?? 0),
    totalCost: nullableMoney(row.total_cost),
    totalTokens: nullableNumber(row.total_tokens),
    traceId: String(row.trace_id ?? ""),
  };
}

function mapSpan(row: Record<string, unknown>): Span {
  return {
    agentId: String(row.agent_id ?? ""),
    agentName: String(row.agent_name ?? ""),
    apiKeyId: String(row.api_key_id ?? ""),
    cacheReadTokens: nullableNumber(row.cache_read_tokens),
    cacheWriteTokens: nullableNumber(row.cache_write_tokens),
    chatId: String(row.chat_id ?? ""),
    conversationId: String(row.conversation_id ?? ""),
    costCacheRead: nullableMoney(row.cost_cache_read),
    costCacheWrite: nullableMoney(row.cost_cache_write),
    costInput: nullableMoney(row.cost_input),
    costOutput: nullableMoney(row.cost_output),
    costReasoning: nullableMoney(row.cost_reasoning),
    costTotal: nullableMoney(row.cost_total),
    deploymentEnvironment: String(row.deployment_environment ?? ""),
    durationMs: Number(row.duration_ms ?? 0),
    durationNs: String(row.duration_ns ?? "0"),
    endTime: isoFromMs(Number(row.end_time ?? 0)),
    endTimeMs: Number(row.end_time ?? 0),
    events: parseJson(row.events_json, []),
    id: Number(row.id ?? 0),
    ingestedAt: isoFromMs(Number(row.ingested_at ?? 0)),
    input: nullableString(row.input),
    inputMessages: nullableString(row.input_messages),
    inputTokens: nullableNumber(row.input_tokens),
    links: parseJson(row.links_json, []),
    llmModelName: String(row.llm_model_name ?? ""),
    llmProvider: String(row.llm_provider ?? ""),
    llmResponseModel: String(row.llm_response_model ?? ""),
    observationKind: asObservationKind(String(row.observation_kind ?? "SPAN")),
    output: nullableString(row.output),
    outputMessages: nullableString(row.output_messages),
    outputTokens: nullableNumber(row.output_tokens),
    parentSpanId: String(row.parent_span_id ?? ""),
    projectId: String(row.project_id ?? ""),
    reasoningTokens: nullableNumber(row.reasoning_tokens),
    resourceAttributes: parseJson(row.resource_attributes, {}),
    resourceAttributesDouble: parseJson(row.resource_attributes_double, {}),
    resourceAttributesInt: parseJson(row.resource_attributes_int, {}),
    retrievalDocuments: nullableString(row.retrieval_documents),
    scopeName: String(row.scope_name ?? ""),
    scopeVersion: String(row.scope_version ?? ""),
    serviceName: String(row.service_name ?? ""),
    serviceVersion: String(row.service_version ?? ""),
    sessionId: nullableString(row.session_id),
    spanAttributes: parseJson(row.span_attributes, {}),
    spanAttributesDouble: parseJson(row.span_attributes_double, {}),
    spanAttributesInt: parseJson(row.span_attributes_int, {}),
    spanId: String(row.span_id ?? ""),
    spanKind: String(row.span_kind ?? ""),
    spanName: String(row.span_name ?? ""),
    startTime: isoFromMs(Number(row.start_time ?? 0)),
    startTimeMs: Number(row.start_time ?? 0),
    statusCode: String(row.status_code ?? ""),
    statusMessage: String(row.status_message ?? ""),
    teamId: String(row.team_id ?? ""),
    totalTokens: nullableNumber(row.total_tokens),
    traceId: String(row.trace_id ?? ""),
    traceState: String(row.trace_state ?? ""),
    userId: nullableString(row.user_id),
  };
}

function encodeCursor(input: { value: string | number | undefined; id: string }): string {
  return Buffer.from(JSON.stringify(input), "utf8").toString("base64url");
}

function decodeCursor(cursor: string | null | undefined): { value: string | number; id: string } | null {
  if (!cursor) return null;
  try {
    const parsed = JSON.parse(Buffer.from(cursor, "base64url").toString("utf8")) as {
      value?: string | number;
      id?: string;
    };
    if (parsed.value == null || !parsed.id) return null;
    return { id: parsed.id, value: parsed.value };
  } catch {
    return null;
  }
}

function encodeSessionSpanCursor(input: {
  value: string | number | undefined;
  traceId: string;
  spanId: string;
}): string {
  return Buffer.from(JSON.stringify(input), "utf8").toString("base64url");
}

function decodeSessionSpanCursor(
  cursor: string | null | undefined,
): { value: string | number; traceId: string; spanId: string } | null {
  if (!cursor) return null;
  try {
    const parsed = JSON.parse(Buffer.from(cursor, "base64url").toString("utf8")) as {
      value?: string | number;
      traceId?: string;
      spanId?: string;
    };
    if (parsed.value == null || !parsed.traceId || !parsed.spanId) return null;
    return { spanId: parsed.spanId, traceId: parsed.traceId, value: parsed.value };
  } catch {
    return null;
  }
}

function clampLimit(limit: number | undefined, max = 1000): number {
  if (!limit || limit < 1) return 50;
  return Math.min(max, Math.floor(limit));
}

function parseJson<T>(value: unknown, fallback: T): T {
  if (typeof value !== "string" || !value) return fallback;
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

function tableCount(sqlite: Database, tableName: string): number {
  const row = sqlite
    .query<{ count: number }, []>(`SELECT count(*) AS count FROM ${tableName}`)
    .get();
  return row?.count ?? 0;
}

function resetAutoincrement(sqlite: Database, tableNames: string[]) {
  try {
    const placeholders = tableNames.map(() => "?").join(", ");
    sqlite
      .query(`DELETE FROM sqlite_sequence WHERE name IN (${placeholders})`)
      .run(...tableNames);
  } catch {
    // sqlite_sequence only exists after AUTOINCREMENT tables have been written.
  }
}

function parseStringArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === "string");
  }
  if (typeof value !== "string" || !value.trim()) return [];
  const parsed = parseJson<unknown>(value, null);
  if (Array.isArray(parsed)) {
    return parsed.filter((item): item is string => typeof item === "string");
  }
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function csvToStrings(value: unknown): string[] {
  if (typeof value !== "string" || !value.trim()) return [];
  return [...new Set(value.split(",").map((item) => item.trim()).filter(Boolean))];
}

function intersectValues(
  left: readonly string[] | undefined,
  right: readonly string[],
): string[] {
  const rightSet = new Set(right);
  if (!left?.length) return [...rightSet];
  return left.filter((value) => rightSet.has(value));
}

function nullableNumber(value: unknown): number | null {
  if (value == null) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function nullableMoney(value: unknown): string | null {
  if (value == null) return null;
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return n.toFixed(10).replace(/\.?0+$/, "");
}

function nullableString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function emptyToNull(value: string): string | null {
  return value.length > 0 ? value : null;
}

function asObservationKind(value: string): ObservationKind {
  const kinds = new Set<ObservationKind>([
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
  ]);
  return kinds.has(value as ObservationKind) ? (value as ObservationKind) : "SPAN";
}

function asTraceSource(value: string): TraceSource {
  return value === "langfuse" ? "langfuse" : "local";
}

function isoFromMs(ms: number): string {
  return new Date(ms).toISOString();
}
