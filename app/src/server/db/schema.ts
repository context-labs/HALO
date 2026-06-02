import {
  index,
  integer,
  real,
  sqliteTable,
  text,
  uniqueIndex,
} from "drizzle-orm/sqlite-core";

export const spans = sqliteTable(
  "spans",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    projectId: text("project_id").notNull(),
    teamId: text("team_id").notNull(),
    apiKeyId: text("api_key_id").notNull(),
    traceId: text("trace_id").notNull(),
    spanId: text("span_id").notNull(),
    parentSpanId: text("parent_span_id").notNull().default(""),
    traceState: text("trace_state").notNull().default(""),
    startTime: integer("start_time").notNull(),
    endTime: integer("end_time").notNull(),
    startTimeUnixNano: text("start_time_unix_nano").notNull(),
    endTimeUnixNano: text("end_time_unix_nano").notNull(),
    durationNs: text("duration_ns").notNull(),
    durationMs: real("duration_ms").notNull(),
    ingestedAt: integer("ingested_at").notNull(),
    spanName: text("span_name").notNull().default(""),
    spanKind: text("span_kind").notNull().default("SPAN_KIND_UNSPECIFIED"),
    serviceName: text("service_name").notNull().default(""),
    serviceVersion: text("service_version").notNull().default(""),
    deploymentEnvironment: text("deployment_environment").notNull().default(""),
    scopeName: text("scope_name").notNull().default(""),
    scopeVersion: text("scope_version").notNull().default(""),
    statusCode: text("status_code").notNull().default("STATUS_CODE_UNSET"),
    statusMessage: text("status_message").notNull().default(""),
    observationKind: text("observation_kind").notNull().default("SPAN"),
    llmProvider: text("llm_provider").notNull().default(""),
    llmModelName: text("llm_model_name").notNull().default(""),
    llmResponseModel: text("llm_response_model").notNull().default(""),
    inputTokens: integer("input_tokens"),
    outputTokens: integer("output_tokens"),
    totalTokens: integer("total_tokens"),
    cacheReadTokens: integer("cache_read_tokens"),
    cacheWriteTokens: integer("cache_write_tokens"),
    reasoningTokens: integer("reasoning_tokens"),
    costTotal: real("cost_total"),
    costInput: real("cost_input"),
    costOutput: real("cost_output"),
    costCacheRead: real("cost_cache_read"),
    costCacheWrite: real("cost_cache_write"),
    costReasoning: real("cost_reasoning"),
    userId: text("user_id"),
    sessionId: text("session_id"),
    conversationId: text("conversation_id").notNull().default(""),
    chatId: text("chat_id").notNull().default(""),
    agentName: text("agent_name").notNull().default(""),
    agentId: text("agent_id").notNull().default(""),
    inputMessages: text("input_messages"),
    outputMessages: text("output_messages"),
    input: text("input"),
    output: text("output"),
    retrievalDocuments: text("retrieval_documents"),
    resourceAttributes: text("resource_attributes").notNull(),
    resourceAttributesInt: text("resource_attributes_int").notNull(),
    resourceAttributesDouble: text("resource_attributes_double").notNull(),
    spanAttributes: text("span_attributes").notNull(),
    spanAttributesInt: text("span_attributes_int").notNull(),
    spanAttributesDouble: text("span_attributes_double").notNull(),
    eventsJson: text("events_json").notNull(),
    linksJson: text("links_json").notNull(),
  },
  (table) => [
    uniqueIndex("spans_project_trace_span_uidx").on(
      table.projectId,
      table.traceId,
      table.spanId,
    ),
    index("spans_project_start_idx").on(table.projectId, table.startTime),
    index("spans_project_trace_idx").on(table.projectId, table.traceId),
    index("spans_project_kind_idx").on(table.projectId, table.observationKind),
    index("spans_project_status_idx").on(table.projectId, table.statusCode),
    index("spans_project_model_idx").on(table.projectId, table.llmModelName),
    index("spans_project_agent_idx").on(table.projectId, table.agentName),
    index("spans_project_session_idx").on(table.projectId, table.sessionId),
  ],
);

export const traceSummaries = sqliteTable(
  "trace_summaries",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    projectId: text("project_id").notNull(),
    traceId: text("trace_id").notNull(),
    sessionId: text("session_id"),
    startTime: integer("start_time").notNull(),
    endTime: integer("end_time").notNull(),
    durationNs: text("duration_ns").notNull(),
    durationMs: real("duration_ms").notNull(),
    rootSpanName: text("root_span_name").notNull().default(""),
    rootObservationKind: text("root_observation_kind").notNull().default("SPAN"),
    spanCount: integer("span_count").notNull(),
    llmSpanCount: integer("llm_span_count").notNull(),
    totalTokens: integer("total_tokens"),
    cacheReadTokens: integer("cache_read_tokens"),
    totalCost: real("total_cost"),
    hasError: integer("has_error", { mode: "boolean" }).notNull().default(false),
    serviceName: text("service_name").notNull().default(""),
    serviceVersion: text("service_version").notNull().default(""),
    deploymentEnvironment: text("deployment_environment").notNull().default(""),
    agentName: text("agent_name").notNull().default(""),
    agentId: text("agent_id").notNull().default(""),
    source: text("source").notNull().default("local"),
    sourceTraceId: text("source_trace_id"),
    sourceConnectionId: text("source_connection_id"),
    sourceConnectionName: text("source_connection_name"),
    sourceImportJobId: text("source_import_job_id"),
    sourceImportedAt: integer("source_imported_at"),
    sourceUrl: text("source_url"),
    sourceTagsJson: text("source_tags_json").notNull().default("[]"),
    updatedAt: integer("updated_at").notNull(),
  },
  (table) => [
    uniqueIndex("trace_summaries_project_trace_uidx").on(
      table.projectId,
      table.traceId,
    ),
    index("trace_summaries_project_start_idx").on(table.projectId, table.startTime),
    index("trace_summaries_project_status_idx").on(table.projectId, table.hasError),
    index("trace_summaries_project_agent_idx").on(table.projectId, table.agentName),
    index("trace_summaries_project_session_idx").on(table.projectId, table.sessionId),
    index("trace_summaries_project_source_idx").on(table.projectId, table.source),
  ],
);

export const ingestBatches = sqliteTable(
  "ingest_batches",
  {
    id: text("id").primaryKey(),
    receivedAt: integer("received_at").notNull(),
    sizeBytes: integer("size_bytes").notNull(),
    acceptedSpanCount: integer("accepted_span_count").notNull(),
    traceCount: integer("trace_count").notNull(),
    contentEncoding: text("content_encoding").notNull(),
    status: text("status").notNull(),
    errorMessage: text("error_message"),
  },
  (table) => [index("ingest_batches_received_at_idx").on(table.receivedAt)],
);

export const liveEvents = sqliteTable(
  "live_events",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    createdAt: integer("created_at").notNull(),
    traceId: text("trace_id"),
    eventType: text("event_type").notNull(),
    payloadJson: text("payload_json").notNull(),
  },
  (table) => [
    index("live_events_created_at_idx").on(table.createdAt),
    index("live_events_trace_id_idx").on(table.traceId, table.id),
    index("live_events_event_type_idx").on(table.eventType, table.id),
  ],
);

export const langfuseConnections = sqliteTable(
  "langfuse_connections",
  {
    id: text("id").primaryKey(),
    name: text("name").notNull(),
    baseUrl: text("base_url").notNull(),
    publicKey: text("public_key").notNull(),
    secretKey: text("secret_key").notNull(),
    projectId: text("project_id"),
    projectName: text("project_name"),
    organizationId: text("organization_id"),
    organizationName: text("organization_name"),
    discoveredFacetsJson: text("discovered_facets_json").notNull().default("{}"),
    lastStatus: text("last_status").notNull().default("unknown"),
    lastError: text("last_error"),
    lastConnectedAt: integer("last_connected_at"),
    createdAt: integer("created_at").notNull(),
    updatedAt: integer("updated_at").notNull(),
  },
  (table) => [
    index("langfuse_connections_updated_at_idx").on(table.updatedAt),
    uniqueIndex("langfuse_connections_base_public_uidx").on(
      table.baseUrl,
      table.publicKey,
    ),
  ],
);

export const langfuseImportJobs = sqliteTable(
  "langfuse_import_jobs",
  {
    id: text("id").primaryKey(),
    connectionId: text("connection_id").notNull(),
    bunqueueJobId: text("bunqueue_job_id"),
    status: text("status").notNull(),
    filtersJson: text("filters_json").notNull().default("{}"),
    progress: integer("progress").notNull().default(0),
    totalTraces: integer("total_traces").notNull().default(0),
    importedTraces: integer("imported_traces").notNull().default(0),
    totalObservations: integer("total_observations").notNull().default(0),
    importedObservations: integer("imported_observations").notNull().default(0),
    failedTraces: integer("failed_traces").notNull().default(0),
    errorMessage: text("error_message"),
    currentTraceId: text("current_trace_id"),
    currentTraceName: text("current_trace_name"),
    createdAt: integer("created_at").notNull(),
    updatedAt: integer("updated_at").notNull(),
    startedAt: integer("started_at"),
    finishedAt: integer("finished_at"),
  },
  (table) => [
    index("langfuse_import_jobs_connection_idx").on(table.connectionId),
    index("langfuse_import_jobs_status_idx").on(table.status, table.updatedAt),
    index("langfuse_import_jobs_updated_at_idx").on(table.updatedAt),
  ],
);

export const haloEngineSettings = sqliteTable("halo_engine_settings", {
  id: text("id").primaryKey().default("default"),
  repoUrl: text("repo_url").notNull().default("https://github.com/context-labs/HALO"),
  installPath: text("install_path").notNull(),
  status: text("status").notNull().default("not_installed"),
  commitSha: text("commit_sha"),
  lastError: text("last_error"),
  installedAt: integer("installed_at"),
  updatedAt: integer("updated_at").notNull(),
});

export const haloModelProviders = sqliteTable(
  "halo_model_providers",
  {
    id: text("id").primaryKey(),
    name: text("name").notNull(),
    providerType: text("provider_type").notNull(),
    baseUrl: text("base_url").notNull(),
    apiKey: text("api_key").notNull(),
    model: text("model").notNull(),
    headersJson: text("headers_json").notNull().default("{}"),
    lastStatus: text("last_status").notNull().default("unknown"),
    lastError: text("last_error"),
    lastTestedAt: integer("last_tested_at"),
    createdAt: integer("created_at").notNull(),
    updatedAt: integer("updated_at").notNull(),
  },
  (table) => [index("halo_model_providers_updated_at_idx").on(table.updatedAt)],
);

export const haloRuns = sqliteTable(
  "halo_runs",
  {
    id: text("id").primaryKey(),
    bunqueueJobId: text("bunqueue_job_id"),
    title: text("title").notNull(),
    status: text("status").notNull(),
    targetType: text("target_type").notNull(),
    filtersJson: text("filters_json").notNull().default("{}"),
    providerId: text("provider_id"),
    providerName: text("provider_name").notNull().default(""),
    model: text("model").notNull().default(""),
    prompt: text("prompt").notNull(),
    maxDepth: integer("max_depth").notNull().default(1),
    maxTurns: integer("max_turns").notNull().default(8),
    maxParallel: integer("max_parallel").notNull().default(2),
    traceCount: integer("trace_count").notNull().default(0),
    sessionCount: integer("session_count").notNull().default(0),
    spanCount: integer("span_count").notNull().default(0),
    progress: integer("progress").notNull().default(0),
    exportPath: text("export_path"),
    resultPath: text("result_path"),
    finalAnswer: text("final_answer"),
    finalAnswerSource: text("final_answer_source"),
    errorMessage: text("error_message"),
    createdAt: integer("created_at").notNull(),
    updatedAt: integer("updated_at").notNull(),
    startedAt: integer("started_at"),
    finishedAt: integer("finished_at"),
  },
  (table) => [
    index("halo_runs_status_idx").on(table.status, table.updatedAt),
    index("halo_runs_updated_at_idx").on(table.updatedAt),
  ],
);

export const haloRunEvents = sqliteTable(
  "halo_run_events",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    runId: text("run_id").notNull(),
    sequence: integer("sequence").notNull(),
    eventType: text("event_type").notNull(),
    payloadJson: text("payload_json").notNull(),
    createdAt: integer("created_at").notNull(),
  },
  (table) => [
    index("halo_run_events_run_idx").on(table.runId, table.sequence),
    index("halo_run_events_created_at_idx").on(table.createdAt),
  ],
);

export const haloRunArtifacts = sqliteTable(
  "halo_run_artifacts",
  {
    id: text("id").primaryKey(),
    runId: text("run_id").notNull(),
    artifactType: text("artifact_type").notNull(),
    path: text("path").notNull(),
    sizeBytes: integer("size_bytes").notNull().default(0),
    createdAt: integer("created_at").notNull(),
  },
  (table) => [index("halo_run_artifacts_run_idx").on(table.runId)],
);

export type SpanRowRecord = typeof spans.$inferSelect;
export type NewSpanRowRecord = typeof spans.$inferInsert;
export type TraceSummaryRecord = typeof traceSummaries.$inferSelect;
export type LiveEventRecord = typeof liveEvents.$inferSelect;
export type LangfuseConnectionRecord = typeof langfuseConnections.$inferSelect;
export type NewLangfuseConnectionRecord = typeof langfuseConnections.$inferInsert;
export type LangfuseImportJobRecord = typeof langfuseImportJobs.$inferSelect;
export type NewLangfuseImportJobRecord = typeof langfuseImportJobs.$inferInsert;
export type HaloEngineSettingsRecord = typeof haloEngineSettings.$inferSelect;
export type HaloModelProviderRecord = typeof haloModelProviders.$inferSelect;
export type NewHaloModelProviderRecord = typeof haloModelProviders.$inferInsert;
export type HaloRunRecord = typeof haloRuns.$inferSelect;
export type HaloRunEventRecord = typeof haloRunEvents.$inferSelect;
