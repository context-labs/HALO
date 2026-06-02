export const LOCAL_TELEMETRY_AUTH = {
  apiKeyId: "local-dev",
  projectId: "local-project",
  teamId: "local-team",
} as const;

export const INGEST_HOSTNAME = "127.0.0.1";
export const INGEST_PORT = 8799;
export const INGEST_URL = `http://${INGEST_HOSTNAME}:${INGEST_PORT}`;
export const TRACE_INGEST_PATH = "/v1/traces";
export const TRACE_INGEST_URL = `${INGEST_URL}${TRACE_INGEST_PATH}`;
export const LIVE_WS_PORT = 8800;
export const LIVE_WS_URL = `ws://${INGEST_HOSTNAME}:${LIVE_WS_PORT}`;

export const OBSERVATION_KINDS = [
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
] as const;

export type ObservationKind = (typeof OBSERVATION_KINDS)[number];

export type ProducerShape =
  | "openinference"
  | "openllmetry"
  | "vercel-ai"
  | "langsmith"
  | "pydantic-ai"
  | "codex"
  | "braintrust"
  | "cursor"
  | "elevenlabs"
  | "unknown";

export const TRACE_SOURCES = ["local", "langfuse"] as const;
export type TraceSource = (typeof TRACE_SOURCES)[number];

export interface SpanEvent {
  timestamp: string;
  name: string;
  attributes: Record<string, string>;
}

export interface SpanLink {
  traceId: string;
  spanId: string;
  traceState: string;
  attributes: Record<string, string>;
}

export interface Span {
  id: number;
  projectId: string;
  teamId: string;
  apiKeyId: string;
  traceId: string;
  spanId: string;
  parentSpanId: string;
  traceState: string;
  startTime: string;
  endTime: string;
  startTimeMs: number;
  endTimeMs: number;
  durationNs: string;
  durationMs: number;
  ingestedAt: string;
  spanName: string;
  spanKind: string;
  serviceName: string;
  serviceVersion: string;
  deploymentEnvironment: string;
  scopeName: string;
  scopeVersion: string;
  statusCode: string;
  statusMessage: string;
  observationKind: ObservationKind;
  llmProvider: string;
  llmModelName: string;
  llmResponseModel: string;
  inputTokens: number | null;
  outputTokens: number | null;
  totalTokens: number | null;
  cacheReadTokens: number | null;
  cacheWriteTokens: number | null;
  reasoningTokens: number | null;
  costTotal: string | null;
  costInput: string | null;
  costOutput: string | null;
  costCacheRead: string | null;
  costCacheWrite: string | null;
  costReasoning: string | null;
  userId: string | null;
  sessionId: string | null;
  conversationId: string;
  chatId: string;
  agentName: string;
  agentId: string;
  inputMessages: string | null;
  outputMessages: string | null;
  input: string | null;
  output: string | null;
  retrievalDocuments: string | null;
  resourceAttributes: Record<string, string>;
  resourceAttributesInt: Record<string, number>;
  resourceAttributesDouble: Record<string, number>;
  spanAttributes: Record<string, string>;
  spanAttributesInt: Record<string, number>;
  spanAttributesDouble: Record<string, number>;
  events: SpanEvent[];
  links: SpanLink[];
}

export interface Trace {
  traceId: string;
  projectId: string;
  sessionId: string | null;
  startTime: string;
  endTime: string;
  startTimeMs: number;
  endTimeMs: number;
  durationNs: string;
  durationMs: number;
  rootSpanName: string;
  rootObservationKind: ObservationKind;
  spanCount: number;
  llmSpanCount: number;
  totalTokens: number | null;
  cacheReadTokens: number | null;
  totalCost: string | null;
  hasError: boolean;
  serviceName: string;
  serviceVersion: string;
  deploymentEnvironment: string;
  agentName: string;
  agentId: string;
  source: TraceSource;
  sourceTraceId: string | null;
  sourceConnectionId: string | null;
  sourceConnectionName: string | null;
  sourceImportJobId: string | null;
  sourceImportedAt: string | null;
  sourceImportedAtMs: number | null;
  sourceUrl: string | null;
  sourceTags: string[];
}

export interface SessionSummary {
  sessionId: string;
  projectId: string;
  startTime: string;
  endTime: string;
  startTimeMs: number;
  endTimeMs: number;
  durationNs: string;
  durationMs: number;
  traceCount: number;
  spanCount: number;
  llmSpanCount: number;
  totalTokens: number | null;
  cacheReadTokens: number | null;
  totalCost: string | null;
  hasError: boolean;
  latestTraceId: string;
  latestTraceName: string;
  serviceNames: string[];
  agentNames: string[];
  llmModelNames: string[];
  sources: TraceSource[];
  sourceConnectionNames: string[];
}

export interface SpanNode {
  span: Span;
  children: SpanNode[];
}

export interface TelemetryFilters {
  startDate?: Date | string | number;
  endDate?: Date | string | number;
  observationKinds?: ObservationKind[];
  llmProviders?: string[];
  llmModelNames?: string[];
  serviceNames?: string[];
  deploymentEnvironments?: string[];
  userIds?: string[];
  sessionIds?: string[];
  agents?: string[];
  sources?: TraceSource[];
  status?: "error" | "ok";
  freeText?: string;
  traceId?: string;
  scope?: "all" | "root" | "entrypoint";
}

export type TraceSortKey =
  | "start_time"
  | "duration"
  | "total_cost"
  | "total_tokens"
  | "span_count"
  | "llm_span_count";

export type SpanSortKey =
  | "start_time"
  | "duration_ns"
  | "cost_total"
  | "total_tokens";

export type SessionSortKey =
  | "last_activity"
  | "start_time"
  | "duration"
  | "total_cost"
  | "total_tokens"
  | "trace_count"
  | "span_count"
  | "llm_span_count";

export type FacetId =
  | "observation_kind"
  | "status"
  | "service_name"
  | "deployment_environment"
  | "agent_name"
  | "agent_id"
  | "llm_provider"
  | "llm_model_name"
  | "user_id"
  | "session_id"
  | "source"
  | "duration_ns"
  | "total_tokens"
  | "input_tokens"
  | "cache_read_tokens"
  | "output_tokens"
  | "cost_total"
  | "span_count"
  | "llm_span_count"
  | "span_attributes"
  | "resource_attributes";

export interface FacetOption {
  value: string;
  label: string;
  count: number;
}

export interface NumericFacetSummary {
  min: number | null;
  max: number | null;
}

export interface FilterFacets {
  categorical: Partial<Record<FacetId, FacetOption[]>>;
  numeric: Partial<Record<FacetId, NumericFacetSummary>>;
  attributeKeys: Partial<Record<"span" | "resource", FacetOption[]>>;
}

export interface SpanDbRow {
  project_id: string;
  team_id: string;
  api_key_id: string;
  trace_id: string;
  span_id: string;
  parent_span_id: string;
  trace_state: string;
  start_time: number;
  end_time: number;
  start_time_unix_nano: string;
  end_time_unix_nano: string;
  duration_ns: string;
  duration_ms: number;
  ingested_at: number;
  span_name: string;
  span_kind: string;
  service_name: string;
  service_version: string;
  deployment_environment: string;
  scope_name: string;
  scope_version: string;
  status_code: string;
  status_message: string;
  observation_kind: ObservationKind;
  llm_provider: string;
  llm_model_name: string;
  llm_response_model: string;
  input_tokens: number | null;
  output_tokens: number | null;
  total_tokens: number | null;
  cache_read_tokens: number | null;
  cache_write_tokens: number | null;
  reasoning_tokens: number | null;
  cost_total: number | null;
  cost_input: number | null;
  cost_output: number | null;
  cost_cache_read: number | null;
  cost_cache_write: number | null;
  cost_reasoning: number | null;
  user_id: string | null;
  session_id: string | null;
  conversation_id: string;
  chat_id: string;
  agent_name: string;
  agent_id: string;
  input_messages: string | null;
  output_messages: string | null;
  input: string | null;
  output: string | null;
  retrieval_documents: string | null;
  resource_attributes: string;
  resource_attributes_int: string;
  resource_attributes_double: string;
  span_attributes: string;
  span_attributes_int: string;
  span_attributes_double: string;
  events_json: string;
  links_json: string;
}
