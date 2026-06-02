export type LangfuseImportStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "interrupted";

export type LangfuseTraceFilters = {
  fromTimestamp?: string;
  toTimestamp?: string;
  environment?: string;
  traceName?: string;
  tag?: string;
  userId?: string;
  sessionId?: string;
  version?: string;
  release?: string;
};

export type LangfuseFacetValue = {
  label: string;
  value: string;
  count: number;
};

export type LangfuseDiscovery = {
  baseUrl: string;
  project: {
    id: string;
    name: string;
    organization?: {
      id: string;
      name: string;
    } | null;
  } | null;
  traces: {
    sampleSize: number;
    totalItems: number;
  };
  facets: {
    environments: LangfuseFacetValue[];
    releases: LangfuseFacetValue[];
    sessions: LangfuseFacetValue[];
    tags: LangfuseFacetValue[];
    traceNames: LangfuseFacetValue[];
    users: LangfuseFacetValue[];
    versions: LangfuseFacetValue[];
  };
};

export type LangfuseConnection = {
  id: string;
  baseUrl: string;
  createdAt: string;
  discoveredFacets: LangfuseDiscovery["facets"];
  lastConnectedAt: string | null;
  lastError: string | null;
  lastStatus: string;
  name: string;
  organizationId: string | null;
  organizationName: string | null;
  projectId: string | null;
  projectName: string | null;
  publicKey: string;
  updatedAt: string;
};

export type LangfuseImportJob = {
  id: string;
  bunqueueJobId: string | null;
  connectionId: string;
  connectionName: string | null;
  currentTraceId: string | null;
  currentTraceName: string | null;
  errorMessage: string | null;
  failedTraces: number;
  filters: LangfuseTraceFilters;
  finishedAt: string | null;
  importedObservations: number;
  importedTraces: number;
  progress: number;
  startedAt: string | null;
  status: LangfuseImportStatus;
  totalObservations: number;
  totalTraces: number;
  createdAt: string;
  updatedAt: string;
};

export type StoredLangfuseConnection = LangfuseConnection & {
  secretKey: string;
};

export type LangfuseTraceListItem = {
  id: string;
  timestamp?: string;
  name?: string | null;
  input?: unknown;
  output?: unknown;
  sessionId?: string | null;
  release?: string | null;
  version?: string | null;
  userId?: string | null;
  metadata?: unknown;
  tags?: string[];
  public?: boolean;
  environment?: string | null;
  observations?: unknown[];
  totalCost?: number | null;
  latency?: number | null;
};

export type LangfuseObservation = {
  id: string;
  traceId?: string | null;
  type?: string | null;
  name?: string | null;
  startTime?: string;
  endTime?: string | null;
  completionStartTime?: string | null;
  model?: string | null;
  providedModelName?: string | null;
  modelParameters?: unknown;
  input?: unknown;
  output?: unknown;
  version?: string | null;
  metadata?: unknown;
  usage?: Record<string, unknown> | null;
  usageDetails?: Record<string, unknown> | null;
  costDetails?: Record<string, unknown> | null;
  level?: string | null;
  statusMessage?: string | null;
  parentObservationId?: string | null;
  promptId?: string | null;
  promptName?: string | null;
  promptVersion?: number | null;
  latency?: number | null;
  timeToFirstToken?: number | null;
  environment?: string | null;
  totalCost?: number | null;
};

export type LangfuseTraceWithDetails = LangfuseTraceListItem & {
  htmlPath?: string;
  observations: LangfuseObservation[];
};
