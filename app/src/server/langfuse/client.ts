import type {
  LangfuseDiscovery,
  LangfuseFacetValue,
  LangfuseTraceFilters,
  LangfuseTraceListItem,
  LangfuseTraceWithDetails,
} from "./types";

type ProjectResponse = {
  data?: Array<{
    id: string;
    name: string;
    organization?: { id: string; name: string } | null;
    metadata?: unknown;
  }>;
};

type TraceListResponse = {
  data?: LangfuseTraceListItem[];
  meta?: {
    limit?: number;
    page?: number;
    totalItems?: number;
    totalPages?: number;
  };
};

export class LangfuseApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "LangfuseApiError";
  }
}

export class LangfuseApiClient {
  readonly baseUrl: string;
  private readonly authHeader: string;

  constructor(input: { baseUrl: string; publicKey: string; secretKey: string }) {
    this.baseUrl = normalizeLangfuseBaseUrl(input.baseUrl);
    this.authHeader = basicAuthHeader(input.publicKey, input.secretKey);
  }

  async health(signal?: AbortSignal) {
    return this.fetchJson<{ status?: string; version?: string }>("/api/public/health", {
      signal,
      unauthenticated: true,
    });
  }

  async projects(signal?: AbortSignal) {
    return this.fetchJson<ProjectResponse>("/api/public/projects", { signal });
  }

  async listTraces(
    input: {
      filters?: LangfuseTraceFilters;
      limit?: number;
      page?: number;
      orderBy?: string;
    },
    signal?: AbortSignal,
  ) {
    const url = buildTraceListPath(input);
    return this.fetchJson<TraceListResponse>(url, { signal });
  }

  async getTrace(traceId: string, signal?: AbortSignal) {
    return this.fetchJson<LangfuseTraceWithDetails>(
      `/api/public/traces/${encodeURIComponent(traceId)}`,
      { signal },
    );
  }

  private async fetchJson<T>(
    path: string,
    options: { signal?: AbortSignal; unauthenticated?: boolean } = {},
  ): Promise<T> {
    const url = new URL(path, this.baseUrl);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20_000);
    const signal = mergeSignals(controller.signal, options.signal);

    try {
      const response = await fetch(url, {
        headers: options.unauthenticated
          ? undefined
          : { authorization: this.authHeader },
        signal,
      });
      if (!response.ok) {
        const message = await response.text().catch(() => "");
        throw new LangfuseApiError(
          readableLangfuseError(response.status, message),
          response.status,
        );
      }
      return (await response.json()) as T;
    } catch (error) {
      if (error instanceof LangfuseApiError) throw error;
      if (error instanceof Error && error.name === "AbortError") {
        throw new LangfuseApiError("Timed out while connecting to Langfuse");
      }
      throw new LangfuseApiError(
        error instanceof Error ? error.message : "Could not connect to Langfuse",
      );
    } finally {
      clearTimeout(timeout);
    }
  }
}

export async function discoverLangfuse(input: {
  baseUrl: string;
  publicKey: string;
  secretKey: string;
}): Promise<LangfuseDiscovery> {
  const client = new LangfuseApiClient(input);
  await client.health();
  const projects = await client.projects();
  const traceList = await client.listTraces({
    limit: 100,
    orderBy: "timestamp.desc",
    page: 1,
  });
  const project = projects.data?.[0] ?? null;
  const traces = traceList.data ?? [];

  return {
    baseUrl: client.baseUrl,
    facets: deriveFacets(traces),
    project: project
      ? {
          id: project.id,
          name: project.name,
          organization: project.organization ?? null,
        }
      : null,
    traces: {
      sampleSize: traces.length,
      totalItems: traceList.meta?.totalItems ?? traces.length,
    },
  };
}

export function normalizeLangfuseBaseUrl(input: string): string {
  const trimmed = input.trim();
  if (!trimmed) throw new LangfuseApiError("Langfuse API URL is required");
  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    throw new LangfuseApiError("Langfuse API URL must be a valid URL");
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new LangfuseApiError("Langfuse API URL must start with http or https");
  }
  url.pathname = url.pathname.replace(/\/+$/, "");
  url.search = "";
  url.hash = "";
  return url.toString().replace(/\/$/, "");
}

export function basicAuthHeader(publicKey: string, secretKey: string): string {
  const pk = publicKey.trim();
  const sk = secretKey.trim();
  if (!pk || !sk) {
    throw new LangfuseApiError("Langfuse public key and secret key are required");
  }
  return `Basic ${Buffer.from(`${pk}:${sk}`).toString("base64")}`;
}

export function buildTraceListPath(input: {
  filters?: LangfuseTraceFilters;
  limit?: number;
  page?: number;
  orderBy?: string;
}) {
  const params = new URLSearchParams();
  params.set("page", String(input.page ?? 1));
  params.set("limit", String(input.limit ?? 50));
  params.set("orderBy", input.orderBy ?? "timestamp.asc");

  const filters = compactFilters(input.filters);
  if (filters.fromTimestamp) params.set("fromTimestamp", filters.fromTimestamp);
  if (filters.toTimestamp) params.set("toTimestamp", filters.toTimestamp);
  if (filters.environment) params.append("environment", filters.environment);
  if (filters.traceName) params.set("name", filters.traceName);
  if (filters.tag) params.append("tags", filters.tag);
  if (filters.userId) params.set("userId", filters.userId);
  if (filters.sessionId) params.set("sessionId", filters.sessionId);
  if (filters.version) params.set("version", filters.version);
  if (filters.release) params.set("release", filters.release);

  return `/api/public/traces?${params.toString()}`;
}

export function compactFilters(
  filters: LangfuseTraceFilters | undefined,
): LangfuseTraceFilters {
  if (!filters) return {};
  return Object.fromEntries(
    Object.entries(filters).filter(([, value]) => value != null && value !== ""),
  ) as LangfuseTraceFilters;
}

function deriveFacets(traces: LangfuseTraceListItem[]): LangfuseDiscovery["facets"] {
  return {
    environments: facet(traces.map((trace) => trace.environment)),
    releases: facet(traces.map((trace) => trace.release)),
    sessions: facet(traces.map((trace) => trace.sessionId)),
    tags: facet(traces.flatMap((trace) => trace.tags ?? [])),
    traceNames: facet(traces.map((trace) => trace.name)),
    users: facet(traces.map((trace) => trace.userId)),
    versions: facet(traces.map((trace) => trace.version)),
  };
}

function facet(values: Array<string | null | undefined>): LangfuseFacetValue[] {
  const counts = new Map<string, number>();
  for (const raw of values) {
    const value = raw?.trim();
    if (!value) continue;
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, 50)
    .map(([value, count]) => ({ count, label: value, value }));
}

function readableLangfuseError(status: number, body: string) {
  if (status === 401 || status === 403) {
    return "Langfuse rejected the supplied public key or secret key.";
  }
  if (status === 404) {
    return "Langfuse endpoint was not found. Check the API URL.";
  }
  const trimmed = body.trim();
  return trimmed
    ? `Langfuse returned HTTP ${status}: ${trimmed.slice(0, 300)}`
    : `Langfuse returned HTTP ${status}`;
}

function mergeSignals(
  first: AbortSignal,
  second: AbortSignal | undefined,
): AbortSignal {
  if (!second) return first;
  if (first.aborted || second.aborted) {
    const controller = new AbortController();
    controller.abort();
    return controller.signal;
  }
  const controller = new AbortController();
  const abort = () => controller.abort();
  first.addEventListener("abort", abort, { once: true });
  second.addEventListener("abort", abort, { once: true });
  return controller.signal;
}
