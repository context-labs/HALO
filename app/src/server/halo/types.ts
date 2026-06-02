import type { TelemetryFilters } from "../telemetry/types";

export const HALO_REPO_URL = "https://github.com/context-labs/HALO";
export const HALO_RUN_TARGET_TYPES = ["trace_group", "session_group"] as const;
export const HALO_PROVIDER_TYPES = [
  "openai",
  "anthropic_compat",
  "custom",
] as const;
export const HALO_RUN_STATUSES = [
  "queued",
  "exporting",
  "running",
  "completed",
  "incomplete",
  "failed",
  "cancelled",
  "interrupted",
] as const;

export type HaloProviderType = (typeof HALO_PROVIDER_TYPES)[number];
export type HaloRunTargetType = (typeof HALO_RUN_TARGET_TYPES)[number];
export type HaloRunStatus = (typeof HALO_RUN_STATUSES)[number];

export type HaloEngineStatus = {
  commitSha: string | null;
  defaultInstallPath: string;
  installPath: string;
  installedAt: string | null;
  lastError: string | null;
  repoUrl: string;
  status: "not_installed" | "installing" | "installed" | "error";
  checks: {
    git: string | null;
    python: string | null;
    uv: string | null;
    importable: boolean;
  };
  updatedAt: string | null;
};

export type HaloModelProvider = {
  id: string;
  apiKeyMasked: string;
  baseUrl: string;
  createdAt: string;
  headers: Record<string, string>;
  lastError: string | null;
  lastStatus: string;
  lastTestedAt: string | null;
  model: string;
  name: string;
  providerType: HaloProviderType;
  updatedAt: string;
};

export type StoredHaloModelProvider = HaloModelProvider & {
  apiKey: string;
};

export type HaloRun = {
  id: string;
  bunqueueJobId: string | null;
  createdAt: string;
  errorMessage: string | null;
  exportPath: string | null;
  filters: TelemetryFilters;
  finalAnswer: string | null;
  finalAnswerSource: string | null;
  finishedAt: string | null;
  maxDepth: number;
  maxParallel: number;
  maxTurns: number;
  model: string;
  progress: number;
  prompt: string;
  providerId: string | null;
  providerName: string;
  resultPath: string | null;
  sessionCount: number;
  spanCount: number;
  startedAt: string | null;
  status: HaloRunStatus;
  targetType: HaloRunTargetType;
  title: string;
  traceCount: number;
  updatedAt: string;
};

export type HaloRunEvent = {
  id: number;
  createdAt: string;
  eventType: string;
  payload: Record<string, unknown>;
  runId: string;
  sequence: number;
};

export type HaloRunPreview = {
  sessionCount: number;
  spanCount: number;
  targetType: HaloRunTargetType;
  traceCount: number;
  warnings: string[];
};

export type StartHaloRunInput = {
  filters: TelemetryFilters;
  maxDepth: number;
  maxParallel: number;
  maxTurns: number;
  model?: string;
  prompt: string;
  providerId: string;
  targetType: HaloRunTargetType;
  title?: string;
};

export type HaloRunSnapshot = {
  errorMessage: string | null;
  finalAnswer: string | null;
  finishedAt: string | null;
  id: string;
  model: string;
  progress: number;
  providerName: string;
  sessionCount: number;
  spanCount: number;
  startedAt: string | null;
  status: HaloRunStatus;
  targetType: HaloRunTargetType;
  title: string;
  traceCount: number;
  updatedAt: string;
};
