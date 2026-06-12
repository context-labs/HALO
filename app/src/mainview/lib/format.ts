import type { FacetOption } from "../../server/telemetry/types";

export type DateRange = "1h" | "24h" | "7d" | "all";

export function startDateForRange(range: DateRange) {
  if (range === "all") return undefined;
  const ms =
    range === "1h"
      ? 60 * 60 * 1000
      : range === "24h"
        ? 24 * 60 * 60 * 1000
        : 7 * 24 * 60 * 60 * 1000;
  // Snap to a 5-minute boundary so query keys stay stable across mounts —
  // otherwise every navigation produces a fresh key, misses the cache, and
  // flashes loading states.
  const step = 5 * 60 * 1000;
  return new Date(Math.floor((Date.now() - ms) / step) * step);
}

export function toFacetOptions(
  options: FacetOption[] | undefined,
  allLabel: string,
) {
  return [
    { label: allLabel, value: "all" },
    ...(options ?? []).map((option) => ({
      count: option.count,
      label: option.label || option.value,
      value: option.value,
    })),
  ];
}

export function sourceLabel(value: string, fallback = value) {
  if (value === "local") return "Local";
  if (value === "langfuse") return "Langfuse";
  if (value === "phoenix") return "Phoenix";
  return fallback;
}

export function shortList(values: string[], fallback: string) {
  if (values.length === 0) return fallback;
  if (values.length === 1) return values[0];
  return `${values[0]} +${values.length - 1}`;
}

export function kindVariant(kind: string) {
  if (kind === "LLM") return "status-brand" as const;
  if (kind === "TOOL") return "status-warning" as const;
  if (kind === "AGENT" || kind === "CHAIN") return "status-running" as const;
  return "outline" as const;
}

export function compactNumber(value: number) {
  return new Intl.NumberFormat(undefined, { notation: "compact" }).format(value);
}

export function formatDuration(ms: number) {
  if (ms < 1) return "<1ms";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)}s`;
  if (ms < 3_600_000) {
    const minutes = Math.floor(ms / 60_000);
    const seconds = Math.round((ms % 60_000) / 1000);
    return seconds > 0 ? `${minutes}m ${seconds}s` : `${minutes}m`;
  }
  if (ms < 86_400_000) {
    const hours = Math.floor(ms / 3_600_000);
    const minutes = Math.round((ms % 3_600_000) / 60_000);
    return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
  }
  const days = Math.floor(ms / 86_400_000);
  const hours = Math.round((ms % 86_400_000) / 3_600_000);
  return hours > 0 ? `${days}d ${hours}h` : `${days}d`;
}

export function formatMoney(value: number) {
  if (!value) return "$0";
  if (value < 0.01) return `$${value.toFixed(5)}`;
  return `$${value.toLocaleString(undefined, {
    maximumFractionDigits: 2,
    minimumFractionDigits: 2,
  })}`;
}

export function formatTimestamp(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "unknown";
  return new Intl.DateTimeFormat(undefined, {
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    month: "short",
  }).format(date);
}

export function relativeTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "recently";
  const seconds = Math.max(1, Math.round((Date.now() - date.getTime()) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return `${days}d ago`;
}

export function prettyMaybeJson(value: string | null | undefined) {
  if (!value) return "";
  try {
    return JSON.stringify(JSON.parse(value), null, 2);
  } catch {
    return value;
  }
}

/** Shorten a long hex id (trace/span/session) for display; full value belongs in `title`. */
export function shortId(value: string, length = 12) {
  if (value.length <= length) return value;
  return `${value.slice(0, length)}…`;
}
