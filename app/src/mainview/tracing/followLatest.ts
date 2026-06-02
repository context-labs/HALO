import type { LiveEvent } from "../../server/live/events";

export function traceIdsForLiveEvent(event: LiveEvent): string[] {
  if (event.payload.type === "span.upserted") return [event.payload.span.traceId];
  if (event.payload.type === "trace.upserted") return [event.payload.trace.traceId];
  if (event.payload.type === "import.job.updated") return [];
  if ("traceIds" in event.payload) return event.payload.traceIds;
  return [];
}

export function latestTraceIdFromLiveEvent(event: LiveEvent): string | null {
  return traceIdsForLiveEvent(event).at(-1) ?? null;
}

export function nextFollowLatestTraceId({
  currentTraceId,
  event,
  followLatest,
}: {
  currentTraceId?: string | null;
  event: LiveEvent;
  followLatest: boolean;
}) {
  if (!followLatest) return null;
  const latestTraceId = latestTraceIdFromLiveEvent(event);
  if (!latestTraceId || latestTraceId === currentTraceId) return null;
  return latestTraceId;
}
