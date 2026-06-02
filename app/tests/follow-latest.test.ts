import { describe, expect, test } from "bun:test";
import type { LiveEvent, LiveEventPayload } from "../src/server/live/events";
import {
  latestTraceIdFromLiveEvent,
  nextFollowLatestTraceId,
  traceIdsForLiveEvent,
} from "../src/mainview/tracing/followLatest";

describe("follow latest trace selection", () => {
  test("ignores events without trace IDs", () => {
    const event = liveEvent({
      acceptedSpanCount: 0,
      batchId: "batch-1",
      receivedAt: new Date(0).toISOString(),
      sizeBytes: 0,
      traceCount: 0,
      traceIds: [],
      type: "ingest.accepted",
    });

    expect(traceIdsForLiveEvent(event)).toEqual([]);
    expect(latestTraceIdFromLiveEvent(event)).toBeNull();
    expect(
      nextFollowLatestTraceId({
        currentTraceId: "trace-a",
        event,
        followLatest: true,
      }),
    ).toBeNull();
  });

  test("keeps the current trace when the latest event matches it", () => {
    const event = traceEvent("trace-a");

    expect(
      nextFollowLatestTraceId({
        currentTraceId: "trace-a",
        event,
        followLatest: true,
      }),
    ).toBeNull();
  });

  test("returns a new trace when follow mode is on", () => {
    const event = spanEvent("trace-b");

    expect(
      nextFollowLatestTraceId({
        currentTraceId: "trace-a",
        event,
        followLatest: true,
      }),
    ).toBe("trace-b");
  });

  test("does nothing when follow mode is off", () => {
    const event = spanEvent("trace-b");

    expect(
      nextFollowLatestTraceId({
        currentTraceId: "trace-a",
        event,
        followLatest: false,
      }),
    ).toBeNull();
  });

  test("chooses the latest trace ID from a multi-trace event", () => {
    const event = liveEvent({
      acceptedSpanCount: 4,
      batchId: "batch-2",
      traceCount: 3,
      traceIds: ["trace-a", "trace-b", "trace-c"],
      type: "telemetry.changed",
    });

    expect(traceIdsForLiveEvent(event)).toEqual(["trace-a", "trace-b", "trace-c"]);
    expect(latestTraceIdFromLiveEvent(event)).toBe("trace-c");
    expect(
      nextFollowLatestTraceId({
        currentTraceId: "trace-a",
        event,
        followLatest: true,
      }),
    ).toBe("trace-c");
  });
});

function spanEvent(traceId: string) {
  return liveEvent({
    span: { traceId },
    type: "span.upserted",
  } as LiveEventPayload);
}

function traceEvent(traceId: string) {
  return liveEvent({
    trace: { traceId },
    type: "trace.upserted",
  } as LiveEventPayload);
}

function liveEvent(payload: LiveEventPayload): LiveEvent {
  return {
    createdAt: new Date(0).toISOString(),
    createdAtMs: 0,
    eventType: payload.type,
    haloRunId: null,
    id: 1,
    importJobId: null,
    payload,
    traceId: null,
  };
}
