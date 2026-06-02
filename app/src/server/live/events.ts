import { EventEmitter } from "node:events";
import type { Database } from "bun:sqlite";
import type { HaloRunEvent, HaloRunSnapshot } from "../halo/types";
import type { Span, Trace } from "../telemetry/types";

const REPLAY_LIMIT = 1_000;
const RETAINED_EVENT_LIMIT = 10_000;

export type LiveEventType =
  | "span.upserted"
  | "trace.upserted"
  | "ingest.accepted"
  | "telemetry.changed"
  | "import.job.updated"
  | "halo.run.updated"
  | "halo.run.event"
  | "halo.run.completed"
  | "halo.run.failed";

export type SpanUpsertedPayload = {
  type: "span.upserted";
  span: Span;
};

export type TraceUpsertedPayload = {
  type: "trace.upserted";
  trace: Trace;
};

export type IngestAcceptedPayload = {
  type: "ingest.accepted";
  acceptedSpanCount: number;
  batchId: string;
  receivedAt: string;
  sizeBytes: number;
  traceCount: number;
  traceIds: string[];
};

export type TelemetryChangedPayload = {
  type: "telemetry.changed";
  acceptedSpanCount: number;
  batchId: string;
  traceCount: number;
  traceIds: string[];
};

export type ImportJobSnapshot = {
  id: string;
  bunqueueJobId: string | null;
  connectionId: string;
  connectionName?: string | null;
  currentTraceId: string | null;
  currentTraceName: string | null;
  errorMessage: string | null;
  failedTraces: number;
  finishedAt: string | null;
  importedObservations: number;
  importedTraces: number;
  progress: number;
  startedAt: string | null;
  status: string;
  totalObservations: number;
  totalTraces: number;
  updatedAt: string;
};

export type ImportJobUpdatedPayload = {
  type: "import.job.updated";
  job: ImportJobSnapshot;
};

export type HaloRunUpdatedPayload = {
  type: "halo.run.updated";
  run: HaloRunSnapshot;
};

export type HaloRunEventPayload = {
  type: "halo.run.event" | "halo.run.completed" | "halo.run.failed";
  event: HaloRunEvent;
  run: HaloRunSnapshot;
};

export type LiveEventPayload =
  | SpanUpsertedPayload
  | TraceUpsertedPayload
  | IngestAcceptedPayload
  | TelemetryChangedPayload
  | ImportJobUpdatedPayload
  | HaloRunUpdatedPayload
  | HaloRunEventPayload;

export type LiveEvent = {
  createdAt: string;
  createdAtMs: number;
  eventType: LiveEventType;
  haloRunId: string | null;
  id: number;
  importJobId: string | null;
  payload: LiveEventPayload;
  traceId: string | null;
};

export type LiveEventFilter = {
  haloRunId?: string | null;
  importJobId?: string | null;
  traceId?: string | null;
};

export type LiveEventStore = ReturnType<typeof createLiveEventStore>;

export function createLiveEventStore(sqlite: Database) {
  const emitter = new EventEmitter();
  emitter.setMaxListeners(250);

  function publish(input: {
    eventType: LiveEventType;
    payload: LiveEventPayload;
    traceId?: string | null;
  }): LiveEvent {
    const createdAtMs = Date.now();
    const traceId = input.traceId ?? traceIdFromPayload(input.payload);
    sqlite
      .query(
        `INSERT INTO live_events (created_at, trace_id, event_type, payload_json)
         VALUES (?, ?, ?, ?)`,
      )
      .run(createdAtMs, traceId, input.eventType, JSON.stringify(input.payload));

    const id =
      sqlite.query<{ id: number }, []>("SELECT last_insert_rowid() AS id").get()
        ?.id ?? 0;
    const event: LiveEvent = {
      createdAt: new Date(createdAtMs).toISOString(),
      createdAtMs,
      eventType: input.eventType,
      haloRunId: haloRunIdFromPayload(input.payload),
      id,
      importJobId: importJobIdFromPayload(input.payload),
      payload: input.payload,
      traceId,
    };

    emitter.emit("event", event);
    pruneOldEvents(sqlite);
    return event;
  }

  function replay(afterId: number | null | undefined, filter?: LiveEventFilter) {
    if (afterId == null || !Number.isFinite(afterId)) {
      return [];
    }

    return sqlite
      .query<
        {
          id: number;
          created_at: number;
          trace_id: string | null;
          event_type: LiveEventType;
          payload_json: string;
        },
        [number, number]
      >(
        `SELECT id, created_at, trace_id, event_type, payload_json
         FROM live_events
         WHERE id > ?
         ORDER BY id ASC
         LIMIT ?`,
      )
      .all(afterId, REPLAY_LIMIT)
      .map(mapStoredEvent)
      .filter((event) => eventMatchesFilter(event, filter));
  }

  function subscribe(
    listener: (event: LiveEvent) => void,
    filter?: LiveEventFilter,
  ) {
    const filteredListener = (event: LiveEvent) => {
      if (eventMatchesFilter(event, filter)) {
        listener(event);
      }
    };
    emitter.on("event", filteredListener);
    return () => emitter.off("event", filteredListener);
  }

  return {
    publish,
    replay,
    subscribe,
  };
}

export function eventMatchesFilter(
  event: LiveEvent,
  filter?: LiveEventFilter,
): boolean {
  const importJobId = filter?.importJobId;
  if (importJobId) return event.importJobId === importJobId;
  const haloRunId = filter?.haloRunId;
  if (haloRunId) return event.haloRunId === haloRunId;
  const traceId = filter?.traceId;
  if (!traceId) return true;
  if (event.traceId === traceId) return true;
  return traceIdsFromPayload(event.payload).includes(traceId);
}

function mapStoredEvent(row: {
  id: number;
  created_at: number;
  trace_id: string | null;
  event_type: LiveEventType;
  payload_json: string;
}): LiveEvent {
  const payload = JSON.parse(row.payload_json) as LiveEventPayload;
  return {
    createdAt: new Date(row.created_at).toISOString(),
    createdAtMs: row.created_at,
    eventType: row.event_type,
    haloRunId: haloRunIdFromPayload(payload),
    id: row.id,
    importJobId: importJobIdFromPayload(payload),
    payload,
    traceId: row.trace_id,
  };
}

function pruneOldEvents(sqlite: Database) {
  sqlite
    .query(
      `DELETE FROM live_events
       WHERE id NOT IN (
         SELECT id FROM live_events
         ORDER BY id DESC
         LIMIT ?
       )`,
    )
    .run(RETAINED_EVENT_LIMIT);
}

function traceIdFromPayload(payload: LiveEventPayload): string | null {
  if (payload.type === "span.upserted") return payload.span.traceId;
  if (payload.type === "trace.upserted") return payload.trace.traceId;
  return null;
}

function importJobIdFromPayload(payload: LiveEventPayload): string | null {
  if (payload.type === "import.job.updated") return payload.job.id;
  return null;
}

function haloRunIdFromPayload(payload: LiveEventPayload): string | null {
  if (payload.type === "halo.run.updated") return payload.run.id;
  if (
    payload.type === "halo.run.event" ||
    payload.type === "halo.run.completed" ||
    payload.type === "halo.run.failed"
  ) {
    return payload.run.id;
  }
  return null;
}

function traceIdsFromPayload(payload: LiveEventPayload): string[] {
  if (payload.type === "span.upserted") return [payload.span.traceId];
  if (payload.type === "trace.upserted") return [payload.trace.traceId];
  if (payload.type === "import.job.updated") return [];
  if (
    payload.type === "halo.run.updated" ||
    payload.type === "halo.run.event" ||
    payload.type === "halo.run.completed" ||
    payload.type === "halo.run.failed"
  ) {
    return [];
  }
  if ("traceIds" in payload) return payload.traceIds;
  return [];
}
