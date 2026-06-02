import { describe, expect, test } from "bun:test";

import { presentHaloAgentStep } from "../src/mainview/halo/haloAgentConsolePresenter";
import type { HaloRunEvent } from "../src/server/halo/types";

describe("HALO agent console presenter", () => {
  test("renders assistant tool calls as terminal commands with parsed arguments", () => {
    const rows = presentHaloAgentStep(
      makeEvent({
        item: {
          content: null,
          role: "assistant",
          tool_calls: [
            {
              id: "call_abc123456789",
              function: {
                arguments: JSON.stringify({ includeSpans: true, traceId: "trace-1" }),
                name: "view_trace",
              },
              type: "function",
            },
          ],
        },
      }),
    );

    expect(rows).toHaveLength(1);
    expect(rows[0]?.kind).toBe("call");
    expect(rows[0]?.title).toBe("view_trace");
    expect(rows[0]?.command).toBe(
      "halo tool view_trace --include-spans --trace-id trace-1",
    );
    expect(rows[0]?.summaries).toContainEqual({
      label: "TraceId",
      value: "trace-1",
    });
  });

  test("renders tool results with parsed JSON summaries", () => {
    const rows = presentHaloAgentStep(
      makeEvent({
        item: {
          content: JSON.stringify({
            result: {
              service_names: ["gator-agent"],
              total_spans: 60,
              total_traces: 6,
            },
          }),
          role: "tool",
          tool_call_id: "call_abc123456789",
        },
      }),
    );

    expect(rows).toHaveLength(1);
    expect(rows[0]?.kind).toBe("result");
    expect(rows[0]?.body?.kind).toBe("json");
    expect(rows[0]?.summaries).toEqual([
      { label: "Service Names", value: "1 item" },
      { label: "Total Spans", value: "60" },
      { label: "Total Traces", value: "6" },
    ]);
  });

  test("renders assistant text as a message row", () => {
    const rows = presentHaloAgentStep(
      makeEvent({
        item: {
          content: "I found the slowest turn.",
          role: "assistant",
        },
      }),
    );

    expect(rows).toHaveLength(1);
    expect(rows[0]?.kind).toBe("message");
    expect(rows[0]?.copyText).toBe("I found the slowest turn.");
  });

  test("preserves malformed JSON arguments as text fallback", () => {
    const rows = presentHaloAgentStep(
      makeEvent({
        item: {
          content: null,
          role: "assistant",
          tool_calls: [
            {
              id: "call_bad",
              function: {
                arguments: "{not json",
                name: "view_trace",
              },
              type: "function",
            },
          ],
        },
      }),
    );

    expect(rows).toHaveLength(1);
    expect(rows[0]?.kind).toBe("call");
    expect(rows[0]?.command).toBe("halo tool view_trace");
    expect(rows[0]?.body).toEqual({ kind: "text", text: "{not json" });
  });

  test("falls back to a raw event for unknown payloads", () => {
    const rows = presentHaloAgentStep(
      makeEvent({
        mystery: {
          nested: true,
        },
        type: "custom_agent_event",
      }),
    );

    expect(rows).toHaveLength(1);
    expect(rows[0]?.kind).toBe("raw");
    expect(rows[0]?.title).toBe("custom_agent_event");
    expect(rows[0]?.body?.kind).toBe("json");
  });
});

function makeEvent(payload: Record<string, unknown>): HaloRunEvent {
  return {
    createdAt: "2026-05-28T19:21:53.000Z",
    eventType: "agent_step",
    id: 42,
    payload: {
      agent_id: "root-1",
      agent_name: "root",
      depth: 0,
      sequence: 7,
      type: "agent_step",
      ...payload,
    },
    runId: "run-1",
    sequence: 7,
  };
}
