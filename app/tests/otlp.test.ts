import { describe, expect, test } from "bun:test";
import {
  buildSpanRowsFromOtlp,
  decodeOtlpJsonBody,
} from "../src/server/telemetry/otlp";
import { makeGenAiPayload, makeTracePayload } from "./support/otlp-fixtures";

describe("OTLP JSON normalization", () => {
  test("maps OpenInference/Catalyst spans into canonical rows", () => {
    const rows = buildSpanRowsFromOtlp(makeTracePayload(), 1710000001000);

    expect(rows).toHaveLength(2);
    expect(rows[0]?.observation_kind).toBe("AGENT");
    expect(rows[0]?.service_name).toBe("halo-agent");
    expect(rows[0]?.agent_name).toBe("Local agent");

    const llm = rows.find((row) => row.observation_kind === "LLM");
    expect(llm?.llm_provider).toBe("openai");
    expect(llm?.llm_model_name).toBe("gpt-5-mini");
    expect(llm?.total_tokens).toBe(30);
    expect(llm?.cost_total).toBe(0.00042);
  });

  test("maps GenAI/OpenLLMetry spans and errors", () => {
    const rows = buildSpanRowsFromOtlp(makeGenAiPayload(), 1710000003000);

    expect(rows).toHaveLength(1);
    expect(rows[0]?.observation_kind).toBe("LLM");
    expect(rows[0]?.llm_provider).toBe("anthropic");
    expect(rows[0]?.llm_model_name).toBe("claude-opus-4");
    expect(rows[0]?.status_code).toBe("STATUS_CODE_ERROR");
    expect(rows[0]?.total_tokens).toBe(10);
  });

  test("accepts empty payloads", () => {
    expect(decodeOtlpJsonBody("")).toEqual({ resourceSpans: [] });
    expect(buildSpanRowsFromOtlp({ resourceSpans: [] })).toEqual([]);
  });
});
