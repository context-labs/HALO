import type { OtlpExportTraceServiceRequest } from "../../src/server/telemetry/otlp";

export const TRACE_ID = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
export const ROOT_SPAN_ID = "bbbbbbbbbbbbbbbb";
export const LLM_SPAN_ID = "cccccccccccccccc";

export function makeTracePayload(): OtlpExportTraceServiceRequest {
  return {
    resourceSpans: [
      {
        resource: {
          attributes: [
            str("service.name", "halo-agent"),
            str("service.version", "0.1.0"),
            str("deployment.environment", "local"),
          ],
        },
        scopeSpans: [
          {
            scope: {
              name: "@inference/tracing.openai",
              version: "1.0.0",
            },
            spans: [
              {
                attributes: [
                  str("openinference.span.kind", "AGENT"),
                  str("agent.name", "Local agent"),
                  str("agent.id", "agent-1"),
                  str("session.id", "session-1"),
                  str("input.value", "Write a tiny plan"),
                  str("output.value", "Call the model and summarize"),
                ],
                endTimeUnixNano: "1710000000800000000",
                kind: 2,
                name: "agent.run",
                spanId: ROOT_SPAN_ID,
                startTimeUnixNano: "1710000000000000000",
                status: { code: 1 },
                traceId: TRACE_ID,
              },
              {
                attributes: [
                  str("openinference.span.kind", "LLM"),
                  str("llm.provider", "openai"),
                  str("llm.model_name", "gpt-5-mini"),
                  int("llm.token_count.prompt", 12),
                  int("llm.token_count.completion", 18),
                  int("llm.token_count.total", 30),
                  double("llm.cost.total", 0.00042),
                  str("llm.input_messages", '[{"role":"user","content":"Plan"}]'),
                  str("llm.output_messages", '[{"role":"assistant","content":"Done"}]'),
                ],
                endTimeUnixNano: "1710000000700000000",
                kind: 3,
                name: "openai.chat.completions",
                parentSpanId: ROOT_SPAN_ID,
                spanId: LLM_SPAN_ID,
                startTimeUnixNano: "1710000000200000000",
                status: { code: 1 },
                traceId: TRACE_ID,
              },
            ],
          },
        ],
      },
    ],
  };
}

export function makeGenAiPayload(): OtlpExportTraceServiceRequest {
  return {
    resourceSpans: [
      {
        resource: {
          attributes: [str("service.name", "genai-service")],
        },
        scopeSpans: [
          {
            scope: { name: "traceloop", version: "0.1.0" },
            spans: [
              {
                attributes: [
                  str("gen_ai.operation.name", "chat"),
                  str("gen_ai.system", "anthropic"),
                  str("gen_ai.request.model", "claude-opus-4"),
                  int("gen_ai.usage.input_tokens", 4),
                  int("gen_ai.usage.output_tokens", 6),
                  int("gen_ai.usage.total_tokens", 10),
                ],
                endTimeUnixNano: "1710000002000000000",
                name: "chat anthropic",
                spanId: "dddddddddddddddd",
                startTimeUnixNano: "1710000001000000000",
                status: { code: 2, message: "boom" },
                traceId: "dddddddddddddddddddddddddddddddd",
              },
            ],
          },
        ],
      },
    ],
  };
}

function str(key: string, value: string) {
  return { key, value: { stringValue: value } };
}

function int(key: string, value: number) {
  return { key, value: { intValue: value } };
}

function double(key: string, value: number) {
  return { key, value: { doubleValue: value } };
}
