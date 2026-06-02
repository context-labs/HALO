import type { OtlpExportTraceServiceRequest } from "../src/server/telemetry/otlp";

const DEFAULT_ENDPOINT = "http://127.0.0.1:8799/v1/traces";
const DEFAULT_SPAN_COUNT = 10;
const DEFAULT_DELAY_MS = 150;
const MIN_DURATION_MS = 1_000;
const MAX_DURATION_MS = 3_000;

type RandomSource = () => number;

export type FiredTestSpan = {
  durationMs: number;
  endTimeUnixNano: string;
  name: string;
  spanId: string;
  startTimeUnixNano: string;
};

export type FireTestSpansResult = {
  endpoint: string;
  spans: FiredTestSpan[];
  traceId: string;
};

export type FireTestSpansOptions = {
  delayMs?: number;
  endpoint?: string;
  rng?: RandomSource;
  spanCount?: number;
};

export async function fireTestSpans(
  options: FireTestSpansOptions = {},
): Promise<FireTestSpansResult> {
  const endpoint = options.endpoint ?? Bun.env.CATALYST_OTLP_ENDPOINT ?? DEFAULT_ENDPOINT;
  const spanCount = options.spanCount ?? DEFAULT_SPAN_COUNT;
  const delayMs = options.delayMs ?? DEFAULT_DELAY_MS;
  const traceId = randomHexId(16);
  const spans = buildTestSpans({
    rng: options.rng ?? Math.random,
    spanCount,
    traceId,
  });

  for (const span of spans) {
    const response = await fetch(endpoint, {
      body: JSON.stringify(makeSingleSpanPayload(traceId, span)),
      headers: { "content-type": "application/json" },
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(
        `failed to fire ${span.name}: ${response.status} ${response.statusText}`,
      );
    }
    if (delayMs > 0) {
      await Bun.sleep(delayMs);
    }
  }

  return {
    endpoint,
    spans,
    traceId,
  };
}

export function buildTestSpans({
  rng = Math.random,
  spanCount = DEFAULT_SPAN_COUNT,
  traceId = randomHexId(16),
}: {
  rng?: RandomSource;
  spanCount?: number;
  traceId?: string;
} = {}): FiredTestSpan[] {
  if (!Number.isInteger(spanCount) || spanCount < 1) {
    throw new Error("spanCount must be a positive integer");
  }

  const baseStart = BigInt(Date.now()) * 1_000_000n;
  return Array.from({ length: spanCount }, (_, index) => {
    const durationMs = randomDurationMs(rng);
    const start = baseStart + BigInt(index * 250) * 1_000_000n;
    const end = start + BigInt(durationMs) * 1_000_000n;
    return {
      durationMs,
      endTimeUnixNano: String(end),
      name: `test.random_duration_span.${index + 1}`,
      spanId: randomHexId(8),
      startTimeUnixNano: String(start),
    };
  });
}

export function randomDurationMs(rng: RandomSource = Math.random) {
  const value = Math.min(0.999_999_999, Math.max(0, rng()));
  return MIN_DURATION_MS + Math.floor(value * (MAX_DURATION_MS - MIN_DURATION_MS + 1));
}

function makeSingleSpanPayload(
  traceId: string,
  span: FiredTestSpan,
): OtlpExportTraceServiceRequest {
  return {
    resourceSpans: [
      {
        resource: {
          attributes: [
            stringAttr("service.name", "halo-span-fire-test"),
            stringAttr("service.version", "0.1.0"),
            stringAttr("deployment.environment", "local"),
          ],
        },
        scopeSpans: [
          {
            scope: {
              name: "@context-labs/halo-app/scripts.fire-test-spans",
              version: "0.1.0",
            },
            spans: [
              {
                attributes: [
                  stringAttr("openinference.span.kind", "CHAIN"),
                  stringAttr("agent.name", "Span fire test"),
                  stringAttr("agent.id", "span-fire-test"),
                  stringAttr("session.id", `span-fire-test:${traceId}`),
                  stringAttr("input.value", `Start ${span.name}`),
                  stringAttr(
                    "output.value",
                    `${span.name} completed in ${span.durationMs}ms`,
                  ),
                  intAttr("halo.test.duration_ms", span.durationMs),
                ],
                endTimeUnixNano: span.endTimeUnixNano,
                kind: 2,
                name: span.name,
                spanId: span.spanId,
                startTimeUnixNano: span.startTimeUnixNano,
                status: { code: 1 },
                traceId,
              },
            ],
          },
        ],
      },
    ],
  };
}

function stringAttr(key: string, value: string) {
  return { key, value: { stringValue: value } };
}

function intAttr(key: string, value: number) {
  return { key, value: { intValue: value } };
}

function randomHexId(byteLength: number) {
  const bytes = crypto.getRandomValues(new Uint8Array(byteLength));
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function numberFlag(args: string[], name: string, fallback: number) {
  const index = args.indexOf(name);
  if (index === -1) return fallback;
  const raw = args[index + 1];
  const value = raw == null ? Number.NaN : Number(raw);
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be a number`);
  }
  return value;
}

function stringFlag(args: string[], name: string, fallback: string) {
  const index = args.indexOf(name);
  if (index === -1) return fallback;
  const value = args[index + 1];
  if (!value) {
    throw new Error(`${name} requires a value`);
  }
  return value;
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  const result = await fireTestSpans({
    delayMs: numberFlag(args, "--delay-ms", DEFAULT_DELAY_MS),
    endpoint: stringFlag(
      args,
      "--endpoint",
      Bun.env.CATALYST_OTLP_ENDPOINT ?? DEFAULT_ENDPOINT,
    ),
    spanCount: numberFlag(args, "--count", DEFAULT_SPAN_COUNT),
  });

  console.log(JSON.stringify(result, null, 2));
}
