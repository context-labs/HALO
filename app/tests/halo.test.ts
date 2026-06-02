import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, describe, expect, test } from "bun:test";
import { createDatabase, ensureSchema } from "../src/server/db/client";
import { exportHaloTraceJsonl, previewHaloRunExport } from "../src/server/halo/exporter";
import { haloRunnerPathCandidates } from "../src/server/halo/runQueue";
import {
  getHaloProvider,
  normalizedHaloInstallPath,
  saveHaloProvider,
} from "../src/server/halo/storage";
import { ingestTelemetry } from "../src/server/telemetry/storage";
import { TRACE_ID, makeTracePayload } from "./support/otlp-fixtures";

let tempDirs: string[] = [];

afterEach(() => {
  for (const dir of tempDirs) {
    rmSync(dir, { force: true, recursive: true });
  }
  tempDirs = [];
});

describe("HALO provider settings", () => {
  test("stores provider secrets locally but returns masked values to the UI", () => {
    const database = createDatabase(":memory:");
    ensureSchema(database.sqlite);

    try {
      const saved = saveHaloProvider(database.sqlite, {
        apiKey: "sk-test-1234567890",
        baseUrl: "https://api.openai.com/v1/",
        model: "gpt-test",
        name: "OpenAI",
        providerType: "openai",
      });

      expect(saved.apiKeyMasked).toBe("sk-t…7890");
      expect("apiKey" in saved).toBe(false);
      expect(saved.baseUrl).toBe("https://api.openai.com/v1");

      const stored = getHaloProvider(database.sqlite, saved.id);
      expect(stored?.apiKey).toBe("sk-test-1234567890");
      expect(stored?.headers).toEqual({});
    } finally {
      database.sqlite.close(false);
    }
  });
});

describe("HALO production paths", () => {
  test("repairs legacy app-bundle HALO engine paths to app data", () => {
    const defaultPath =
      "/Users/example/Library/Application Support/net.inference.halo/halo-engine";

    expect(
      normalizedHaloInstallPath(
        "/Applications/HALO.app/Contents/MacOS/data/halo-engine",
        defaultPath,
      ),
    ).toBe(defaultPath);
    expect(normalizedHaloInstallPath("/opt/halo-engine", defaultPath)).toBe(
      resolve("/opt/halo-engine"),
    );
  });

  test("looks for the bundled runner beside packaged app scripts", () => {
    const candidates = haloRunnerPathCandidates({
      cwd: "/tmp/halo",
      env: {},
      importMetaUrl:
        "file:///Applications/HALO.app/Contents/Resources/app/bun/main.js",
    });

    expect(candidates).toContain(
      "/Applications/HALO.app/Contents/Resources/app/scripts/halo-local-runner.py",
    );
  });
});

describe("HALO trace export", () => {
  test("previews and exports trace groups as HALO JSONL", () => {
    const database = createDatabase(":memory:");
    ensureSchema(database.sqlite);
    ingestTelemetry(database.sqlite, {
      body: JSON.stringify(makeTracePayload()),
      contentEncoding: "test",
      sizeBytes: 1,
    });
    const outputDir = mkdtempSync(join(tmpdir(), "halo-export-"));
    tempDirs.push(outputDir);

    try {
      const preview = previewHaloRunExport(database.sqlite, {
        filters: { serviceNames: ["halo-agent"] },
        targetType: "trace_group",
      });
      expect(preview.traceCount).toBe(1);
      expect(preview.spanCount).toBe(2);

      const exported = exportHaloTraceJsonl(database.sqlite, {
        filters: { serviceNames: ["halo-agent"] },
        outputDir,
        runId: "run-1",
        targetType: "trace_group",
      });
      const lines = readFileSync(exported.path, "utf8").trim().split("\n");
      const first = JSON.parse(lines[0] ?? "{}");

      expect(lines).toHaveLength(2);
      expect(first.trace_id).toBe(TRACE_ID);
      expect(first.parent_span_id).toBe("");
      expect(first.resource.attributes["service.name"]).toBe("halo-agent");
      expect(first.attributes["openinference.span.kind"]).toBe("AGENT");
      expect(first.attributes["input.value"]).toBe("Write a tiny plan");
    } finally {
      database.sqlite.close(false);
    }
  });

  test("trace-group export honors exact trace ID filters", () => {
    const database = createDatabase(":memory:");
    ensureSchema(database.sqlite);
    const otherTraceId = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    ingestTelemetry(database.sqlite, {
      body: JSON.stringify(makeTracePayload()),
      contentEncoding: "test",
      sizeBytes: 1,
    });
    ingestTelemetry(database.sqlite, {
      body: JSON.stringify(
        JSON.parse(JSON.stringify(makeTracePayload()).replaceAll(TRACE_ID, otherTraceId)),
      ),
      contentEncoding: "test",
      sizeBytes: 1,
    });
    const outputDir = mkdtempSync(join(tmpdir(), "halo-trace-filter-export-"));
    tempDirs.push(outputDir);

    try {
      const preview = previewHaloRunExport(database.sqlite, {
        filters: { traceId: otherTraceId },
        targetType: "trace_group",
      });
      expect(preview.traceCount).toBe(1);
      expect(preview.spanCount).toBe(2);

      const exported = exportHaloTraceJsonl(database.sqlite, {
        filters: { traceId: otherTraceId },
        outputDir,
        runId: "run-trace-filter",
        targetType: "trace_group",
      });
      const traceIds = readFileSync(exported.path, "utf8")
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line).trace_id);

      expect(new Set(traceIds)).toEqual(new Set([otherTraceId]));
    } finally {
      database.sqlite.close(false);
    }
  });

  test("session-group export includes all traces in matched sessions", () => {
    const database = createDatabase(":memory:");
    ensureSchema(database.sqlite);
    ingestTelemetry(database.sqlite, {
      body: JSON.stringify(makeTracePayload()),
      contentEncoding: "test",
      sizeBytes: 1,
    });
    const outputDir = mkdtempSync(join(tmpdir(), "halo-session-export-"));
    tempDirs.push(outputDir);

    try {
      const preview = previewHaloRunExport(database.sqlite, {
        filters: { sessionIds: ["session-1"] },
        targetType: "session_group",
      });
      expect(preview.sessionCount).toBe(1);
      expect(preview.traceCount).toBe(1);
      expect(preview.spanCount).toBe(2);

      const exported = exportHaloTraceJsonl(database.sqlite, {
        filters: { sessionIds: ["session-1"] },
        outputDir,
        runId: "run-2",
        targetType: "session_group",
      });
      expect(readFileSync(exported.path, "utf8").trim().split("\n")).toHaveLength(2);
    } finally {
      database.sqlite.close(false);
    }
  });

  test("session-group export accepts persisted ISO date filters", () => {
    const database = createDatabase(":memory:");
    ensureSchema(database.sqlite);
    ingestTelemetry(database.sqlite, {
      body: JSON.stringify(makeTracePayload()),
      contentEncoding: "test",
      sizeBytes: 1,
    });
    const outputDir = mkdtempSync(join(tmpdir(), "halo-session-date-export-"));
    tempDirs.push(outputDir);

    try {
      const preview = previewHaloRunExport(database.sqlite, {
        filters: {
          startDate: "2024-03-09T15:59:59.000Z",
        },
        targetType: "session_group",
      });
      expect(preview.sessionCount).toBe(1);
      expect(preview.traceCount).toBe(1);
      expect(preview.spanCount).toBe(2);

      const exported = exportHaloTraceJsonl(database.sqlite, {
        filters: {
          endDate: "2024-03-09T16:00:01.000Z",
          startDate: "2024-03-09T15:59:59.000Z",
        },
        outputDir,
        runId: "run-session-date-filter",
        targetType: "session_group",
      });
      expect(readFileSync(exported.path, "utf8").trim().split("\n")).toHaveLength(2);
    } finally {
      database.sqlite.close(false);
    }
  });
});
