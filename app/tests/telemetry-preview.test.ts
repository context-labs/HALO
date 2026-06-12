import { describe, expect, test } from "bun:test";

import {
  extractInputPreview,
  extractOutputPreview,
  truncatePreview,
} from "../src/server/telemetry/preview";

const empty = {
  input: null,
  inputMessages: null,
  output: null,
  outputMessages: null,
};

describe("extractInputPreview", () => {
  test("prefers the first user message from inputMessages", () => {
    const preview = extractInputPreview({
      ...empty,
      input: '{"raw": "blob"}',
      inputMessages: JSON.stringify([
        { role: "system", content: "You are a helpful agent." },
        { role: "user", content: "Refactor token refresh in the auth module." },
        { role: "user", content: "Second question" },
      ]),
    });
    expect(preview).toBe("Refactor token refresh in the auth module.");
  });

  test("handles parts-array content", () => {
    const preview = extractInputPreview({
      ...empty,
      inputMessages: JSON.stringify([
        { role: "user", content: [{ type: "text", text: "Hello" }, { type: "image" }] },
      ]),
    });
    expect(preview).toBe("Hello");
  });

  test("falls back to plain-text input", () => {
    expect(
      extractInputPreview({ ...empty, input: "  Why did the build\nfail?  " }),
    ).toBe("Why did the build fail?");
  });

  test("extracts user message from JSON input column", () => {
    const preview = extractInputPreview({
      ...empty,
      input: JSON.stringify({
        messages: [{ role: "user", content: "From the loose column" }],
      }),
    });
    expect(preview).toBe("From the loose column");
  });

  test("accepts a bare single message object", () => {
    expect(
      extractInputPreview({
        ...empty,
        input: JSON.stringify({ role: "user", content: "Single message" }),
      }),
    ).toBe("Single message");
  });

  test("returns null for unusable JSON and empty values", () => {
    expect(extractInputPreview({ ...empty })).toBeNull();
    expect(
      extractInputPreview({ ...empty, input: '{"pattern": "refreshToken"}' }),
    ).toBeNull();
  });
});

describe("extractOutputPreview", () => {
  test("uses the last assistant message", () => {
    const preview = extractOutputPreview({
      ...empty,
      outputMessages: JSON.stringify([
        { role: "assistant", content: "Working on it…" },
        { role: "tool", content: "{}" },
        { role: "assistant", content: "Done — suite is green." },
      ]),
    });
    expect(preview).toBe("Done — suite is green.");
  });

  test("falls back to plain-text output", () => {
    expect(
      extractOutputPreview({ ...empty, output: "Wrote 14.2 KB." }),
    ).toBe("Wrote 14.2 KB.");
  });

  test("skips assistant messages with empty content", () => {
    const preview = extractOutputPreview({
      ...empty,
      outputMessages: JSON.stringify([
        { role: "assistant", content: "Real answer" },
        { role: "assistant", content: "", tool_calls: [{ function: { name: "x" } }] },
      ]),
    });
    expect(preview).toBe("Real answer");
  });
});

describe("truncatePreview", () => {
  test("collapses whitespace", () => {
    expect(truncatePreview("a\n\n  b\tc")).toBe("a b c");
  });

  test("caps length with ellipsis", () => {
    const long = "x".repeat(400);
    const out = truncatePreview(long);
    expect(out.length).toBeLessThanOrEqual(200);
    expect(out.endsWith("…")).toBe(true);
  });
});
