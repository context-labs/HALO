import { describe, expect, test } from "bun:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import {
  dashboardLinkLabel,
  linkifyDashboardTags,
  parseDashboardLink,
} from "../src/mainview/halo/reportLinks";
import { RunReportView } from "../src/mainview/halo/RunReportView";

describe("halo report dashboard links", () => {
  test("linkifies bracket-only trace and span tags before markdown parsing", () => {
    const markdown = [
      "See [trace:0123456789abcdef0123456789abcdef].",
      "Open [span:0123456789abcdef0123456789abcdef:fedcba9876543210], then inspect.",
    ].join("\n");

    expect(linkifyDashboardTags(markdown)).toBe(
      [
        "See [halo-trace-0123456789abcdef0123456789abcdef](#halo-trace-0123456789abcdef0123456789abcdef).",
        "Open [halo-span-0123456789abcdef0123456789abcdef-fedcba9876543210](#halo-span-0123456789abcdef0123456789abcdef-fedcba9876543210), then inspect.",
      ].join("\n"),
    );
  });

  test("accepts uppercase and variable-length hex ids but normalizes links", () => {
    const markdown = "Check [TRACE:ABCDEF] and [SPAN:ABCDEF:123ABC].";

    expect(linkifyDashboardTags(markdown)).toBe(
      "Check [halo-trace-abcdef](#halo-trace-abcdef) and [halo-span-abcdef-123abc](#halo-span-abcdef-123abc).",
    );
  });

  test("does not link bare ids or malformed trace and span text", () => {
    const markdown = [
      "Bare 0123456789abcdef0123456789abcdef stays text.",
      "Missing bracket trace:0123456789abcdef stays text.",
      "Incomplete [span:0123456789abcdef] stays text.",
      "Non-hex [trace:not-a-trace] stays text.",
    ].join("\n");

    expect(linkifyDashboardTags(markdown)).toBe(markdown);
  });

  test("handles code spans and fences without corrupting markdown", () => {
    const markdown = [
      "Exact inline code `[trace:ABCDEF]` links.",
      "Mixed inline code `look at [trace:ABCDEF]` stays code.",
      "```",
      "[trace:ABCDEF]",
      "```",
    ].join("\n");

    expect(linkifyDashboardTags(markdown)).toBe(
      [
        "Exact inline code [halo-trace-abcdef](#halo-trace-abcdef) links.",
        "Mixed inline code `look at [trace:ABCDEF]` stays code.",
        "```",
        "[trace:ABCDEF]",
        "```",
      ].join("\n"),
    );
  });

  test("does not rewrite an existing markdown link label", () => {
    expect(linkifyDashboardTags("[trace:abcdef](https://example.test)")).toBe(
      "[trace:abcdef](https://example.test)",
    );
  });

  test("parses internal dashboard hrefs with normalized ids", () => {
    expect(parseDashboardLink("#halo-trace-ABCDEF")).toEqual({
      kind: "trace",
      traceId: "abcdef",
    });
    expect(parseDashboardLink("#halo-span-ABCDEF-123ABC")).toEqual({
      kind: "span",
      spanId: "123abc",
      traceId: "abcdef",
    });
    expect(parseDashboardLink("https://example.test")).toBeNull();
  });

  test("formats dashboard link labels as the original model tag markings", () => {
    expect(dashboardLinkLabel({ kind: "trace", traceId: "abcdef" })).toBe(
      "[trace:abcdef]",
    );
    expect(
      dashboardLinkLabel({
        kind: "span",
        spanId: "123abc",
        traceId: "abcdef",
      }),
    ).toBe("[span:abcdef:123abc]");
  });

  test("renders report tags as clickable buttons with tag labels", () => {
    const html = renderToStaticMarkup(
      createElement(RunReportView, {
        markdown: [
          "See [trace:ABCDEF].",
          "Open `[span:ABCDEF:123ABC]`.",
          "Bare 0123456789abcdef should not link.",
        ].join("\n"),
        onOpenSpanLink: () => {},
        onOpenTraceLink: () => {},
      }),
    );

    expect(html).toContain('data-halo-report-link="trace"');
    expect(html).toContain('data-trace-id="abcdef"');
    expect(html).toContain("[trace:abcdef]");
    expect(html).toContain('data-halo-report-link="span"');
    expect(html).toContain('data-span-id="123abc"');
    expect(html).toContain("[span:abcdef:123abc]");
    expect(html).not.toContain("halo-trace-abcdef");
    expect(html).toContain("Bare 0123456789abcdef should not link.");
  });
});
