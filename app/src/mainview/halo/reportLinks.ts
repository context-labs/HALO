export type DashboardLink =
  | { kind: "span"; spanId: string; traceId: string }
  | { kind: "trace"; traceId: string };

const DASHBOARD_TAG_PATTERN =
  /\[(trace|span):([0-9a-fA-F]+)(?::([0-9a-fA-F]+))?\]/gi;

export function linkifyDashboardTags(markdown: string) {
  let inFence = false;
  return markdown
    .split("\n")
    .map((line) => {
      if (line.trimStart().startsWith("```")) {
        inFence = !inFence;
        return line;
      }
      if (inFence) return line;
      return linkifyLine(line);
    })
    .join("\n");
}

export function parseDashboardLink(href: string | undefined): DashboardLink | null {
  const traceMatch = href?.match(/^#halo-trace-([0-9a-fA-F]+)$/);
  if (traceMatch?.[1]) {
    return { kind: "trace", traceId: traceMatch[1].toLowerCase() };
  }

  const spanMatch = href?.match(/^#halo-span-([0-9a-fA-F]+)-([0-9a-fA-F]+)$/);
  if (spanMatch?.[1] && spanMatch[2]) {
    return {
      kind: "span",
      spanId: spanMatch[2].toLowerCase(),
      traceId: spanMatch[1].toLowerCase(),
    };
  }

  return null;
}

function linkifyLine(line: string) {
  let output = "";
  let cursor = 0;

  while (cursor < line.length) {
    const tickStart = line.indexOf("`", cursor);
    if (tickStart === -1) {
      output += linkifyTextSegment(line.slice(cursor));
      break;
    }

    output += linkifyTextSegment(line.slice(cursor, tickStart));
    const tickEnd = endOfCodeSpan(line, tickStart);
    if (tickEnd === -1) {
      output += line.slice(tickStart);
      break;
    }

    const markerLength = codeSpanMarkerLength(line, tickStart);
    const codeSpan = line.slice(tickStart, tickEnd + markerLength);
    const codeContent = line.slice(tickStart + markerLength, tickEnd);
    const tag = parseDashboardTag(codeContent.trim());
    output += tag ? dashboardLinkMarkdown(tag) : codeSpan;
    cursor = tickEnd + markerLength;
  }

  return output;
}

function linkifyTextSegment(segment: string) {
  return segment.replace(DASHBOARD_TAG_PATTERN, (match, kind, traceId, spanId, offset, source) => {
    if (source[offset + match.length] === "(") return match;

    const tag = parseDashboardTag(match);
    return tag ? dashboardLinkMarkdown(tag) : match;
  });
}

function parseDashboardTag(text: string): DashboardLink | null {
  const match = /^\[(trace|span):([0-9a-fA-F]+)(?::([0-9a-fA-F]+))?\]$/i.exec(text);
  if (!match?.[1] || !match[2]) return null;

  const kind = match[1].toLowerCase();
  const traceId = match[2].toLowerCase();
  const spanId = match[3]?.toLowerCase();
  if (kind === "trace" && !spanId) return { kind: "trace", traceId };
  if (kind === "span" && spanId) return { kind: "span", spanId, traceId };
  return null;
}

function dashboardLinkMarkdown(tag: DashboardLink) {
  if (tag.kind === "trace") {
    return `[halo-trace-${tag.traceId}](#halo-trace-${tag.traceId})`;
  }
  return `[halo-span-${tag.traceId}-${tag.spanId}](#halo-span-${tag.traceId}-${tag.spanId})`;
}

export function dashboardLinkLabel(tag: DashboardLink) {
  if (tag.kind === "trace") return `[trace:${tag.traceId}]`;
  return `[span:${tag.traceId}:${tag.spanId}]`;
}

function endOfCodeSpan(line: string, tickStart: number) {
  const markerLength = codeSpanMarkerLength(line, tickStart);
  const marker = "`".repeat(markerLength);
  return line.indexOf(marker, tickStart + markerLength);
}

function codeSpanMarkerLength(line: string, tickStart: number) {
  let length = 0;
  while (line[tickStart + length] === "`") length += 1;
  return length;
}
