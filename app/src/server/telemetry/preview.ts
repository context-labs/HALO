/**
 * Input/output previews for trace rows. Derives a one-line "what the user
 * asked" / "what the agent returned" from a span's canonical message columns
 * (ingest normalizes OpenInference / GenAI semconv / Vercel AI shapes into
 * OpenAI-style arrays) so list views never have to load span payloads.
 */

export const PREVIEW_MAX_LENGTH = 200;

type PreviewSpanContent = {
  input: string | null;
  inputMessages: string | null;
  output: string | null;
  outputMessages: string | null;
};

export function extractInputPreview(span: PreviewSpanContent): string | null {
  const fromMessages = firstTextByRole(span.inputMessages, isUserRole);
  if (fromMessages) return truncatePreview(fromMessages);
  const loose = looseText(span.input, isUserRole);
  return loose ? truncatePreview(loose) : null;
}

export function extractOutputPreview(span: PreviewSpanContent): string | null {
  const fromMessages = lastTextByRole(span.outputMessages, isAssistantRole);
  if (fromMessages) return truncatePreview(fromMessages);
  const loose = looseText(span.output, isAssistantRole);
  return loose ? truncatePreview(loose) : null;
}

/** Collapse whitespace and cap length so summary rows stay small. */
export function truncatePreview(
  text: string,
  maxLength: number = PREVIEW_MAX_LENGTH,
): string {
  const collapsed = text.replace(/\s+/g, " ").trim();
  if (collapsed.length <= maxLength) return collapsed;
  return `${collapsed.slice(0, maxLength - 1).trimEnd()}…`;
}

function firstTextByRole(
  raw: string | null,
  roleMatches: (role: string) => boolean,
): string | null {
  for (const message of messagesFrom(raw)) {
    if (roleMatches(message.role) && message.text.trim()) return message.text;
  }
  return null;
}

function lastTextByRole(
  raw: string | null,
  roleMatches: (role: string) => boolean,
): string | null {
  let found: string | null = null;
  for (const message of messagesFrom(raw)) {
    if (roleMatches(message.role) && message.text.trim()) found = message.text;
  }
  return found;
}

/**
 * The loose input/output columns may hold plain text, a message array, a
 * single message object, or arbitrary JSON. Prefer role-matched message text,
 * fall back to plain text, and ignore unrecognizable JSON blobs (raw tool
 * payloads make useless previews).
 */
function looseText(
  raw: string | null,
  roleMatches: (role: string) => boolean,
): string | null {
  if (!raw?.trim()) return null;
  const parsed = tryParseJson(raw);
  if (parsed === null) return raw;

  const messages = [...normalizedMessages(parsed)];
  const matched = messages.find(
    (message) => roleMatches(message.role) && message.text.trim(),
  );
  if (matched) return matched.text;
  const any = messages.find((message) => message.text.trim());
  if (any) return any.text;
  if (typeof parsed === "string") return parsed;
  if (isRecord(parsed) && typeof parsed.text === "string") return parsed.text;
  return null;
}

type PreviewMessage = { role: string; text: string };

function* messagesFrom(raw: string | null): Generator<PreviewMessage> {
  const parsed = tryParseJson(raw);
  if (parsed === null) return;
  yield* normalizedMessages(parsed);
}

function* normalizedMessages(parsed: unknown): Generator<PreviewMessage> {
  const list = Array.isArray(parsed)
    ? parsed
    : isRecord(parsed) && Array.isArray(parsed.messages)
      ? parsed.messages
      : isRecord(parsed) &&
          isRecord(parsed.context) &&
          Array.isArray(parsed.context.messages)
        ? parsed.context.messages
        : isRecord(parsed)
          ? [parsed]
          : null;
  if (!list) return;
  for (const item of list) {
    if (!isRecord(item)) continue;
    const inner =
      isRecord(item.message) && !("role" in item) ? item.message : item;
    if (!isRecord(inner)) continue;
    const role =
      typeof inner.role === "string"
        ? inner.role
        : typeof inner.type === "string"
          ? inner.type
          : "";
    yield { role: role.toLowerCase(), text: stringifyContent(inner.content) };
  }
}

function isUserRole(role: string) {
  return role === "user" || role === "human";
}

function isAssistantRole(role: string) {
  return role === "assistant" || role === "ai" || role === "model";
}

/** Flatten string | parts-array | object content into displayable text. */
function stringifyContent(content: unknown): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (!isRecord(part)) return "";
        if (typeof part.text === "string") return part.text;
        return "";
      })
      .filter(Boolean)
      .join(" ");
  }
  if (isRecord(content) && typeof content.text === "string") return content.text;
  return "";
}

function tryParseJson(raw: string | null): unknown {
  if (!raw) return null;
  const trimmed = raw.trim();
  if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return null;
  try {
    return JSON.parse(trimmed);
  } catch {
    return null;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
