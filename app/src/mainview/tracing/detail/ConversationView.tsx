import { useMemo, useState } from "react";
import { ChevronRight, User } from "lucide-react";

import { cn } from "~/lib/ui";
import {
  ChatAgentContent,
  ChatAgentText,
  ChatAvatar,
  ChatTurn,
  ChatUserBubble,
} from "~/components/chat";
import { compactNumber, formatDuration, formatMoney } from "~/lib/format";
import type { Span, SpanNode } from "../../../server/telemetry/types";
import {
  buildSessionConversation,
  buildTraceConversation,
  type ConversationGroup,
  type ConversationMessage,
} from "./conversation";
import { spanColor, useTraceColors, type TraceColors } from "./spanKinds";

export function ConversationView({
  heat,
  maxCost,
  mode,
  onSelect,
  selectedSpanKey,
  spans,
  tree,
}: {
  heat: boolean;
  maxCost: number;
  mode: "session" | "trace";
  onSelect: (key: string | null) => void;
  selectedSpanKey: string | null;
  spans: Span[];
  tree: SpanNode[];
}) {
  const colors = useTraceColors();
  const model = useMemo(
    () =>
      mode === "session"
        ? buildSessionConversation(tree)
        : buildTraceConversation(spans, tree),
    [mode, spans, tree],
  );

  return (
    <div className="flex min-h-0 min-w-0 flex-1">
      <div
        className="min-w-0 flex-1 overflow-y-auto"
        onClick={() => onSelect(null)}
      >
        <div className="mx-auto max-w-3xl space-y-7 px-6 pb-16 pt-8">
          {model.messages.map((message, index) => (
            <ConversationBubble
              colors={colors}
              heat={heat}
              key={`${message.role}-${message.atMs}-${index}`}
              maxCost={maxCost}
              message={message}
              onSelect={onSelect}
              selectedSpanKey={selectedSpanKey}
            />
          ))}
          {/* pl-10 = avatar width + gutter, aligning with the message bodies. */}
          <div className="flex items-center gap-2 pl-10 text-xs text-muted-foreground">
            <span
              className={cn(
                "h-1.5 w-1.5 rounded-full",
                model.outcome.ok ? "bg-detail-success" : "bg-detail-failure",
              )}
            />
            run {model.outcome.ok ? "completed" : "failed"} ·{" "}
            {formatDuration(model.outcome.durationMs)}
            {model.outcome.costTotal != null
              ? ` · ${formatMoney(model.outcome.costTotal)}`
              : ""}
          </div>
        </div>
      </div>
    </div>
  );
}

function ConversationBubble({
  colors,
  heat,
  maxCost,
  message,
  onSelect,
  selectedSpanKey,
}: {
  colors: TraceColors;
  heat: boolean;
  maxCost: number;
  message: ConversationMessage;
  onSelect: (key: string) => void;
  selectedSpanKey: string | null;
}) {
  const isUser = message.role === "user";
  return (
    <ChatTurn
      avatar={
        isUser ? (
          <ChatAvatar className="bg-muted">
            <User className="h-3.5 w-3.5 text-muted-foreground" />
          </ChatAvatar>
        ) : (
          <ChatAvatar className="bg-gradient-to-br from-trace-llm to-trace-agent text-[11px] font-semibold text-white">
            ⊶
          </ChatAvatar>
        )
      }
      name={isUser ? "User" : "Agent"}
      timestamp={isoTime(message.atMs)}
    >
      {isUser ? (
        <ChatUserBubble text={message.text} />
      ) : (
        <ChatAgentContent>
          {message.text ? <ChatAgentText text={message.text} /> : null}
          {message.group ? (
            <div className={message.text ? "pt-2" : undefined}>
              <SpansDisclosure
                colors={colors}
                group={message.group}
                heat={heat}
                maxCost={maxCost}
                onSelect={onSelect}
                selectedSpanKey={selectedSpanKey}
              />
            </div>
          ) : null}
        </ChatAgentContent>
      )}
    </ChatTurn>
  );
}

function SpansDisclosure({
  colors,
  group,
  heat,
  maxCost,
  onSelect,
  selectedSpanKey,
}: {
  colors: TraceColors;
  group: ConversationGroup;
  heat: boolean;
  maxCost: number;
  onSelect: (key: string) => void;
  selectedSpanKey: string | null;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button
        className={cn(
          "inline-flex items-center gap-2 rounded-lg border px-2.5 py-1.5 text-[11.5px] transition-colors",
          open
            ? "border-border/70 bg-muted text-foreground"
            : "border-subtle bg-background-muted/60 text-muted-foreground hover:bg-muted/60 hover:text-foreground",
        )}
        onClick={(event) => {
          event.stopPropagation();
          setOpen((current) => !current);
        }}
        type="button"
      >
        <ChevronRight
          className={cn("h-3 w-3 transition-transform", open && "rotate-90")}
        />
        <span className="font-semibold">
          {group.rows.length} {group.rows.length === 1 ? "span" : "spans"}
        </span>
        {group.toolCallCount > 0 ? (
          <>
            <span className="text-muted-foreground/70">·</span>
            <span>
              {group.toolCallCount}{" "}
              {group.toolCallCount === 1 ? "tool call" : "tool calls"}
            </span>
          </>
        ) : null}
        <span className="text-muted-foreground/70">·</span>
        <span className="font-geist-mono text-muted-foreground">
          {formatDuration(Math.max(0, group.endMs - group.startMs))}
        </span>
        {group.costTotal != null ? (
          <>
            <span className="text-muted-foreground/70">·</span>
            <span className="font-geist-mono text-muted-foreground">
              {formatMoney(group.costTotal)}
            </span>
          </>
        ) : null}
        {group.failedCount > 0 ? (
          <span className="font-semibold text-detail-failure">
            · {group.failedCount} failed
          </span>
        ) : null}
      </button>
      {open ? (
        <div className="mt-2 overflow-hidden rounded-lg border border-subtle bg-background-muted/50 py-1">
          {group.rows.map((row) => {
            const selected = row.key === selectedSpanKey;
            const error = row.span.statusCode === "STATUS_CODE_ERROR";
            const color = spanColor(row.span, colors, heat, maxCost);
            return (
              <div
                className={cn(
                  "flex cursor-pointer items-center gap-2 border-l-2 py-1.5 pr-2.5",
                  selected
                    ? "border-l-detail-brand bg-muted"
                    : "border-l-transparent hover:bg-muted/50",
                )}
                key={row.key}
                onClick={(event) => {
                  event.stopPropagation();
                  onSelect(row.key);
                }}
                role="button"
                style={{ paddingLeft: 12 + row.depth * 16 }}
                tabIndex={0}
              >
                <span
                  className="h-[7px] w-[7px] flex-none rounded-sm"
                  style={{ background: color }}
                />
                <span
                  className={cn(
                    "min-w-0 flex-1 truncate text-xs",
                    selected
                      ? "font-semibold text-foreground"
                      : "text-muted-foreground",
                  )}
                >
                  {row.span.spanName}
                </span>
                <span
                  className="flex-none text-[9px] font-semibold uppercase tracking-[0.07em]"
                  style={{ color }}
                >
                  {row.span.observationKind.toLowerCase()}
                </span>
                {row.span.observationKind === "LLM" &&
                row.span.totalTokens != null ? (
                  <span className="w-12 flex-none text-right font-geist-mono text-[10px] text-muted-foreground">
                    {compactNumber(row.span.totalTokens)} tok
                  </span>
                ) : null}
                <span
                  className={cn(
                    "w-12 flex-none text-right font-geist-mono text-[10px]",
                    error ? "text-detail-failure" : "text-muted-foreground",
                  )}
                >
                  {error ? "✕ " : ""}
                  {formatDuration(
                    Math.max(0, row.span.endTimeMs - row.span.startTimeMs),
                  )}
                </span>
              </div>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

function isoTime(ms: number): string | undefined {
  if (!Number.isFinite(ms) || ms <= 0) return undefined;
  return new Date(ms).toISOString();
}
