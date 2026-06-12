import { useState } from "react";
import { ChevronDown, ChevronRight, Wrench } from "lucide-react";

import { Badge, cn } from "~/lib/ui";
import type {
  ParsedConversation,
  ParsedMessage,
  ThreadRole,
} from "../llmMessages";
import { CopyTextButton } from "./CopyTextButton";

const ROLE_VARIANT: Record<ThreadRole, "outline" | "secondary" | "status-brand" | "status-warning"> = {
  assistant: "status-brand",
  other: "outline",
  system: "outline",
  tool: "status-warning",
  user: "secondary",
};

const COLLAPSE_THRESHOLD = 1_200;

export function LlmSpanView({ conversation }: { conversation: ParsedConversation }) {
  return (
    <div className="space-y-3">
      {conversation.messages.map((message) => (
        <MessageRow key={message.key} message={message} />
      ))}
    </div>
  );
}

function MessageRow({ message }: { message: ParsedMessage }) {
  return (
    <div className="relative rounded-md border border-subtle bg-background-muted/60 p-3">
      {/* pr-9 keeps the source tag clear of the corner copy button. */}
      <div className="flex items-center gap-2 pr-9">
        <Badge size="sm" variant={ROLE_VARIANT[message.role]}>
          {message.roleLabel}
        </Badge>
        {message.toolCallId ? (
          <span className="font-mono text-[11px] text-muted-foreground">
            for {message.toolCallId}
          </span>
        ) : null}
        <span className="ml-auto text-[11px] uppercase tracking-wide text-muted-foreground/60">
          {message.source}
        </span>
      </div>
      {message.content ? <CopyTextButton text={message.content} /> : null}
      {message.content ? (
        <CollapsibleText text={message.content} />
      ) : null}
      {message.toolCalls.map((toolCall, index) => (
        <ToolCallChip key={toolCall.id ?? index} toolCall={toolCall} />
      ))}
    </div>
  );
}

function CollapsibleText({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  const collapsible = text.length > COLLAPSE_THRESHOLD;
  const visible =
    collapsible && !expanded ? `${text.slice(0, COLLAPSE_THRESHOLD)}…` : text;
  return (
    <div className="mt-2">
      <pre
        className={cn(
          "whitespace-pre-wrap break-words font-sans text-sm leading-relaxed",
        )}
      >
        {visible}
      </pre>
      {collapsible ? (
        <button
          className="mt-1 text-xs font-medium text-link hover:underline"
          onClick={() => setExpanded((current) => !current)}
          type="button"
        >
          {expanded ? "Show less" : `Show all ${text.length.toLocaleString()} characters`}
        </button>
      ) : null}
    </div>
  );
}

function ToolCallChip({ toolCall }: { toolCall: { argsRaw: string; id?: string; name: string } }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-2 rounded-md border border-detail-warning/30 bg-detail-warning/5">
      <button
        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-sm"
        onClick={() => setOpen((current) => !current)}
        type="button"
      >
        {open ? (
          <ChevronDown className="h-3 w-3 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground" />
        )}
        <Wrench className="h-3.5 w-3.5 shrink-0 text-detail-warning" />
        <span className="truncate font-mono text-xs">{toolCall.name}</span>
        {toolCall.id ? (
          <span className="ml-auto truncate font-mono text-[11px] text-muted-foreground">
            {toolCall.id}
          </span>
        ) : null}
      </button>
      {open ? (
        <pre className="max-h-56 overflow-auto border-t border-detail-warning/20 p-2.5 text-xs leading-relaxed">
          {prettyArgs(toolCall.argsRaw) || "(no arguments)"}
        </pre>
      ) : null}
    </div>
  );
}

function prettyArgs(raw: string) {
  if (!raw) return "";
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw;
  }
}
