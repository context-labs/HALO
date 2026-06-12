import type { ReactNode } from "react";

import { cn } from "~/lib/ui";
import { formatTimestamp } from "~/lib/format";

/**
 * Shared chat atoms used by the HALO analysis thread and the trace
 * conversation view, so both read identically: a hanging avatar in a 16px
 * gutter, a name + timestamp header, and body content that left-aligns with
 * the name. Body text dims to 85% so sender names carry the contrast.
 */

/** One turn: avatar gutter + header + content column. */
export function ChatTurn({
  avatar,
  children,
  name,
  timestamp,
}: {
  avatar: ReactNode;
  children: ReactNode;
  name: string;
  /** ISO timestamp; omitted when unknown. */
  timestamp?: string;
}) {
  return (
    <div className="turn-fade-in flex gap-4">
      {avatar}
      <div className="min-w-0 flex-1">
        <div className="flex h-6 items-center gap-2">
          <span className="text-sm font-semibold">{name}</span>
          {timestamp ? (
            <span className="text-xs text-muted-foreground">
              {formatTimestamp(timestamp)}
            </span>
          ) : null}
        </div>
        {children}
      </div>
    </div>
  );
}

/** The 24px round avatar that hangs in the gutter. */
export function ChatAvatar({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "grid h-6 w-6 flex-none place-items-center rounded-full",
        className,
      )}
    >
      {children}
    </span>
  );
}

/**
 * User message bubble. -mx-4 bleeds it exactly across the avatar gutter so
 * the text inside stays flush with the name; py-3 puts the text 12px below
 * the header — the same distance ChatAgentContent's mt-3 produces.
 */
export function ChatUserBubble({ text }: { text: string }) {
  return (
    <div className="-mx-4 rounded-xl bg-background-muted px-4 py-3">
      <p className="whitespace-pre-wrap text-[0.9375rem] leading-[1.75] tracking-[-0.011em] text-foreground/85">
        {text}
      </p>
    </div>
  );
}

/** Agent-side content stack: body, disclosures, action rows. */
export function ChatAgentContent({ children }: { children: ReactNode }) {
  return <div className="mt-3 space-y-3">{children}</div>;
}

/** Plain-text agent message body (markdown answers style their own). */
export function ChatAgentText({ text }: { text: string }) {
  return (
    <p className="min-w-0 whitespace-pre-wrap text-[0.9375rem] leading-[1.75] tracking-[-0.011em] text-foreground/85 antialiased [text-wrap:pretty]">
      {text}
    </p>
  );
}
