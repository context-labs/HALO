import { useEffect, useRef, useState } from "react";
import { Check, Copy } from "lucide-react";

import { cn } from "~/lib/ui";

/**
 * Copy affordance pinned to a content block's top-right corner (the parent
 * needs `relative`). Always copies the full payload, even when the visible
 * text is clamped.
 */
export function CopyTextButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const resetTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (resetTimer.current) clearTimeout(resetTimer.current);
    },
    [],
  );
  return (
    <button
      aria-label="Copy to clipboard"
      className={cn(
        "absolute right-2 top-2 grid h-7 w-7 place-items-center rounded-md border border-subtle bg-background-muted text-muted-foreground shadow-sm transition-colors hover:text-foreground",
        copied && "text-detail-success hover:text-detail-success",
      )}
      onClick={() => {
        void navigator.clipboard.writeText(text).then(() => {
          setCopied(true);
          if (resetTimer.current) clearTimeout(resetTimer.current);
          resetTimer.current = setTimeout(() => setCopied(false), 1_500);
        });
      }}
      title="Copy to clipboard"
      type="button"
    >
      {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
    </button>
  );
}
