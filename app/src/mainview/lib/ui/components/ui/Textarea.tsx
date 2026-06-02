import * as React from "react";

import { cn } from "~/lib/ui/utils/utils";

export type TextareaProps = React.ComponentProps<"textarea"> & {
  error?: string | null;
  hasError?: boolean;
  hint?: string;
  label?: string;
};

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, error, hasError, hint, label, ...props }, ref) => {
    return (
      <div className="w-full">
        {label && (
          <label className="text-sm font-medium leading-none">{label}</label>
        )}
        <textarea
          className={cn(
            `
              flex min-h-[80px] w-full rounded-md border border-input
              bg-background px-3 py-2 text-base

              disabled:cursor-not-allowed disabled:opacity-50

              focus-visible:outline-hidden

              md:text-sm

              placeholder:text-muted-foreground
            `,
            (error ?? hasError) && "border-detail-failure",
            label && "mt-[4px]",
            className,
          )}
          ref={ref}
          {...props}
        />
        {error && <p className="mt-2 text-sm text-detail-failure">{error}</p>}
        {hint && <p className="mt-1 text-sm text-muted-foreground">{hint}</p>}
      </div>
    );
  },
);
Textarea.displayName = "Textarea";

export { Textarea };
