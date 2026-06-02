import type { LucideIcon } from "lucide-react";
import React from "react";

import { cn } from "~/lib/ui/utils/utils";

type EmptyStateProps = {
  icon: LucideIcon;
  title: string;
  description: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
};

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center rounded-xl border border-dashed border-subtle/60 px-6 py-12 text-center",
        className,
      )}
    >
      <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-full bg-muted/60 ring-1 ring-border/40">
        <Icon className="h-5 w-5 text-muted-foreground/70" />
      </div>
      <p className="text-sm font-medium text-foreground/80">{title}</p>
      <p className="mt-1 max-w-sm text-sm leading-relaxed text-muted-foreground">
        {description}
      </p>
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}
