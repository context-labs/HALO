import type { VariantProps } from "class-variance-authority";
import * as React from "react";
import { cva } from "class-variance-authority";

import { cn } from "~/lib/ui/utils/utils";

const badgeVariants = cva(
  `
    inline-flex items-center justify-center whitespace-nowrap rounded-full border text-xs transition-colors

    focus:outline-hidden focus:ring-2 focus:ring-ring focus:ring-offset-2
  `,
  {
    defaultVariants: {
      variant: "default",
      size: "default",
    },
    variants: {
      variant: {
        default: `
          border-transparent bg-primary text-primary-foreground

          hover:bg-primary/80
        `,
        destructive: `
          border-transparent bg-destructive text-destructive-foreground

          hover:bg-destructive/80
        `,
        failure: `
          border-transparent bg-detail-failure text-detail-failure-foreground

          hover:bg-detail-failure/80
        `,
        outline: "text-foreground",
        secondary: `
          border-transparent bg-secondary text-secondary-foreground

          hover:bg-secondary/80
        `,
        success: "border bg-background font-medium text-detail-success",

        // Semantic status variants (soft tint background, no border)
        "status-pending":
          "border-transparent bg-detail-warning/10 text-detail-warning",
        "status-running":
          "border-transparent bg-detail-brand/10 text-detail-brand",
        "status-success":
          "border-transparent bg-detail-success/10 text-detail-success",
        "status-failure":
          "border-transparent bg-detail-failure/10 text-detail-failure",
        "status-warning":
          "border-transparent bg-detail-warning/10 text-detail-warning",
        "status-brand":
          "border-transparent bg-detail-brand/10 text-detail-brand",
      },
      size: {
        default: "px-3 py-1.5 font-semibold",
        sm: "px-2 py-0.5 font-medium",
      },
    },
  },
);

export type BadgeProps = React.HTMLAttributes<HTMLDivElement> &
  VariantProps<typeof badgeVariants>;

function Badge({ className, variant, size, ...props }: BadgeProps) {
  return (
    <div
      className={cn(badgeVariants({ variant, size }), className)}
      {...props}
    />
  );
}

export { Badge, badgeVariants };
