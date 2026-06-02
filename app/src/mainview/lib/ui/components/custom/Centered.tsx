import React from "react";

import { cn } from "~/lib/ui/utils/utils";

type CenteredProps = React.HtmlHTMLAttributes<HTMLDivElement> & {
  children: React.ReactNode;
};

export const Centered = React.forwardRef<HTMLDivElement, CenteredProps>(
  ({ children, className, ...rest }, ref) => {
    return (
      <div
        className={cn("flex items-center justify-center", className)}
        ref={ref}
        {...rest}
      >
        {children}
      </div>
    );
  },
);

Centered.displayName = "Centered";
