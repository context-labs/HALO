import React from "react";

import { cn } from "~/lib/ui/utils/utils";

type GridProps = React.HtmlHTMLAttributes<HTMLDivElement> & {
  children: React.ReactNode;
};

export const Grid = React.forwardRef<HTMLDivElement, GridProps>(
  ({ children, className, ...rest }, ref) => {
    return (
      <div className={cn("grid", className)} ref={ref} {...rest}>
        {children}
      </div>
    );
  },
);

Grid.displayName = "Grid";
