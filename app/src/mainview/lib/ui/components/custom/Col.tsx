import React from "react";

import { cn } from "~/lib/ui/utils/utils";

type ColProps = React.HtmlHTMLAttributes<HTMLDivElement> & {
  children: React.ReactNode;
};

export const Col = React.forwardRef<HTMLDivElement, ColProps>(
  ({ children, className, ...rest }, ref) => {
    return (
      <div className={cn("flex flex-col", className)} ref={ref} {...rest}>
        {children}
      </div>
    );
  },
);

Col.displayName = "Col";
