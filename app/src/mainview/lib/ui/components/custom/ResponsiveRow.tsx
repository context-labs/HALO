import React from "react";

import { Col } from "~/lib/ui/components/custom/Col";
import { cn } from "~/lib/ui/utils/utils";

type ResponsiveRowProps = React.HtmlHTMLAttributes<HTMLDivElement> & {
  children: React.ReactNode;
};

export const ResponsiveRow = React.forwardRef<
  HTMLDivElement,
  ResponsiveRowProps
>(({ children, className, ...rest }, ref) => {
  return (
    <Col
      className={cn(
        `
          flex items-start

          md:flex-row md:items-center
        `,
        className,
      )}
      ref={ref}
      {...rest}
    >
      {children}
    </Col>
  );
});
