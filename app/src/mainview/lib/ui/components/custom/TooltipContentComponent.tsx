import type React from "react";

import { Col } from "~/lib/ui/components/custom/Col";
import { cn } from "~/lib/ui/utils/utils";

type TooltipContentComponentProps = {
  content: React.ReactNode | string[];
  title: React.ReactNode;
  className?: string;
};

export function TooltipContentComponent({
  className,
  content,
  title,
}: TooltipContentComponentProps) {
  return (
    <Col className={cn("w-[400px] gap-2 pb-1", className)}>
      <p className="font-semibold">{title}</p>
      {Array.isArray(content) ? (
        content.map((item, index) => (
          <p className="text-sm text-muted-foreground" key={index}>
            {item}
          </p>
        ))
      ) : typeof content === "string" ? (
        <p className="text-sm text-muted-foreground">{content}</p>
      ) : (
        content
      )}
    </Col>
  );
}
