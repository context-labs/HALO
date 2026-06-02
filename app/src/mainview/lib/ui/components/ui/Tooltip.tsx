import * as React from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";

import { cn } from "~/lib/ui/utils/utils";

const TooltipProvider = TooltipPrimitive.Provider;

const TooltipRoot = TooltipPrimitive.Root;

const TooltipTrigger = TooltipPrimitive.Trigger;

const TooltipContent = React.forwardRef<
  React.ComponentRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <TooltipPrimitive.Content
    className={cn(
      `
        z-50 max-w-64 overflow-hidden rounded-lg border bg-popover px-3 py-1.5
        text-sm text-popover-foreground shadow-md animate-in fade-in-0
        zoom-in-95

        data-[side=bottom]:slide-in-from-top-2

        data-[side=left]:slide-in-from-right-2

        data-[side=right]:slide-in-from-left-2

        data-[side=top]:slide-in-from-bottom-2

        data-[state=closed]:animate-out data-[state=closed]:fade-out-0
        data-[state=closed]:zoom-out-95
      `,
      className,
    )}
    ref={ref}
    sideOffset={sideOffset}
    {...props}
  />
));
TooltipContent.displayName = TooltipPrimitive.Content.displayName;

type TooltipProps = {
  children: React.ReactNode;
  content: React.ReactNode;
  disabled?: boolean;
  side?: TooltipPrimitive.TooltipContentProps["side"];
  contentClassName?: string;
};

function Tooltip({
  children,
  content,
  disabled = false,
  side,
  contentClassName,
}: TooltipProps) {
  return (
    <TooltipProvider delayDuration={100}>
      <TooltipRoot>
        {!disabled && (
          <TooltipPrimitive.Portal>
            <TooltipContent side={side} className={contentClassName}>
              {content}
            </TooltipContent>
          </TooltipPrimitive.Portal>
        )}
        <TooltipTrigger asChild>{children}</TooltipTrigger>
      </TooltipRoot>
    </TooltipProvider>
  );
}

export { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider };
