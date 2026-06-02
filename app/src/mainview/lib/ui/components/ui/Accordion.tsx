import * as React from "react";
import * as AccordionPrimitive from "@radix-ui/react-accordion";
import { ChevronDown } from "lucide-react";

import { cn } from "~/lib/ui/utils/utils";

const Accordion = AccordionPrimitive.Root;

const AccordionItem = React.forwardRef<
  React.ComponentRef<typeof AccordionPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Item>
>(({ className, ...props }, ref) => (
  <AccordionPrimitive.Item
    className={cn("border-b", className)}
    ref={ref}
    {...props}
  />
));
AccordionItem.displayName = "AccordionItem";

const AccordionTrigger = React.forwardRef<
  React.ComponentRef<typeof AccordionPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Trigger> & {
    enableFocusStyle?: boolean;
  }
>(({ children, className, enableFocusStyle = false, ...props }, ref) => (
  <AccordionPrimitive.Header className="flex">
    <AccordionPrimitive.Trigger
      aria-label={props["aria-label"] ?? "Accordion Trigger"}
      className={cn(
        `
          flex flex-1 items-center py-4 font-medium transition-all

          [&[data-state=open]>svg]:rotate-180
        `,
        className,
        !enableFocusStyle &&
          `
            focus-visible:ring-0 focus-visible:ring-transparent

            focus:outline-hidden focus:ring-0 focus:ring-transparent
            focus:ring-offset-0
          `,
      )}
      ref={ref}
      {...props}
    >
      {children}
      <ChevronDown
        className={cn(
          `ml-2 h-5 w-5 shrink-0 transition-transform duration-200`,
        )}
      />
    </AccordionPrimitive.Trigger>
  </AccordionPrimitive.Header>
));
AccordionTrigger.displayName = AccordionPrimitive.Trigger.displayName;

const AccordionContent = React.forwardRef<
  React.ComponentRef<typeof AccordionPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Content>
>(({ children, className, ...props }, ref) => (
  <AccordionPrimitive.Content
    className={`
      overflow-hidden text-sm transition-all

      data-[state=closed]:animate-accordion-up

      data-[state=open]:animate-accordion-down
    `}
    ref={ref}
    {...props}
  >
    <div className={cn("pb-4 pt-0", className)}>{children}</div>
  </AccordionPrimitive.Content>
));

AccordionContent.displayName = AccordionPrimitive.Content.displayName;

export { Accordion, AccordionItem, AccordionTrigger, AccordionContent };
