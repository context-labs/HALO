import * as React from "react";
import * as TabsPrimitive from "@radix-ui/react-tabs";

import { cn } from "~/lib/ui/utils/utils";

const Tabs = TabsPrimitive.Root;

const TabsList = React.forwardRef<
  React.ComponentRef<typeof TabsPrimitive.List>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.List
    className={cn(
      `
        flex h-auto flex-wrap items-center justify-start gap-2 rounded-lg p-1
        text-muted-foreground

        sm:inline-flex sm:h-auto sm:justify-center
      `,
      className,
    )}
    ref={ref}
    {...props}
  />
));
TabsList.displayName = TabsPrimitive.List.displayName;

const TabsTrigger = React.forwardRef<
  React.ComponentRef<typeof TabsPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Trigger
    className={cn(
      `
        inline-flex items-center justify-center whitespace-nowrap rounded-md
        bg-secondary/20 px-4 py-1.5 text-sm font-medium ring-offset-background
        transition-all

        data-[state=active]:text-foreground data-[state=active]:shadow-xs

        disabled:pointer-events-none disabled:opacity-50

        focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring
        focus-visible:ring-offset-2

        hover:bg-secondary/80 hover:text-foreground
      `,
      `
        w-full

        sm:w-auto
      `,
      className,
    )}
    ref={ref}
    {...props}
  />
));
TabsTrigger.displayName = TabsPrimitive.Trigger.displayName;

const TabsContent = React.forwardRef<
  React.ComponentRef<typeof TabsPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Content
    className={cn(
      `
        mt-[2px] ring-offset-background

        focus-visible:outline-hidden focus-visible:ring-0
      `,
      className,
    )}
    ref={ref}
    {...props}
  />
));
TabsContent.displayName = TabsPrimitive.Content.displayName;

export { Tabs, TabsList, TabsTrigger, TabsContent };
