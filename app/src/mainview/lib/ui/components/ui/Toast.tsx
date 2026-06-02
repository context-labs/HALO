import type { VariantProps } from "class-variance-authority";
import * as React from "react";
import * as ToastPrimitives from "@radix-ui/react-toast";
import { cva } from "class-variance-authority";
import { X } from "lucide-react";

import { cn } from "~/lib/ui/utils/utils";

const ToastProvider = ToastPrimitives.Provider;

const ToastViewport = React.forwardRef<
  React.ComponentRef<typeof ToastPrimitives.Viewport>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Viewport>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Viewport
    className={cn(
      `
        pointer-events-none fixed right-0 bottom-0 z-100 flex max-h-screen
        w-full flex-col p-4

        md:w-fit md:min-w-[420px] md:max-w-[calc(100vw-2rem)]
      `,
      className,
    )}
    ref={ref}
    {...props}
  />
));
ToastViewport.displayName = ToastPrimitives.Viewport.displayName;

const toastVariants = cva(
  `
    group pointer-events-auto relative flex w-full items-center justify-between
    space-x-4 overflow-hidden rounded-xl border p-6 pr-8 shadow-lg transition-all

    data-[state=closed]:animate-out data-[state=closed]:fade-out-80
    data-[state=closed]:slide-out-to-right-full

    data-[state=open]:animate-in data-[state=open]:slide-in-from-bottom-full

    data-[swipe=cancel]:translate-x-0

    data-[swipe=end]:translate-x-(--radix-toast-swipe-end-x)
    data-[swipe=end]:animate-out

    data-[swipe=move]:translate-x-(--radix-toast-swipe-move-x)
    data-[swipe=move]:transition-none
  `,
  {
    defaultVariants: {
      variant: "info",
    },
    variants: {
      variant: {
        destructive:
          "group border border-l-4 border-l-detail-failure bg-background text-foreground",
        info: "border bg-background text-foreground",
        success:
          "border border-l-4 border-l-detail-success bg-background text-foreground",
        warning:
          "border border-l-4 border-l-detail-warning bg-background text-foreground",
      },
    },
  },
);

const Toast = React.forwardRef<
  React.ComponentRef<typeof ToastPrimitives.Root>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Root> &
    VariantProps<typeof toastVariants>
>(({ className, variant, ...props }, ref) => {
  return (
    <ToastPrimitives.Root
      className={cn(toastVariants({ variant }), className)}
      ref={ref}
      {...props}
    />
  );
});
Toast.displayName = ToastPrimitives.Root.displayName;

const ToastAction = React.forwardRef<
  React.ComponentRef<typeof ToastPrimitives.Action>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Action>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Action
    className={cn(
      `
        inline-flex h-8 shrink-0 items-center justify-center rounded-md border
        bg-transparent px-3 text-sm font-medium ring-offset-background
        transition-colors

        disabled:pointer-events-none disabled:opacity-50

        focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring
        focus-visible:ring-offset-2

        group-[.destructive]:border-muted/40
        hover:group-[.destructive]:border-destructive/30
        hover:group-[.destructive]:bg-destructive
        hover:group-[.destructive]:text-destructive-foreground
        focus:group-[.destructive]:ring-destructive
      `,
      className,
    )}
    ref={ref}
    {...props}
  />
));
ToastAction.displayName = ToastPrimitives.Action.displayName;

const ToastClose = React.forwardRef<
  React.ComponentRef<typeof ToastPrimitives.Close>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Close>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Close
    className={cn(
      `
        absolute right-2 top-2 rounded-md p-1 opacity-70
        transition-opacity

        focus-visible:opacity-100 focus-visible:outline-hidden
        focus-visible:ring-2

        group-hover:opacity-100

        hover:opacity-100
      `,
      className,
      "text-muted-foreground hover:text-foreground",
    )}
    ref={ref}
    toast-close=""
    {...props}
  >
    <X className="h-4 w-4" />
  </ToastPrimitives.Close>
));
ToastClose.displayName = ToastPrimitives.Close.displayName;

const ToastTitle = React.forwardRef<
  React.ComponentRef<typeof ToastPrimitives.Title>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Title>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Title
    className={cn("text-sm font-semibold", className)}
    ref={ref}
    {...props}
  />
));
ToastTitle.displayName = ToastPrimitives.Title.displayName;

const ToastDescription = React.forwardRef<
  React.ComponentRef<typeof ToastPrimitives.Description>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Description>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Description
    className={cn("text-sm opacity-90", className)}
    ref={ref}
    {...props}
  />
));
ToastDescription.displayName = ToastPrimitives.Description.displayName;

type ToastProps = React.ComponentPropsWithoutRef<typeof Toast>;

type ToastActionElement = React.ReactElement<typeof ToastAction>;

export {
  type ToastProps,
  type ToastActionElement,
  ToastProvider,
  ToastViewport,
  Toast,
  ToastTitle,
  ToastDescription,
  ToastClose,
  ToastAction,
};
