import type { VariantProps } from "class-variance-authority";
import * as React from "react";
import * as SheetPrimitive from "@radix-ui/react-dialog";
import { cva } from "class-variance-authority";
import { X } from "lucide-react";

import { cn } from "~/lib/ui/utils/utils";

const Sheet = SheetPrimitive.Root;

const SheetTrigger = SheetPrimitive.Trigger;

const SheetClose = SheetPrimitive.Close;

const SheetPortal = SheetPrimitive.Portal;

const SheetOverlay = React.forwardRef<
  React.ComponentRef<typeof SheetPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof SheetPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <SheetPrimitive.Overlay
    className={cn(
      `
        fixed inset-0 z-50 bg-black/60 backdrop-blur-sm

        data-[state=closed]:animate-out data-[state=closed]:fade-out-0

        data-[state=open]:animate-in data-[state=open]:fade-in-0
      `,
      className,
    )}
    {...props}
    ref={ref}
  />
));
SheetOverlay.displayName = SheetPrimitive.Overlay.displayName;

const sheetVariants = cva(
  `
    fixed z-50 gap-4 bg-background p-6 shadow-lg outline-hidden transition
    ease-in-out border-subtle

    data-[state=closed]:duration-150 data-[state=closed]:animate-out

    data-[state=open]:duration-150 data-[state=open]:animate-in
  `,
  {
    defaultVariants: {
      side: "right",
    },
    variants: {
      side: {
        bottom: cn(`
          inset-x-0 bottom-0 border-t

          data-[state=closed]:slide-out-to-bottom

          data-[state=open]:slide-in-from-bottom
        `),
        left: cn(`
          inset-y-0 left-0 h-full w-3/4 border-r

          data-[state=closed]:slide-out-to-left

          data-[state=open]:slide-in-from-left

          sm:max-w-sm
        `),
        right: cn(`
          inset-y-0 right-0 h-full w-3/4 border-l

          data-[state=closed]:slide-out-to-right

          data-[state=open]:slide-in-from-right

          sm:max-w-sm
        `),
        top: cn(`
          inset-x-0 top-0 border-b

          data-[state=closed]:slide-out-to-top

          data-[state=open]:slide-in-from-top
        `),
      },
    },
  },
);

type SheetContentProps = React.ComponentPropsWithoutRef<
  typeof SheetPrimitive.Content
> & {
  closeButtonAriaLabel?: string;
  showCloseButton?: boolean;
} & VariantProps<typeof sheetVariants>;

const SheetContent = React.forwardRef<
  React.ComponentRef<typeof SheetPrimitive.Content>,
  SheetContentProps
>(
  (
    {
      children,
      className,
      closeButtonAriaLabel = "Close Side Panel",
      showCloseButton = true,
      side = "right",
      ...props
    },
    ref,
  ) => (
    <SheetPortal>
      <SheetOverlay />
      <SheetPrimitive.Content
        onOpenAutoFocus={(e) => e.preventDefault()}
        className={cn(sheetVariants({ side }), className)}
        ref={ref}
        {...props}
      >
        {children}
        {showCloseButton ? (
          <SheetPrimitive.Close
            aria-label={closeButtonAriaLabel}
            className={`
              absolute right-3 top-3 z-10 h-8 w-8 rounded-md
              bg-secondary opacity-70
              ring-offset-background transition-opacity

              data-[state=open]:bg-secondary

              disabled:pointer-events-none

              focus-visible:outline-hidden focus-visible:ring-2
              focus-visible:ring-ring focus-visible:ring-offset-2

              hover:opacity-100
            `}
          >
            <X className="m-auto h-4 w-4" />
            <span className="sr-only">{closeButtonAriaLabel}</span>
          </SheetPrimitive.Close>
        ) : null}
      </SheetPrimitive.Content>
    </SheetPortal>
  ),
);
SheetContent.displayName = SheetPrimitive.Content.displayName;

const SheetHeader = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={cn(
      `
        flex flex-col space-y-2 text-center

        sm:text-left
      `,
      className,
    )}
    {...props}
  />
);
SheetHeader.displayName = "SheetHeader";

const SheetFooter = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={cn(
      `
        flex flex-col-reverse

        sm:flex-row sm:justify-end sm:space-x-2
      `,
      className,
    )}
    {...props}
  />
);
SheetFooter.displayName = "SheetFooter";

const SheetTitle = React.forwardRef<
  React.ComponentRef<typeof SheetPrimitive.Title>,
  React.ComponentPropsWithoutRef<typeof SheetPrimitive.Title>
>(({ className, ...props }, ref) => (
  <SheetPrimitive.Title className={cn(className)} ref={ref} {...props} />
));
SheetTitle.displayName = SheetPrimitive.Title.displayName;

const SheetDescription = React.forwardRef<
  React.ComponentRef<typeof SheetPrimitive.Description>,
  React.ComponentPropsWithoutRef<typeof SheetPrimitive.Description>
>(({ className, ...props }, ref) => (
  <SheetPrimitive.Description
    className={cn("text-sm text-muted-foreground", className)}
    ref={ref}
    {...props}
  />
));
SheetDescription.displayName = SheetPrimitive.Description.displayName;

export {
  Sheet,
  SheetPortal,
  SheetOverlay,
  SheetTrigger,
  SheetClose,
  SheetContent,
  SheetHeader,
  SheetFooter,
  SheetTitle,
  SheetDescription,
};
