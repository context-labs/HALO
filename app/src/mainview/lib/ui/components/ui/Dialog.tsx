import * as React from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";

import type { ButtonVariantProp } from "~/lib/ui/components/ui/Button";
import { Button } from "~/lib/ui/components/ui/Button";
import { cn } from "~/lib/ui/utils/utils";

const DialogRoot = DialogPrimitive.Root;

const DialogTrigger = DialogPrimitive.Trigger;

const DialogPortal = DialogPrimitive.Portal;

const DialogClose = DialogPrimitive.Close;

const DialogOverlay = React.forwardRef<
  React.ComponentRef<typeof DialogPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Overlay
    className={cn(
      `
        fixed inset-0 z-50 overflow-hidden bg-black/60 backdrop-blur-sm

        data-[state=closed]:animate-out data-[state=closed]:fade-out-0

        data-[state=open]:animate-in data-[state=open]:fade-in-0
      `,
      className,
    )}
    ref={ref}
    {...props}
  />
));
DialogOverlay.displayName = DialogPrimitive.Overlay.displayName;

const DialogContent = React.forwardRef<
  React.ComponentRef<typeof DialogPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content> & {
    forceFocus?: boolean;
  }
>(({ children, className, forceFocus, ...props }, ref) => (
  <DialogPortal>
    <DialogOverlay
      className={cn(`
        fixed inset-0 flex flex-col items-center

        sm:overflow-hidden
      `)}
    >
      <div
        className={cn(`
          flex h-full w-full flex-col

          sm:h-auto sm:min-h-fit sm:w-auto sm:justify-center
        `)}
      >
        <DialogPrimitive.Content
          className={cn(
            `
              relative z-50 flex h-full max-h-screen w-full flex-col border
              bg-background shadow-lg duration-200

              data-[state=closed]:animate-out data-[state=closed]:fade-out-0
              data-[state=closed]:zoom-out-95
              data-[state=closed]:slide-out-to-top-[5%]

              data-[state=open]:animate-in data-[state=open]:fade-in-0
              data-[state=open]:zoom-in-95
              data-[state=open]:slide-in-from-top-[5%]

              sm:mb-48 sm:mt-24 sm:h-auto sm:max-h-[calc(100vh-12rem)]
              sm:max-w-lg sm:overflow-visible sm:rounded-xl
            `,
            className,
          )}
          onEscapeKeyDown={
            forceFocus
              ? (e) => {
                  e.preventDefault();
                }
              : undefined
          }
          onInteractOutside={forceFocus ? (e) => e.preventDefault() : undefined}
          onPointerDownOutside={
            forceFocus ? (e) => e.preventDefault() : undefined
          }
          ref={ref}
          {...props}
        >
          {children}
          <DialogPrimitive.Close
            className={`
              absolute right-4 top-4 rounded-md opacity-70
              ring-offset-background transition-opacity

              data-[state=open]:bg-accent
              data-[state=open]:text-muted-foreground

              disabled:pointer-events-none

              focus-visible:outline-hidden focus-visible:ring-2
              focus-visible:ring-ring focus-visible:ring-offset-2

              hover:opacity-100
            `}
          >
            <X className="h-6 w-6" />
            <span className="sr-only">Close</span>
          </DialogPrimitive.Close>
        </DialogPrimitive.Content>
      </div>
    </DialogOverlay>
  </DialogPortal>
));
DialogContent.displayName = DialogPrimitive.Content.displayName;

const DialogHeader = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={cn(
      `
        flex flex-col space-y-2 rounded-t-xl border-b bg-background p-6
        text-center

        sm:text-left
      `,
      className,
    )}
    {...props}
  />
);
DialogHeader.displayName = "DialogHeader";

const DialogFooter = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={cn(`flex flex-row space-x-2 border-t p-6`, className)}
    {...props}
  />
);
DialogFooter.displayName = "DialogFooter";

const DialogTitle = React.forwardRef<
  React.ComponentRef<typeof DialogPrimitive.Title>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Title>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Title
    className={cn(
      "text-lg font-semibold leading-none tracking-tight",
      className,
    )}
    ref={ref}
    {...props}
  />
));
DialogTitle.displayName = DialogPrimitive.Title.displayName;

const DialogDescription = React.forwardRef<
  React.ComponentRef<typeof DialogPrimitive.Description>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Description>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Description
    className={cn("text-base text-muted-foreground", className)}
    ref={ref}
    {...props}
  />
));
DialogDescription.displayName = DialogPrimitive.Description.displayName;

type DialogProps = {
  children?: React.ReactNode;
  confirmButtonVariant?: ButtonVariantProp;
  confirmTitle?: string;
  cancelTitle?: string;
  dialogDescription?: React.ReactNode;
  dialogTitle?: React.ReactNode;
  disabled?: boolean;
  leftButton?: React.ReactNode | string;
  loading?: boolean;
  maxWidth?: number;
  modal?: boolean;
  onConfirm: () => void;
  onOpenChange: (open: boolean) => void;
  open: boolean;
  trigger?: React.ReactNode | string;
  footer?: React.ReactNode | string;
  hideCloseButton?: boolean;
  hideConfirmButton?: boolean;
  className?: string;
  confirmButton?: React.ReactNode;
  forceFocus?: boolean;
};

function Dialog(props: DialogProps) {
  const {
    cancelTitle = "Cancel",
    children,
    className,
    confirmButton,
    confirmButtonVariant = "default",
    confirmTitle = "Save Changes",
    dialogDescription,
    dialogTitle = "Are you sure?",
    disabled = false,
    footer,
    forceFocus = false,
    hideCloseButton = false,
    hideConfirmButton = false,
    leftButton = undefined,
    loading = false,
    maxWidth,
    modal = true,
    onConfirm,
    onOpenChange,
    open,
    trigger,
  } = props;

  const shouldShowLeftButton = leftButton != null || !hideCloseButton;
  const shouldShowConfirmButton = confirmButton != null || !hideConfirmButton;
  const shouldShowFooter = shouldShowLeftButton || shouldShowConfirmButton;

  return (
    <DialogRoot
      modal={modal}
      onOpenChange={
        forceFocus
          ? (open) => {
              // If forceFocus is true, only allow explicit open changes, not closing
              if (open === true) {
                onOpenChange(open);
              }
            }
          : onOpenChange
      }
      open={open}
    >
      {trigger && <DialogTrigger asChild>{trigger}</DialogTrigger>}
      <DialogContent
        className={cn(
          `
            max-w-[425px]

            md:w-[700px]
          `,
          className,
        )}
        forceFocus={forceFocus}
        onCloseAutoFocus={(e) => e.preventDefault()}
        onOpenAutoFocus={(e) => e.preventDefault()}
        style={{ maxWidth }}
      >
        <DialogHeader>
          <DialogTitle>{dialogTitle}</DialogTitle>
          {dialogDescription && (
            <DialogDescription>{dialogDescription}</DialogDescription>
          )}
        </DialogHeader>
        {children != null && (
          <div className={`min-h-0 flex-1 overflow-y-auto px-6 pb-6 pt-6`}>
            {children}
          </div>
        )}
        {footer != null && footer}
        {footer == null && shouldShowFooter && (
          <DialogFooter className="flex items-end justify-end gap-2">
            {/* eslint-disable-next-line @typescript-eslint/prefer-nullish-coalescing */}
            {leftButton != null ? (
              leftButton
            ) : hideCloseButton ? (
              <div />
            ) : (
              <Button
                disabled={disabled || loading}
                onClick={() => onOpenChange(false)}
                variant="secondary"
              >
                {cancelTitle}
              </Button>
            )}
            {/* eslint-disable-next-line @typescript-eslint/prefer-nullish-coalescing */}
            {confirmButton != null ? (
              confirmButton
            ) : hideConfirmButton ? (
              <div />
            ) : (
              <Button
                className="w-fit"
                disabled={disabled || loading}
                onClick={onConfirm}
                variant={confirmButtonVariant}
              >
                {loading ? "Loading..." : confirmTitle}
              </Button>
            )}
          </DialogFooter>
        )}
      </DialogContent>
    </DialogRoot>
  );
}

export {
  Dialog,
  DialogRoot,
  DialogPortal,
  DialogOverlay,
  DialogClose,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogFooter,
  DialogTitle,
  DialogDescription,
};
