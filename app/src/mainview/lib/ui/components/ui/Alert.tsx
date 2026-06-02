import type { VariantProps } from "class-variance-authority";
import * as React from "react";
import { cva } from "class-variance-authority";

import { cn } from "~/lib/ui/utils/utils";

const alertVariants = cva(
  `
    relative w-full rounded-lg border bg-card p-4

    [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4 [&>svg]:text-foreground

    [&>svg+div]:translate-y-[-3px]

    [&>svg~*]:pl-7
  `,
  {
    defaultVariants: {
      variant: "default",
    },
    variants: {
      variant: {
        default: "text-foreground",
        destructive: cn(`
          border-detail-failure/50 text-detail-failure

          [&>svg]:text-detail-failure

          dark:border-detail-failure
        `),
        info: cn(`
          border-detail-brand text-foreground

          dark:border-detail-brand
        `),
        warning: cn(`
          border-detail-warning text-detail-warning

          dark:border-detail-warning
        `),
      },
    },
  },
);

type AlertVariantProp = VariantProps<typeof alertVariants>["variant"];

const AlertBase = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof alertVariants>
>(({ className, variant, ...props }, ref) => (
  <div
    className={cn(alertVariants({ variant }), className)}
    ref={ref}
    role="alert"
    {...props}
  />
));
AlertBase.displayName = "Alert";

const AlertTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h5
    className={cn("mb-1 font-medium leading-none tracking-tight", className)}
    ref={ref}
    {...props}
  />
));
AlertTitle.displayName = "AlertTitle";

const AlertDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <div
    className={cn(
      `
        text-sm

        [&_p]:leading-relaxed
      `,
      className,
    )}
    ref={ref}
    {...props}
  />
));
AlertDescription.displayName = "AlertDescription";

type AlertProps = React.ComponentPropsWithoutRef<typeof AlertBase> & {
  title?: React.ReactNode;
  description?: React.ReactNode;
  icon?: React.ReactNode;
  variant?: AlertVariantProp;
};

const Alert = React.forwardRef<HTMLDivElement, AlertProps>(
  ({ children, description, icon, title, variant, ...props }, ref) => {
    return (
      <AlertBase ref={ref} {...props} variant={variant}>
        {icon}
        {title && <AlertTitle>{title}</AlertTitle>}
        {description && <AlertDescription>{description}</AlertDescription>}
        {children}
      </AlertBase>
    );
  },
);
Alert.displayName = "Alert";

export { Alert, AlertTitle, AlertDescription };
