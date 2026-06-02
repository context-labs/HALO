import type { VariantProps } from "class-variance-authority";
import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva } from "class-variance-authority";

import { cn } from "~/lib/ui/utils/utils";

const buttonVariants = cva(
  `
    inline-flex cursor-pointer items-center justify-center whitespace-nowrap
    rounded-md text-sm font-medium ring-offset-background transition-all
    duration-200

    disabled:cursor-not-allowed disabled:opacity-50

    focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring
    focus-visible:ring-offset-2
  `,
  {
    defaultVariants: {
      size: "default",
      variant: "default",
    },
    variants: {
      size: {
        default: "h-10 px-4 py-2",
        icon: "h-9 w-9",
        "2xl": "h-16 px-8 text-lg",
        xl: "h-14 px-6 text-base",
        lg: "h-11 px-4 text-base",
        sm: "h-8 px-3 text-xs",
        xs: "h-7 px-2 py-1 text-xs",
      },
      variant: {
        default: cn(`
          border border-primary-border bg-primary text-primary-foreground

          dark:border-primary-cta-border dark:bg-primary-cta
          dark:text-primary-cta-foreground dark:hover:bg-primary-cta-hover

          hover:bg-primary-hover
        `),
        primary: cn(`
          border border-primary-border bg-primary text-primary-foreground

          dark:border-primary-cta-border dark:bg-primary-cta
          dark:text-primary-cta-foreground dark:hover:bg-primary-cta-hover

          hover:bg-primary-hover
        `),
        destructive: cn(`
          border border-destructive-border bg-destructive
          text-destructive-foreground

          hover:bg-destructive-hover
        `),
        ghost: cn(`
          hover:bg-muted/90 hover:text-foreground
        `),
        link: cn(`
          text-md font-normal text-link underline-offset-4

          hover:text-link-hover hover:underline
        `),
        outline: cn(`
          border border-subtle bg-background-muted

          hover:bg-accent
        `),
        plainLink: cn(`
          text-foreground underline-offset-4

          hover:text-foreground hover:underline
        `),
        secondary: cn(`
          border border-secondary-border bg-secondary text-secondary-foreground

          hover:bg-secondary-hover
        `),
        tertiary: cn(`
          bg-tertiary border border-tertiary-border text-tertiary-foreground

          hover:bg-tertiary-hover
        `),
      },
    },
  },
);

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean;
    enableFocusStyle?: boolean;
    loading?: boolean;
  };

export type ButtonVariantProp = VariantProps<typeof buttonVariants>["variant"];

export type ButtonSizeProp = VariantProps<typeof buttonVariants>["size"];

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      asChild = false,
      className,
      disabled,
      enableFocusStyle = true,
      loading,
      size,
      variant,
      ...props
    },
    ref,
  ) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(
          buttonVariants({ className, size, variant }),
          !enableFocusStyle &&
            `
              focus-visible:ring-0 focus-visible:ring-transparent

              focus:outline-hidden focus:ring-0 focus:ring-transparent
              focus:ring-offset-0
            `,
        )}
        disabled={disabled ?? loading}
        ref={ref}
        {...props}
      />
    );
  },
);
Button.displayName = "Button";

export { Button, buttonVariants };
