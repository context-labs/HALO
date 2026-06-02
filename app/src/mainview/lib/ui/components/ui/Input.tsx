import * as React from "react";

import { cn } from "~/lib/ui/utils/utils";

export type InputProps = React.InputHTMLAttributes<HTMLInputElement> & {
  error?: string | null;
  hasError?: boolean;
  hint?: string;
  label?: string;
  icon?: React.ReactNode;
  containerClassname?: string;
};

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  (
    {
      className,
      containerClassname,
      error,
      hasError,
      hint,
      icon,
      label,
      type,
      ...props
    },
    ref,
  ) => {
    return (
      <div className={cn("w-full", containerClassname)}>
        {label && (
          <label className="text-sm font-medium leading-none">{label}</label>
        )}
        <div className="relative">
          {icon && (
            <div className="absolute left-2 top-1/2 -translate-y-1/2">
              {icon}
            </div>
          )}
          <input
            className={cn(
              `
                flex h-10 w-full rounded-md border border-subtle bg-background px-3
                py-2 text-xs

                disabled:cursor-not-allowed disabled:opacity-50

                file:border-0 file:bg-transparent file:text-sm file:font-medium

                focus-visible:outline-hidden focus-visible:border-foreground/40

                placeholder:text-xs
              `,
              (error ?? hasError) && "border-detail-failure",
              icon && "pl-7",
              label && "mt-[4px]",
              className,
            )}
            ref={ref}
            type={type}
            autoComplete={type === "password" ? "current-password" : "off"}
            {...props}
          />
        </div>
        {error && <p className="mt-2 text-sm text-detail-failure">{error}</p>}
        {hint && <p className="mt-1 text-sm text-muted-foreground">{hint}</p>}
      </div>
    );
  },
);
Input.displayName = "Input";

export { Input };
