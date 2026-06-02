import * as React from "react";
import * as SwitchPrimitives from "@radix-ui/react-switch";

import { cn } from "~/lib/ui/utils/utils";

const SwitchBase = React.forwardRef<
  React.ComponentRef<typeof SwitchPrimitives.Root>,
  React.ComponentPropsWithoutRef<typeof SwitchPrimitives.Root>
>(({ className, ...props }, ref) => (
  <SwitchPrimitives.Root
    className={cn(
      `
        peer inline-flex h-6 w-11 shrink-0 cursor-pointer items-center
        rounded-full border-2 border-transparent transition-colors

        data-[state=checked]:bg-ring

        data-[state=unchecked]:bg-switch

        disabled:cursor-not-allowed disabled:opacity-50

        focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring
        focus-visible:ring-offset-2 focus-visible:ring-offset-background
      `,
      className,
    )}
    {...props}
    ref={ref}
  >
    <SwitchPrimitives.Thumb
      className={cn(
        `
          pointer-events-none block h-5 w-5 rounded-full bg-background shadow-lg
          ring-0 transition-transform

          data-[state=checked]:translate-x-5

          data-[state=unchecked]:translate-x-0
        `,
      )}
    />
  </SwitchPrimitives.Root>
));
SwitchBase.displayName = SwitchPrimitives.Root.displayName;

type SwitchProps = React.ComponentPropsWithoutRef<typeof SwitchBase> & {
  label?: React.ReactNode;
  labelClassName?: string;
};

const Switch = React.forwardRef<
  React.ComponentRef<typeof SwitchBase>,
  SwitchProps
>(({ className, id, label, labelClassName, ...props }, ref) => {
  return (
    <label
      className={cn("flex cursor-pointer items-center space-x-2", className)}
      htmlFor={id}
    >
      <SwitchBase id={id} ref={ref} {...props} />
      {label && (
        <span
          className={cn(
            `
              font-medium leading-none

              peer-disabled:cursor-not-allowed peer-disabled:opacity-70
            `,
            labelClassName,
          )}
        >
          {label}
        </span>
      )}
    </label>
  );
});
Switch.displayName = "Switch";

export { Switch, SwitchBase };
