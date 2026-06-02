import * as React from "react";
import * as SliderPrimitive from "@radix-ui/react-slider";

import { cn } from "~/lib/ui/utils/utils";

const Slider = React.forwardRef<
  React.ComponentRef<typeof SliderPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>
>(({ className, ...props }, ref) => (
  <SliderPrimitive.Root
    className={cn(
      "relative flex w-full touch-none select-none items-center",
      className,
    )}
    ref={ref}
    {...props}
  >
    <SliderPrimitive.Track
      className={cn(`
        relative h-1.5 w-full grow overflow-hidden rounded-full border
        bg-secondary
      `)}
    >
      <SliderPrimitive.Range className="absolute h-full bg-ring" />
    </SliderPrimitive.Track>
    <SliderPrimitive.Thumb
      aria-label={props["aria-label"] ?? "Slider Thumb"}
      className={`
        block h-4 w-4 rounded-full border-2 border-ring bg-background
        ring-offset-background transition-colors

        disabled:pointer-events-none disabled:opacity-50

        focus-visible:outline-hidden focus-visible:ring-ring

        hover:cursor-pointer
      `}
    />
  </SliderPrimitive.Root>
));
Slider.displayName = SliderPrimitive.Root.displayName;

export { Slider };
