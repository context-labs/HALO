import { useEffect, useState } from "react";

import { useHasMounted } from "~/lib/ui/hooks/useHasMounted.hook";

type BreakpointContext = {
  breakpoint: Breakpoint;
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  isWidescreen: boolean;
};

export const BREAKPOINTS = {
  desktop: 1024,
  mobile: 0,
  tablet: 768,
  widescreen: 1440,
} as const;

export type Breakpoint = keyof typeof BREAKPOINTS;

const sortedBreakpoints = Object.entries(BREAKPOINTS)
  .map(([key, value]) => ({ key, value }))
  .sort((a, b) => a.value - b.value);

const getCurrentBreakpoint = (): Breakpoint => {
  let current: Breakpoint = "desktop";
  if (typeof window === "undefined") {
    return current;
  }

  const width = window.innerWidth;
  for (const { key, value } of sortedBreakpoints) {
    if (width >= value) {
      current = key as Breakpoint;
    }
  }

  return current;
};

export function useBreakpoints(): BreakpointContext {
  const mounted = useHasMounted();
  const [breakpoint, setBreakpoint] = useState(getCurrentBreakpoint());

  useEffect(() => {
    const handleResize = () => {
      const newBreakpoint = getCurrentBreakpoint();
      setBreakpoint(newBreakpoint);
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  let context: BreakpointContext = {
    breakpoint,
    isDesktop: breakpoint === "desktop",
    isMobile: breakpoint === "mobile",
    isTablet: breakpoint === "tablet",
    isWidescreen: breakpoint === "widescreen",
  };

  // We default to desktop breakpoint for SSR.
  if (!mounted) {
    context = {
      breakpoint,
      isDesktop: true,
      isMobile: false,
      isTablet: false,
      isWidescreen: false,
    };
  }

  return context;
}
