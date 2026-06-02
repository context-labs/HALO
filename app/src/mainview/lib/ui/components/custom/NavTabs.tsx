import type { ReactNode } from "react";
import {
  Children,
  isValidElement,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
} from "react";

import { cn } from "~/lib/ui/utils/utils";

type NavTabItem = {
  value: string;
  label: ReactNode;
  count?: number;
  icon?: React.ComponentType<{ className?: string }>;
};

type NavTabsProps = {
  value: string;
  onValueChange: (value: string) => void;
  children: ReactNode;
  className?: string;
};

type NavTabProps = {
  value: string;
  children: ReactNode;
  count?: number;
  icon?: React.ComponentType<{ className?: string }>;
};

// NavTab is a declarative config element — rendering is handled by NavTabs.
function NavTab(_props: NavTabProps) {
  return null;
}

function NavTabs({ value, onValueChange, children, className }: NavTabsProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const indicatorRef = useRef<HTMLDivElement>(null);
  const tabRefs = useRef<Map<string, HTMLButtonElement>>(new Map());
  const hasAnimated = useRef(false);

  // Extract tab config from children
  const tabs = useMemo(() => {
    const items: NavTabItem[] = [];
    for (const child of Children.toArray(children)) {
      if (isValidElement<NavTabProps>(child) && child.type === NavTab) {
        const props = child.props;
        items.push({
          value: props.value,
          label: props.children,
          count: props.count,
          icon: props.icon,
        });
      }
    }
    return items;
  }, [children]);

  const updateIndicator = useCallback(() => {
    const activeTab = tabRefs.current.get(value);
    const container = containerRef.current;
    const indicator = indicatorRef.current;
    if (!activeTab || !container || !indicator) return;

    const containerRect = container.getBoundingClientRect();
    const tabRect = activeTab.getBoundingClientRect();
    const left = tabRect.left - containerRect.left;
    const width = tabRect.width;

    // First render: position instantly without transition
    if (!hasAnimated.current) {
      indicator.style.transition = "none";
      indicator.style.transform = `translateX(${left}px)`;
      indicator.style.width = `${width}px`;
      // Enable transitions after the browser paints
      requestAnimationFrame(() => {
        if (indicatorRef.current) {
          indicatorRef.current.style.transition = "";
        }
        hasAnimated.current = true;
      });
    } else {
      indicator.style.transform = `translateX(${left}px)`;
      indicator.style.width = `${width}px`;
    }
  }, [value]);

  // Update on value change, mount, and when tabs change
  useLayoutEffect(() => {
    updateIndicator();
  }, [updateIndicator, tabs]);

  // Handle resize
  useEffect(() => {
    const observer = new ResizeObserver(() => {
      updateIndicator();
    });
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }
    return () => observer.disconnect();
  }, [updateIndicator]);

  const setTabRef = useCallback(
    (tabValue: string) => (el: HTMLButtonElement | null) => {
      if (el) {
        tabRefs.current.set(tabValue, el);
      } else {
        tabRefs.current.delete(tabValue);
      }
    },
    [],
  );

  return (
    <div
      ref={containerRef}
      className={cn("relative flex items-end", className)}
      role="tablist"
    >
      {tabs.map((tab) => {
        const isActive = tab.value === value;
        const Icon = tab.icon;

        return (
          <button
            key={tab.value}
            ref={setTabRef(tab.value)}
            role="tab"
            aria-selected={isActive}
            onClick={() => onValueChange(tab.value)}
            className={cn(
              "relative inline-flex items-center gap-2 px-4 pb-2.5 pt-1 text-sm font-medium transition-colors duration-150",
              "outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background rounded-sm",
              isActive
                ? "text-foreground"
                : "text-muted-foreground hover:text-foreground/80",
            )}
          >
            {Icon ? <Icon className="h-4 w-4" /> : null}
            {tab.label}
            {tab.count != null && tab.count > 0 ? (
              <span className="rounded-full bg-primary/10 px-1.5 py-0.5 text-[10px] font-semibold tabular-nums leading-none text-primary">
                {tab.count}
              </span>
            ) : null}
          </button>
        );
      })}

      {/* Sliding indicator */}
      <div
        ref={indicatorRef}
        className="pointer-events-none absolute bottom-0 left-0 h-[2px] bg-foreground transition-all duration-200 ease-out"
      />
    </div>
  );
}

export { NavTabs, NavTab };
