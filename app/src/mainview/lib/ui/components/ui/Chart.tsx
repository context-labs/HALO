import * as React from "react";
import * as RechartsPrimitive from "recharts";

import { useBreakpoints } from "~/lib/ui/hooks/useBreakpoints.hook";
import { cn } from "~/lib/ui/utils/utils";

// Format: { THEME_NAME: CSS_SELECTOR }
const THEMES = { dark: ".dark", light: "" } as const;

export type ChartConfig = Record<
  string,
  {
    label?: React.ReactNode;
    icon?: React.ComponentType;
  } & (
    | { color?: never; theme: Record<keyof typeof THEMES, string> }
    | { color?: string; theme?: never }
  )
>;

type ChartContextProps = {
  config: ChartConfig;
};

const ChartContext = React.createContext<ChartContextProps | null>(null);

function useChart() {
  const context = React.useContext(ChartContext);

  if (!context) {
    throw new Error("useChart must be used within a <ChartContainer />");
  }

  return context;
}

const ChartContainer = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div"> & {
    config: ChartConfig;
    children: React.ComponentProps<
      typeof RechartsPrimitive.ResponsiveContainer
    >["children"];
  }
>(({ children, className, config, id, ...props }, ref) => {
  const uniqueId = React.useId();
  const chartId = `chart-${id ?? uniqueId.replace(/:/g, "")}`;

  return (
    <ChartContext.Provider value={{ config }}>
      <div
        className={cn(
          `
            flex aspect-video justify-center text-xs

            [&_.recharts-cartesian-axis-tick_text]:fill-muted-foreground

            [&_.recharts-cartesian-grid_line[stroke='#ccc']]:stroke-border/50

            [&_.recharts-curve.recharts-tooltip-cursor]:stroke-border

            [&_.recharts-dot[stroke='#fff']]:stroke-transparent

            [&_.recharts-layer]:outline-hidden

            [&_.recharts-polar-grid_[stroke='#ccc']]:stroke-border

            [&_.recharts-radial-bar-background-sector]:fill-muted

            [&_.recharts-rectangle.recharts-tooltip-cursor]:fill-muted

            [&_.recharts-reference-line_[stroke='#ccc']]:stroke-border

            [&_.recharts-sector[stroke='#fff']]:stroke-transparent

            [&_.recharts-sector]:outline-hidden

            [&_.recharts-surface]:outline-hidden
          `,
          className,
        )}
        data-chart={chartId}
        ref={ref}
        {...props}
      >
        <ChartStyle config={config} id={chartId} />
        <RechartsPrimitive.ResponsiveContainer>
          {children}
        </RechartsPrimitive.ResponsiveContainer>
      </div>
    </ChartContext.Provider>
  );
});
ChartContainer.displayName = "Chart";

const ChartStyle = ({ config, id }: { id: string; config: ChartConfig }) => {
  const colorConfig = Object.entries(config).filter(
    ([_, config]) => config.theme ?? config.color,
  );

  if (!colorConfig.length) {
    return null;
  }

  return (
    <style
      dangerouslySetInnerHTML={{
        __html: Object.entries(THEMES)
          .map(
            ([theme, prefix]) => `
${prefix} [data-chart=${id}] {
${colorConfig
  .map(([key, itemConfig]) => {
    const color =
      itemConfig.theme?.[theme as keyof typeof itemConfig.theme] ??
      itemConfig.color;
    return color ? `  --color-${key}: ${color};` : null;
  })
  .join("\n")}
}
`,
          )
          .join("\n"),
      }}
    />
  );
};

const ChartTooltip = RechartsPrimitive.Tooltip;

const ChartTooltipContent = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div"> &
    React.ComponentProps<typeof RechartsPrimitive.Tooltip> & {
      hideLabel?: boolean;
      hideIndicator?: boolean;
      indicator?: "dashed" | "dot" | "line";
      nameKey?: string;
      labelKey?: string;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      valueFormatter?: (value: any) => string;
    }
>(
  (
    {
      active,
      className,
      color,
      formatter,
      hideIndicator = false,
      hideLabel = false,
      indicator = "dot",
      label,
      labelClassName,
      labelFormatter,
      labelKey,
      nameKey,
      payload,
      valueFormatter,
    },
    ref,
  ) => {
    const { config } = useChart();
    const { isMobile } = useBreakpoints();

    const tooltipLabel = React.useMemo(() => {
      if (hideLabel || !payload?.length) {
        return null;
      }

      const [item] = payload;
      const key = `${labelKey ?? item?.dataKey ?? item?.name ?? "value"}`;
      const itemConfig = getPayloadConfigFromPayload(config, item, key);
      const value =
        !labelKey && typeof label === "string"
          ? (config[label]?.label ?? label)
          : itemConfig?.label;

      if (labelFormatter) {
        return (
          <div className={cn("font-medium", labelClassName)}>
            {labelFormatter(value, payload)}
          </div>
        );
      }

      if (!value) {
        return null;
      }

      return <div className={cn("font-medium", labelClassName)}>{value}</div>;
    }, [
      label,
      labelFormatter,
      payload,
      hideLabel,
      labelClassName,
      config,
      labelKey,
    ]);

    if (isMobile) {
      return null;
    }

    if (!active || !payload?.length) {
      return null;
    }

    const nestLabel = payload.length === 1 && indicator !== "dot";

    return (
      <div
        className={cn(
          `
            grid min-w-32 items-start gap-1.5 rounded-lg border
            border-subtle bg-background px-2.5 py-1.5 text-xs shadow-xl
          `,
          className,
        )}
        ref={ref}
      >
        {!nestLabel ? tooltipLabel : null}
        <div className="grid gap-1.5">
          {payload.map((item, index) => {
            const key = `${nameKey ?? item.name ?? item.dataKey ?? "value"}`;
            const itemConfig = getPayloadConfigFromPayload(config, item, key);
            const indicatorColor =
              // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
              color ?? (item.payload.fill as string | undefined) ?? item.color;

            return (
              <div
                className={cn(
                  `
                    flex w-full flex-wrap items-stretch gap-2

                    [&>svg]:h-2.5 [&>svg]:w-2.5 [&>svg]:text-muted-foreground
                  `,
                  indicator === "dot" && "items-center",
                )}
                key={item.dataKey}
              >
                {formatter && item.value !== undefined && item.name ? (
                  // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
                  formatter(item.value, item.name, item, index, item.payload)
                ) : (
                  <>
                    {itemConfig?.icon ? (
                      <itemConfig.icon />
                    ) : (
                      !hideIndicator && (
                        <div
                          className={cn(
                            `
                              shrink-0 rounded-[2px] border-subtle
                              bg-(--color-bg)
                            `,
                            {
                              "h-2.5 w-2.5": indicator === "dot",
                              "my-0.5": nestLabel && indicator === "dashed",
                              "w-0 border-[1.5px] border-dashed bg-transparent":
                                indicator === "dashed",
                              "w-1": indicator === "line",
                            },
                          )}
                          style={
                            {
                              "--color-bg": indicatorColor,
                              "--color-border": indicatorColor,
                            } as React.CSSProperties
                          }
                        />
                      )
                    )}
                    <div
                      className={cn(
                        "flex flex-1 justify-between gap-2 leading-none",
                        nestLabel ? "items-end" : "items-center",
                      )}
                    >
                      <div className="grid gap-1.5">
                        {nestLabel ? tooltipLabel : null}
                        <span className="text-muted-foreground">
                          {itemConfig?.label ?? item.name}
                        </span>
                      </div>
                      {item.value && (
                        <span
                          className={cn(`
                            font-mono font-medium tabular-nums text-foreground
                          `)}
                        >
                          {valueFormatter
                            ? valueFormatter(item.value)
                            : item.value.toLocaleString()}
                        </span>
                      )}
                    </div>
                  </>
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  },
);
ChartTooltipContent.displayName = "ChartTooltip";

const ChartLegend = RechartsPrimitive.Legend;

const ChartLegendContent = React.forwardRef<
  HTMLDivElement,
  Pick<RechartsPrimitive.LegendProps, "payload" | "verticalAlign"> &
    React.ComponentProps<"div"> & {
      hideIcon?: boolean;
      nameKey?: string;
    }
>(
  (
    { className, hideIcon = false, nameKey, payload, verticalAlign = "bottom" },
    ref,
  ) => {
    const { config } = useChart();

    if (!payload?.length) {
      return null;
    }

    return (
      <div
        className={cn(
          "flex items-center justify-center gap-4",
          verticalAlign === "top" ? "pb-3" : "pt-3",
          className,
        )}
        ref={ref}
      >
        {payload.map((item) => {
          // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
          const key = `${nameKey ?? item.dataKey ?? "value"}`;
          const itemConfig = getPayloadConfigFromPayload(config, item, key);

          return (
            <div
              className={cn(
                `
                  flex items-center gap-1.5

                  [&>svg]:h-3 [&>svg]:w-3 [&>svg]:text-muted-foreground
                `,
              )}
              key={item.value as string}
            >
              {itemConfig?.icon && !hideIcon ? (
                <itemConfig.icon />
              ) : (
                <div
                  className="h-2 w-2 shrink-0 rounded-[2px]"
                  style={{
                    backgroundColor: item.color,
                  }}
                />
              )}
              {itemConfig?.label}
            </div>
          );
        })}
      </div>
    );
  },
);
ChartLegendContent.displayName = "ChartLegend";

// Helper to extract item config from a payload.
function getPayloadConfigFromPayload(
  config: ChartConfig,
  payload: unknown,
  key: string,
) {
  if (typeof payload !== "object" || payload === null) {
    return undefined;
  }

  const payloadPayload =
    "payload" in payload &&
    typeof payload.payload === "object" &&
    payload.payload !== null
      ? payload.payload
      : undefined;

  let configLabelKey: string = key;

  if (
    key in payload &&
    typeof payload[key as keyof typeof payload] === "string"
  ) {
    configLabelKey = payload[key as keyof typeof payload] as string;
  } else if (
    payloadPayload &&
    key in payloadPayload &&
    typeof payloadPayload[key as keyof typeof payloadPayload] === "string"
  ) {
    configLabelKey = payloadPayload[
      key as keyof typeof payloadPayload
    ] as string;
  }

  return configLabelKey in config ? config[configLabelKey] : config[key];
}

export {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
  ChartStyle,
};
