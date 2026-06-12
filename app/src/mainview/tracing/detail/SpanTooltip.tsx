import { compactNumber, formatDuration, formatMoney } from "~/lib/format";
import type { Span, SpanNode } from "../../../server/telemetry/types";
import { spanColor, useTraceColors } from "./spanKinds";
import { rollupCost, rollupTokens } from "./rollups";

export type SpanHover = { span: Span; x: number; y: number };

export function SpanTooltip({
  heat,
  hover,
  maxCost,
  nodeByKey,
}: {
  heat: boolean;
  hover: SpanHover | null;
  maxCost: number;
  nodeByKey: Map<string, SpanNode>;
}) {
  const colors = useTraceColors();
  if (!hover) return null;
  const span = hover.span;
  const color = spanColor(span, colors, heat, maxCost);
  const left = Math.min(hover.x + 16, window.innerWidth - 280);
  const top = Math.min(hover.y + 18, window.innerHeight - 170);
  const isGroup =
    span.observationKind === "AGENT" || span.observationKind === "CHAIN";
  const node = nodeByKey.get(`${span.traceId}:${span.spanId}`);
  const cost = isGroup && node ? rollupCost(node) : numeric(span.costTotal);
  const tokens =
    isGroup && node
      ? rollupTokens(node)
      : ([span.inputTokens, span.outputTokens] as const);

  return (
    <div
      className="pointer-events-none fixed z-[70] flex min-w-52 flex-col gap-1.5 rounded-lg border border-border/70 bg-popover p-2.5 shadow-xl"
      style={{ left, top }}
    >
      <div className="flex items-center gap-2">
        <span
          className="h-2 w-2 flex-none rounded-sm"
          style={{ background: color }}
        />
        <span className="truncate text-xs font-semibold">{span.spanName}</span>
      </div>
      <TooltipRow label="duration">
        {formatDuration(Math.max(0, span.endTimeMs - span.startTimeMs))}
      </TooltipRow>
      {span.observationKind === "LLM" ? (
        <TooltipRow label="tokens">
          {compactNumber(span.inputTokens ?? 0)} →{" "}
          {compactNumber(span.outputTokens ?? 0)}
        </TooltipRow>
      ) : null}
      {span.observationKind === "LLM" && span.costTotal != null ? (
        <TooltipRow label="cost">
          <span style={heat ? { color } : undefined}>
            {formatMoney(numeric(span.costTotal) ?? 0)}
          </span>
        </TooltipRow>
      ) : null}
      {isGroup && cost != null ? (
        <TooltipRow label="cost (rolled up)">{formatMoney(cost)}</TooltipRow>
      ) : null}
      {isGroup && tokens[0] != null ? (
        <TooltipRow label="tokens (rolled up)">
          {compactNumber(tokens[0] ?? 0)} → {compactNumber(tokens[1] ?? 0)}
        </TooltipRow>
      ) : null}
      {span.statusCode === "STATUS_CODE_ERROR" ? (
        <span className="text-[11px] text-detail-failure">
          ✕ failed — click for output
        </span>
      ) : null}
    </div>
  );
}

function TooltipRow({
  children,
  label,
}: {
  children: React.ReactNode;
  label: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-6 text-[11px]">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-geist-mono text-foreground">{children}</span>
    </div>
  );
}

function numeric(value: string | null): number | null {
  if (value == null) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}
