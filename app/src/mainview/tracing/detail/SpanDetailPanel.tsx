import { useEffect, useRef, useState, type ReactNode } from "react";
import { X } from "lucide-react";

import { cn } from "~/lib/ui";
import {
  compactNumber,
  formatDuration,
  formatMoney,
  prettyMaybeJson,
} from "~/lib/format";
import type { Span, SpanNode } from "../../../server/telemetry/types";
import { parseSpanConversation } from "../llmMessages";
import { isSyntheticSpan } from "../spanTree";
import { CopyTextButton } from "./CopyTextButton";
import { LlmSpanView } from "./LlmSpanView";
import { rollupCost, rollupTokens } from "./rollups";
import { spanColor, useTraceColors } from "./spanKinds";

type PanelTab = "overview" | "raw";

const PANEL_WIDTH_STORAGE_KEY = "halo.spanSheet.width";
const PANEL_DEFAULT_WIDTH = 416;
const PANEL_MIN_WIDTH = 320;
const PANEL_MAX_WIDTH = 760;

/**
 * Span details as a nested sheet: slides in over the trace sheet's content
 * (rather than squeezing it) and resizes via the handle on its left edge.
 * The parent container must be `relative`.
 */
export function SpanDetailPanel({
  domainStartMs,
  heat,
  maxCost,
  node,
  onClose,
  span,
}: {
  domainStartMs: number;
  heat: boolean;
  maxCost: number;
  node: SpanNode | null;
  onClose: () => void;
  span: Span;
}) {
  const colors = useTraceColors();
  const [tab, setTab] = useState<PanelTab>("overview");
  const resize = useResizablePanel();
  const color = spanColor(span, colors, heat, maxCost);
  const isGroup =
    span.observationKind === "AGENT" || span.observationKind === "CHAIN";
  const cost =
    isGroup && node ? rollupCost(node) : numeric(span.costTotal);
  const [tokensIn, tokensOut] =
    isGroup && node
      ? rollupTokens(node)
      : [span.inputTokens, span.outputTokens];
  const error = span.statusCode === "STATUS_CODE_ERROR";
  const conversation = parseSpanConversation(span);
  const model = span.llmModelName || span.llmResponseModel;

  return (
    <aside
      className="absolute inset-y-0 right-0 z-30 flex max-w-[85%] flex-col border-l border-subtle bg-background shadow-[-24px_0_56px_rgba(0,0,0,0.4)] duration-200 animate-in fade-in slide-in-from-right-1/4"
      ref={resize.panelRef}
      style={{ width: resize.width }}
    >
      {/* resize handle */}
      <div
        aria-label="Resize span details"
        aria-orientation="vertical"
        className="group absolute inset-y-0 -left-1 z-10 flex w-2 cursor-col-resize select-none items-stretch justify-center focus-visible:outline-none"
        onKeyDown={resize.onKeyDown}
        onPointerDown={resize.onPointerDown}
        role="separator"
        tabIndex={0}
      >
        <div
          className={cn(
            "w-px bg-transparent transition-colors group-hover:bg-border group-focus-visible:bg-detail-brand/60",
            resize.dragging && "bg-detail-brand/60",
          )}
        />
      </div>
      <div className="flex flex-none items-center gap-2.5 border-b border-subtle px-3.5 py-3">
        <span
          className="h-2.5 w-2.5 flex-none rounded-sm"
          style={{ background: color }}
        />
        <span className="min-w-0 flex-1 truncate text-[13px] font-semibold">
          {span.spanName}
        </span>
        <span
          className="flex-none rounded-full border px-2 py-px text-[10px] font-semibold uppercase tracking-[0.06em]"
          style={{
            background: `color-mix(in srgb, ${color} 14%, transparent)`,
            borderColor: `color-mix(in srgb, ${color} 35%, transparent)`,
            color,
          }}
        >
          {span.observationKind.toLowerCase()}
        </span>
        <button
          aria-label="Close span details"
          className="flex-none text-muted-foreground transition-colors hover:text-foreground"
          onClick={onClose}
          type="button"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="flex flex-none gap-0.5 border-b border-subtle px-3.5 py-2">
        {(["overview", "raw"] as const).map((item) => (
          <button
            className={cn(
              "rounded-md px-2.5 py-1 text-xs font-medium capitalize transition-colors",
              tab === item
                ? "bg-muted text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
            key={item}
            onClick={() => setTab(item)}
            type="button"
          >
            {item}
          </button>
        ))}
      </div>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-3.5 pb-8">
        {tab === "raw" ? (
          <CollapsiblePre text={JSON.stringify(rawSpanView(span), null, 2)} />
        ) : (
          <>
            <div className="grid flex-none grid-cols-3 gap-x-2.5 gap-y-3 rounded-lg border border-subtle bg-background p-3">
              <PanelStat label="start">
                +{formatDuration(Math.max(0, span.startTimeMs - domainStartMs))}
              </PanelStat>
              <PanelStat label="duration">
                {formatDuration(Math.max(0, span.endTimeMs - span.startTimeMs))}
              </PanelStat>
              <PanelStat label={isGroup ? "cost (rolled up)" : "cost"}>
                {cost == null ? "—" : formatMoney(cost)}
              </PanelStat>
              {tokensIn != null || tokensOut != null ? (
                <>
                  <PanelStat label="tokens in">
                    {tokensIn == null ? "—" : compactNumber(tokensIn)}
                  </PanelStat>
                  <PanelStat label="tokens out">
                    {tokensOut == null ? "—" : compactNumber(tokensOut)}
                  </PanelStat>
                </>
              ) : null}
              {model ? <PanelStat label="model">{model}</PanelStat> : null}
              {node && node.children.length > 0 ? (
                <PanelStat label="children">{node.children.length}</PanelStat>
              ) : null}
            </div>

            {error ? (
              <div className="flex-none rounded-md border border-detail-failure/40 bg-detail-failure/10 px-3 py-2 text-xs text-detail-failure">
                ✕ {span.statusMessage || "This span failed — see output below."}
              </div>
            ) : null}

            {isSyntheticSpan(span) ? (
              <p className="text-xs text-muted-foreground">
                {span.statusMessage ||
                  "Synthetic span — grouped for display only."}
              </p>
            ) : null}

            {conversation && span.observationKind === "LLM" ? (
              <LlmSpanView conversation={conversation} />
            ) : (
              <>
                {span.input ? (
                  <CodeSection label="Input" text={span.input} />
                ) : null}
                {span.output ? (
                  <CodeSection label="Output" text={span.output} />
                ) : null}
                {!span.input && !span.output && conversation ? (
                  <LlmSpanView conversation={conversation} />
                ) : null}
              </>
            )}

            {span.events.length > 0 ? (
              <div className="space-y-1.5">
                <SectionLabel>Events</SectionLabel>
                <div className="space-y-1">
                  {span.events.map((event, index) => {
                    const eventMs = Date.parse(event.timestamp);
                    const isErrorEvent = /error|exception/i.test(event.name);
                    return (
                      <div
                        className="flex items-center gap-2 text-[11px]"
                        key={index}
                      >
                        <span
                          className="h-2.5 w-1 flex-none rounded-sm"
                          style={{
                            background: isErrorEvent
                              ? colors.error
                              : colors.marker,
                          }}
                        />
                        <span className="flex-none font-geist-mono text-muted-foreground">
                          {Number.isFinite(eventMs)
                            ? `+${formatDuration(Math.max(0, eventMs - span.startTimeMs))}`
                            : "—"}
                        </span>
                        <span className="truncate text-muted-foreground">
                          {event.name}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : null}
          </>
        )}
      </div>
    </aside>
  );
}

function PanelStat({ children, label }: { children: ReactNode; label: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <span className="truncate text-[10px] uppercase tracking-[0.06em] text-muted-foreground">
        {label}
      </span>
      <span className="truncate font-geist-mono text-xs text-foreground">
        {children}
      </span>
    </div>
  );
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
      {children}
    </div>
  );
}

function CodeSection({ label, text }: { label: string; text: string }) {
  return (
    <div className="space-y-1.5">
      <SectionLabel>{label}</SectionLabel>
      <CollapsiblePre text={prettyMaybeJson(text)} />
    </div>
  );
}

const PRE_COLLAPSE_THRESHOLD = 4_000;

/**
 * Pre block that clamps very large payloads behind a "Show all" toggle.
 * Beyond readability, this keeps the resize drag fast — re-wrapping a
 * multi-hundred-KB payload on every pointer move is what makes a panel feel
 * sluggish. Nothing clips: expansion is explicit and the panel body scrolls.
 */
function CollapsiblePre({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  const collapsible = text.length > PRE_COLLAPSE_THRESHOLD;
  const visible =
    collapsible && !expanded ? `${text.slice(0, PRE_COLLAPSE_THRESHOLD)}…` : text;
  return (
    <div>
      <div className="group relative">
        <pre className="overflow-x-auto whitespace-pre-wrap break-words rounded-md border border-subtle bg-background p-3 font-geist-mono text-[11px] leading-relaxed text-muted-foreground">
          {visible}
        </pre>
        <CopyTextButton text={text} />
      </div>
      {collapsible ? (
        <button
          className="mt-1.5 text-xs font-medium text-link hover:underline"
          onClick={() => setExpanded((current) => !current)}
          type="button"
        >
          {expanded
            ? "Show less"
            : `Show all ${text.length.toLocaleString()} characters`}
        </button>
      ) : null}
    </div>
  );
}


function rawSpanView(span: Span) {
  return {
    agentId: span.agentId,
    agentName: span.agentName,
    cost: {
      input: span.costInput,
      output: span.costOutput,
      total: span.costTotal,
    },
    durationMs: span.durationMs,
    endTime: span.endTime,
    events: span.events,
    input: span.input,
    inputMessages: span.inputMessages,
    links: span.links,
    llmModelName: span.llmModelName,
    llmProvider: span.llmProvider,
    observationKind: span.observationKind,
    output: span.output,
    outputMessages: span.outputMessages,
    parentSpanId: span.parentSpanId,
    resourceAttributes: span.resourceAttributes,
    serviceName: span.serviceName,
    sessionId: span.sessionId,
    spanAttributes: span.spanAttributes,
    spanId: span.spanId,
    spanKind: span.spanKind,
    spanName: span.spanName,
    startTime: span.startTime,
    statusCode: span.statusCode,
    statusMessage: span.statusMessage,
    tokens: {
      cacheRead: span.cacheReadTokens,
      cacheWrite: span.cacheWriteTokens,
      input: span.inputTokens,
      output: span.outputTokens,
      reasoning: span.reasoningTokens,
      total: span.totalTokens,
    },
    traceId: span.traceId,
    userId: span.userId,
  };
}

function numeric(value: string | null): number | null {
  if (value == null) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

/** Sheet width, dragged from the left edge and persisted across sessions. */
function useResizablePanel() {
  const [width, setWidth] = useState(PANEL_DEFAULT_WIDTH);
  const [dragging, setDragging] = useState(false);
  const panelRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const stored = Number(localStorage.getItem(PANEL_WIDTH_STORAGE_KEY));
    if (
      Number.isFinite(stored) &&
      stored >= PANEL_MIN_WIDTH &&
      stored <= PANEL_MAX_WIDTH
    ) {
      setWidth(stored);
    }
  }, []);

  const persist = (value: number) => {
    localStorage.setItem(PANEL_WIDTH_STORAGE_KEY, String(value));
  };

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = width;
    setDragging(true);
    let latest = startWidth;
    let frame = 0;
    // Write the width straight to the DOM while dragging — re-rendering the
    // panel (often holding large JSON blocks) per pointer move makes the
    // handle lag. Writes coalesce to one per animation frame, and React
    // state syncs once on release.
    const onMove = (moveEvent: PointerEvent) => {
      latest = clampPanelWidth(startWidth + (startX - moveEvent.clientX));
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        if (panelRef.current) panelRef.current.style.width = `${latest}px`;
      });
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      if (frame) cancelAnimationFrame(frame);
      if (panelRef.current) panelRef.current.style.width = `${latest}px`;
      setDragging(false);
      setWidth(latest);
      persist(latest);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const delta = event.key === "ArrowLeft" ? 16 : -16;
    setWidth((current) => {
      const next = clampPanelWidth(current + delta);
      persist(next);
      return next;
    });
  };

  return { dragging, onKeyDown, onPointerDown, panelRef, width };
}

function clampPanelWidth(value: number) {
  return Math.min(PANEL_MAX_WIDTH, Math.max(PANEL_MIN_WIDTH, value));
}
