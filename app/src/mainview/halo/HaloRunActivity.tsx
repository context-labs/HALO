import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Braces,
  CheckCircle2,
  Copy,
  FileJson2,
  MessageSquareText,
  Wrench,
} from "lucide-react";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
  Badge,
  Button,
  Code,
  JsonComponent,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Tooltip,
  cn,
  toast,
} from "~/lib/ui";
import type { HaloRunEvent } from "../../server/halo/types";
import {
  presentHaloAgentStep,
  type HaloAgentConsoleKind,
  type HaloAgentConsoleRow,
} from "./haloAgentConsolePresenter";

type ActivityTab = "all" | "call" | "result" | "message";

const TAB_CONFIG: Array<{ label: string; value: ActivityTab }> = [
  { label: "All", value: "all" },
  { label: "Tool calls", value: "call" },
  { label: "Results", value: "result" },
  { label: "Messages", value: "message" },
];

export function HaloRunActivity({
  className,
  events,
}: {
  className?: string;
  events: HaloRunEvent[];
}) {
  const [activeTab, setActiveTab] = useState<ActivityTab>("all");
  const [expandedRows, setExpandedRows] = useState<Record<string, boolean>>({});
  const scrollerRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);

  const rows = useMemo(
    () => events.flatMap((event) => presentHaloAgentStep(event)),
    [events],
  );
  const filteredRows = useMemo(
    () =>
      activeTab === "all"
        ? rows
        : rows.filter((row) => {
            if (activeTab === "message") return row.kind === "message" || row.kind === "raw";
            return row.kind === activeTab;
          }),
    [activeTab, rows],
  );
  const counts = useMemo(
    () => ({
      all: rows.length,
      call: rows.filter((row) => row.kind === "call").length,
      message: rows.filter((row) => row.kind === "message" || row.kind === "raw").length,
      result: rows.filter((row) => row.kind === "result").length,
    }),
    [rows],
  );
  const firstCreatedMs = useMemo(() => {
    const times = rows
      .map((row) => Date.parse(row.createdAt))
      .filter((time) => Number.isFinite(time));
    return times.length > 0 ? Math.min(...times) : null;
  }, [rows]);
  const lastRowKey = filteredRows.at(-1)?.key ?? "";

  useEffect(() => {
    const scroller = scrollerRef.current;
    if (!scroller || !shouldAutoScrollRef.current) return;
    scroller.scrollTo({ behavior: "smooth", top: scroller.scrollHeight });
  }, [lastRowKey, filteredRows.length]);

  function handleScroll() {
    const scroller = scrollerRef.current;
    if (!scroller) return;
    const distanceFromBottom =
      scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
    shouldAutoScrollRef.current = distanceFromBottom < 96;
  }

  return (
    <section className={cn("overflow-hidden border border-subtle bg-card", className)}>
      <div className="border-b border-subtle px-4 py-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="grid h-7 w-7 place-items-center rounded-md border border-subtle bg-background-muted text-muted-foreground">
                <Activity className="h-4 w-4" />
              </span>
              <div>
                <h2 className="text-sm font-semibold">Activity</h2>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  Tool calls, results, and assistant messages from this run.
                </p>
              </div>
            </div>
          </div>
          <Badge className="gap-1.5" size="sm" variant="status-brand">
            {rows.length} event{rows.length === 1 ? "" : "s"}
          </Badge>
        </div>

        <Tabs
          className="mt-3"
          onValueChange={(value) => setActiveTab(value as ActivityTab)}
          value={activeTab}
        >
          <TabsList className="gap-1 rounded-md bg-background-muted p-1">
            {TAB_CONFIG.map((tab) => (
              <TabsTrigger
                className="h-7 w-auto rounded-sm bg-transparent px-2.5 py-1 text-xs text-muted-foreground data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-xs"
                key={tab.value}
                value={tab.value}
              >
                {tab.label}
                <span className="ml-1 text-muted-foreground">{counts[tab.value]}</span>
              </TabsTrigger>
            ))}
          </TabsList>
          {TAB_CONFIG.map((tab) => (
            <TabsContent className="m-0 hidden" key={tab.value} value={tab.value} />
          ))}
        </Tabs>
      </div>

      <div
        className="max-h-[42rem] min-h-72 overflow-auto"
        onScroll={handleScroll}
        ref={scrollerRef}
      >
        {filteredRows.length === 0 ? (
          <div className="grid min-h-56 place-items-center p-6 text-center">
            <div>
              <Activity className="mx-auto h-7 w-7 text-muted-foreground" />
              <p className="mt-3 text-sm font-medium">Waiting for run activity</p>
              <p className="mt-1 text-xs text-muted-foreground">
                HALO tool calls and results will appear here as they arrive.
              </p>
            </div>
          </div>
        ) : (
          <ol className="divide-y divide-subtle">
            {filteredRows.map((row) => (
              <ActivityRow
                elapsed={formatElapsed(row.createdAt, firstCreatedMs)}
                expanded={Boolean(expandedRows[row.key])}
                key={row.key}
                onCopy={copyToClipboard}
                onExpandedChange={(expanded) =>
                  setExpandedRows((current) => ({ ...current, [row.key]: expanded }))
                }
                row={row}
              />
            ))}
          </ol>
        )}
      </div>
    </section>
  );
}

function ActivityRow({
  elapsed,
  expanded,
  onCopy,
  onExpandedChange,
  row,
}: {
  elapsed: string;
  expanded: boolean;
  onCopy: (label: string, value: string) => void;
  onExpandedChange: (expanded: boolean) => void;
  row: HaloAgentConsoleRow;
}) {
  const tone = toneForKind(row.kind);
  const preview = summaryTextForRow(row);

  return (
    <li className="group grid grid-cols-[2rem_minmax(0,1fr)] gap-3 px-4 py-3 transition-colors hover:bg-background-muted/50">
      <div className="pt-0.5">
        <span
          className={cn(
            "grid h-7 w-7 place-items-center rounded-md border",
            tone.icon,
          )}
        >
          <ActivityIcon kind={row.kind} />
        </span>
      </div>
      <div className="min-w-0">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <Badge className="px-2 py-0.5" size="sm" variant={tone.badge}>
                {labelForKind(row.kind)}
              </Badge>
              <span className="truncate text-sm font-medium">{row.title}</span>
              {row.subtitle ? (
                <span className="text-xs text-muted-foreground">{row.subtitle}</span>
              ) : null}
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
              <span>#{row.stepSequence ?? row.sequence}</span>
              <span>·</span>
              <span>{elapsed}</span>
              <span>·</span>
              <span>{row.agentName}</span>
              {row.depth != null ? (
                <>
                  <span>·</span>
                  <span>depth {row.depth}</span>
                </>
              ) : null}
              {row.toolCallId ? (
                <Badge className="font-mono" size="sm" variant="outline">
                  {shortId(row.toolCallId)}
                </Badge>
              ) : null}
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            <CopyButton
              label={row.kind === "call" ? "Copy command" : "Copy output"}
              onClick={() => onCopy(row.kind === "call" ? "Command" : "Output", row.copyText)}
            />
            <CopyButton
              label="Copy raw JSON"
              onClick={() => onCopy("Raw payload", JSON.stringify(row.rawPayload, null, 2))}
            />
          </div>
        </div>

        {row.command ? (
          <div className="mt-2 flex min-w-0 items-start gap-2 rounded-md border border-subtle bg-background-muted px-3 py-2">
            <span className="select-none font-mono text-xs text-muted-foreground">$</span>
            <Code
              className="min-w-0 flex-1 break-words p-0 text-xs leading-relaxed"
              disableCopyToClipboard
            >
              {row.command}
            </Code>
          </div>
        ) : null}

        {row.summaries.length > 0 ? (
          <dl className="mt-2 flex flex-wrap gap-1.5">
            {row.summaries.slice(0, 6).map((summary) => (
              <div
                className="min-w-0 rounded-md border border-subtle bg-background-muted px-2 py-1"
                key={summary.label}
              >
                <dt className="truncate text-[10px] uppercase tracking-wide text-muted-foreground">
                  {summary.label}
                </dt>
                <dd className="mt-0.5 max-w-40 truncate text-xs">{summary.value}</dd>
              </div>
            ))}
          </dl>
        ) : null}

        {preview ? (
          <p className="mt-2 line-clamp-2 text-sm leading-relaxed text-muted-foreground">
            {preview}
          </p>
        ) : null}

        <Accordion
          className="mt-2"
          collapsible
          onValueChange={(value) => onExpandedChange(value === "raw")}
          type="single"
          value={expanded ? "raw" : ""}
        >
          <AccordionItem className="border-none" value="raw">
            <AccordionTrigger
              aria-label="Toggle activity details"
              className="py-1 text-xs font-medium text-muted-foreground hover:text-foreground [&>svg]:h-3 [&>svg]:w-3"
            >
              Details
            </AccordionTrigger>
            <AccordionContent className="pb-0 pt-2">
              {expanded ? (
                <div className="space-y-3 rounded-md border border-subtle bg-background-muted p-3">
                  {row.body ? (
                    <div>
                      <p className="mb-2 text-xs font-medium text-muted-foreground">
                        {row.kind === "call" ? "Arguments" : "Output"}
                      </p>
                      <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-md border border-subtle bg-background p-3 text-xs leading-relaxed">
                        {row.body.text}
                      </pre>
                    </div>
                  ) : null}
                  <div>
                    <p className="mb-2 text-xs font-medium text-muted-foreground">
                      Raw payload
                    </p>
                    <div className="max-h-96 overflow-auto rounded-md border border-subtle bg-background p-2 text-xs">
                      <JsonComponent data={row.rawPayload} collapsed={2} />
                    </div>
                  </div>
                </div>
              ) : null}
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>
    </li>
  );
}

function CopyButton({
  label,
  onClick,
}: {
  label: string;
  onClick: () => void;
}) {
  return (
    <Tooltip content={label}>
      <Button
        aria-label={label}
        className="h-7 w-7 opacity-70 group-hover:opacity-100"
        onClick={onClick}
        size="icon"
        variant="ghost"
      >
        {label === "Copy raw JSON" ? (
          <FileJson2 className="h-3.5 w-3.5" />
        ) : (
          <Copy className="h-3.5 w-3.5" />
        )}
      </Button>
    </Tooltip>
  );
}

function toneForKind(kind: HaloAgentConsoleKind): {
  badge: "status-brand" | "status-success" | "status-warning" | "outline";
  icon: string;
} {
  if (kind === "call") {
    return {
      badge: "status-brand",
      icon: "border-detail-brand/20 bg-detail-brand/10 text-detail-brand",
    };
  }
  if (kind === "result") {
    return {
      badge: "status-success",
      icon: "border-detail-success/20 bg-detail-success/10 text-detail-success",
    };
  }
  if (kind === "message") {
    return {
      badge: "status-warning",
      icon: "border-detail-warning/20 bg-detail-warning/10 text-detail-warning",
    };
  }
  return {
    badge: "outline",
    icon: "border-subtle bg-background-muted text-muted-foreground",
  };
}

function ActivityIcon({ kind }: { kind: HaloAgentConsoleKind }) {
  if (kind === "call") return <Wrench className="h-3.5 w-3.5" />;
  if (kind === "result") return <CheckCircle2 className="h-3.5 w-3.5" />;
  if (kind === "message") return <MessageSquareText className="h-3.5 w-3.5" />;
  return <Braces className="h-3.5 w-3.5" />;
}

function labelForKind(kind: HaloAgentConsoleKind) {
  if (kind === "call") return "Tool call";
  if (kind === "result") return "Result";
  if (kind === "message") return "Message";
  return "Raw";
}

function summaryTextForRow(row: HaloAgentConsoleRow) {
  if (row.summaries.length > 0 || row.command) return null;
  if (!row.body?.text) return null;

  const compact = row.body.text.replace(/\s+/g, " ").trim();
  if (!compact) return null;
  return compact.length > 240 ? `${compact.slice(0, 237)}...` : compact;
}

function formatElapsed(createdAt: string, firstCreatedMs: number | null) {
  const createdMs = Date.parse(createdAt);
  if (!Number.isFinite(createdMs) || firstCreatedMs == null) return "+0ms";
  const elapsedMs = Math.max(0, createdMs - firstCreatedMs);
  if (elapsedMs < 1_000) return `+${elapsedMs}ms`;
  return `+${(elapsedMs / 1_000).toFixed(1)}s`;
}

function shortId(value: string) {
  if (value.length <= 12) return value;
  return `${value.slice(0, 8)}...${value.slice(-4)}`;
}

async function copyToClipboard(label: string, value: string) {
  await navigator.clipboard.writeText(value);
  toast.success({
    description: value.length > 80 ? `${value.slice(0, 77)}...` : value,
    title: `${label} copied`,
  });
}
