import { useMemo, useRef } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Badge, cn } from "~/lib/ui";
import {
  dashboardLinkLabel,
  linkifyDashboardTags,
  parseDashboardLink,
} from "./reportLinks";

/**
 * Renders a completed run's markdown report with jump-chips for its section
 * headings. In-flight streamed text stays in the parent's <pre> — this view
 * only mounts once a final answer exists.
 */
export function RunReportView({
  markdown,
  onOpenSpanLink,
  onOpenTraceLink,
}: {
  markdown: string;
  onOpenSpanLink?: (traceId: string, spanId: string) => void;
  onOpenTraceLink?: (traceId: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sections = useMemo(() => extractHeadings(markdown), [markdown]);
  const linkifiedMarkdown = useMemo(() => linkifyDashboardTags(markdown), [markdown]);

  const jumpTo = (slug: string) => {
    containerRef.current
      ?.querySelector(`[data-heading-slug="${slug}"]`)
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div>
      {sections.length > 1 ? (
        <div className="mb-4 flex flex-wrap gap-1.5">
          {sections.map((section) => (
            <button
              key={section.slug}
              onClick={() => jumpTo(section.slug)}
              type="button"
            >
              <Badge
                className="cursor-pointer hover:bg-muted"
                size="sm"
                variant="outline"
              >
                {section.text}
              </Badge>
            </button>
          ))}
        </div>
      ) : null}
      <div
        className={cn(
          // Body rhythm: 15px Inter with a touch of negative tracking and a
          // relaxed measure reads like set prose rather than UI text.
          "min-w-0 text-[0.9375rem] leading-[1.75] tracking-[-0.011em] antialiased",
          "[&_h1]:mt-8 [&_h1]:mb-3 [&_h1]:text-xl [&_h1]:font-semibold [&_h1]:leading-snug [&_h1]:tracking-[-0.02em] first:[&_h1]:mt-0",
          "[&_h2]:mt-8 [&_h2]:mb-3 [&_h2]:text-lg [&_h2]:font-semibold [&_h2]:leading-snug [&_h2]:tracking-[-0.02em] first:[&_h2]:mt-0",
          "[&_h3]:mt-6 [&_h3]:mb-2 [&_h3]:text-base [&_h3]:font-semibold [&_h3]:leading-snug [&_h3]:tracking-[-0.015em] first:[&_h3]:mt-0",
          "[&_p]:my-4 first:[&_p]:mt-0 last:[&_p]:mb-0",
          "[&_strong]:font-semibold",
          "[&_ul]:my-4 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:my-4 [&_ol]:list-decimal [&_ol]:pl-5",
          "[&_li]:my-1.5 [&_li]:pl-1 [&_li]:leading-[1.65] [&_li]:marker:text-muted-foreground/60",
          "[&_li_p]:my-1.5 [&_li_ul]:my-1.5 [&_li_ol]:my-1.5",
          "[&_a]:text-link [&_a]:underline-offset-2 hover:[&_a]:underline",
          "[&_blockquote]:my-4 [&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-4 [&_blockquote]:text-muted-foreground",
          "[&_code]:rounded [&_code]:bg-background-muted [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[0.85em] [&_code]:tracking-normal",
          "[&_pre]:my-4 [&_pre]:overflow-auto [&_pre]:rounded-md [&_pre]:border [&_pre]:border-subtle [&_pre]:bg-background-muted [&_pre]:p-3 [&_pre]:text-xs [&_pre]:leading-relaxed [&_pre]:tracking-normal [&_pre_code]:bg-transparent [&_pre_code]:p-0",
          "[&_table]:my-4 [&_table]:w-full [&_table]:border-collapse [&_table]:text-sm [&_table]:tracking-normal",
          "[&_th]:border [&_th]:border-border/50 [&_th]:bg-muted/30 [&_th]:px-3 [&_th]:py-1.5 [&_th]:text-left [&_th]:font-medium",
          "[&_td]:border [&_td]:border-border/50 [&_td]:px-3 [&_td]:py-1.5",
          "[&_hr]:my-6 [&_hr]:border-border/50",
        )}
        ref={containerRef}
      >
        <Markdown
          components={{
            a: ({ href, children, ...props }) => {
              const dashboardLink = parseDashboardLink(href);
              if (dashboardLink?.kind === "trace") {
                return (
                  <button
                    className="inline rounded bg-detail-brand/10 px-1 py-0.5 font-mono text-[0.85em] tracking-normal text-link transition hover:bg-detail-brand/15 hover:underline"
                    data-halo-report-link="trace"
                    data-trace-id={dashboardLink.traceId}
                    onClick={() => onOpenTraceLink?.(dashboardLink.traceId)}
                    type="button"
                  >
                    {dashboardLinkLabel(dashboardLink)}
                  </button>
                );
              }
              if (dashboardLink?.kind === "span") {
                return (
                  <button
                    className="inline rounded bg-detail-brand/10 px-1 py-0.5 font-mono text-[0.85em] tracking-normal text-link transition hover:bg-detail-brand/15 hover:underline"
                    data-halo-report-link="span"
                    data-span-id={dashboardLink.spanId}
                    data-trace-id={dashboardLink.traceId}
                    onClick={() =>
                      onOpenSpanLink?.(dashboardLink.traceId, dashboardLink.spanId)
                    }
                    type="button"
                  >
                    {dashboardLinkLabel(dashboardLink)}
                  </button>
                );
              }
              return (
                <a href={href} {...props}>
                  {children}
                </a>
              );
            },
            h1: (props) => <h1 data-heading-slug={slugify(textOf(props.children))} {...props} />,
            h2: (props) => <h2 data-heading-slug={slugify(textOf(props.children))} {...props} />,
            h3: (props) => <h3 data-heading-slug={slugify(textOf(props.children))} {...props} />,
          }}
          remarkPlugins={[remarkGfm]}
        >
          {linkifiedMarkdown}
        </Markdown>
      </div>
    </div>
  );
}

function extractHeadings(markdown: string) {
  const seen = new Set<string>();
  const headings: Array<{ slug: string; text: string }> = [];
  let inFence = false;
  for (const line of markdown.split("\n")) {
    if (line.trimStart().startsWith("```")) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;
    const match = /^(#{1,3})\s+(.+)$/.exec(line.trim());
    if (!match?.[2]) continue;
    const text = match[2].replace(/[#*`_]/g, "").trim();
    const slug = slugify(text);
    if (!text || seen.has(slug)) continue;
    seen.add(slug);
    headings.push({ slug, text });
  }
  return headings;
}

function textOf(children: unknown): string {
  if (typeof children === "string") return children;
  if (Array.isArray(children)) return children.map(textOf).join("");
  if (children && typeof children === "object" && "props" in children) {
    return textOf((children as { props: { children?: unknown } }).props.children);
  }
  return "";
}

function slugify(text: string) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-");
}
