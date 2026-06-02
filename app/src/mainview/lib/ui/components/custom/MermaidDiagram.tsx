import type { ClassNameValue } from "tailwind-merge";
import { useCallback, useEffect, useId, useRef, useState } from "react";
import {
  AlertCircle,
  Code2,
  Copy,
  Download,
  Maximize2,
  Minimize2,
} from "lucide-react";
import { useCopyToClipboard } from "usehooks-ts";

import { Button } from "~/lib/ui/components/ui/Button";
import { Card } from "~/lib/ui/components/ui/Card";
import { Skeleton } from "~/lib/ui/components/ui/Skeleton";
import { Tooltip } from "~/lib/ui/components/ui/Tooltip";
import { toast } from "~/lib/ui/hooks/useToast.hook";
import { useTheme } from "~/lib/ui/providers/ThemeProvider";
import { getMermaidConfig } from "~/lib/ui/utils/mermaidTheme";
import { cn } from "~/lib/ui/utils/utils";

/**
 * Fix Mermaid SVG viewBox to prevent subgraph labels from being clipped.
 * Mermaid calculates the viewBox before rendering subgraph/cluster titles,
 * which causes them to be cut off at the top of their containers.
 */
function fixMermaidViewBox(svg: string): string {
  // Parse the viewBox
  const viewBoxMatch = /viewBox="([^"]+)"/.exec(svg);
  if (!viewBoxMatch?.[1]) return svg;

  const parts = viewBoxMatch[1].split(/[\s,]+/).map(Number);
  const x = parts[0] ?? 0;
  const y = parts[1] ?? 0;
  const width = parts[2] ?? 100;
  const height = parts[3] ?? 100;

  // Count how many subgraphs/clusters exist - more subgraphs may need more padding
  const clusterCount = (svg.match(/class="cluster"/g) ?? []).length;
  const hasSubgraphs = clusterCount > 0;

  // Add extra padding - more aggressive for diagrams with subgraphs
  // Each nested level of subgraphs can add to clipping issues
  const topPadding = hasSubgraphs ? Math.max(50, clusterCount * 15) : 20;
  const sidePadding = hasSubgraphs ? 40 : 20;
  const bottomPadding = 20;

  const newViewBox = `${x - sidePadding} ${y - topPadding} ${width + sidePadding * 2} ${height + topPadding + bottomPadding}`;

  // Also remove any max-width inline styles that might cause issues
  let fixedSvg = svg.replace(/viewBox="[^"]+"/, `viewBox="${newViewBox}"`);
  fixedSvg = fixedSvg.replace(/style="[^"]*max-width:\s*[^;]+;?/g, 'style="');

  return fixedSvg;
}

type MermaidDiagramProps = {
  /** The mermaid diagram code */
  code: string;
  /** Optional className for the container */
  className?: ClassNameValue;
  /** Optional caption to display below the diagram */
  caption?: string;
  /** Whether to show the code toggle button */
  showCodeToggle?: boolean;
};

type RenderState =
  | { status: "loading" }
  | { status: "success"; svg: string }
  | { status: "error"; message: string };

/**
 * A polished Mermaid diagram renderer with:
 * - Theme-aware rendering (auto-switches with light/dark mode)
 * - Loading skeleton state
 * - Error handling with fallback to code view
 * - Copy code to clipboard
 * - Download as SVG
 * - Fullscreen view for complex diagrams
 * - Smooth animations
 */
export function MermaidDiagram({
  code,
  className,
  caption,
  showCodeToggle = true,
}: MermaidDiagramProps) {
  const uniqueId = useId();
  const diagramId = `mermaid-${uniqueId.replace(/:/g, "-")}`;
  const containerRef = useRef<HTMLDivElement>(null);

  const { isDarkTheme } = useTheme();
  const [, copyToClipboard] = useCopyToClipboard();

  const [renderState, setRenderState] = useState<RenderState>({
    status: "loading",
  });
  const [showCode, setShowCode] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Render the mermaid diagram
  const renderDiagram = useCallback(async () => {
    setRenderState({ status: "loading" });

    try {
      // Dynamically import mermaid to avoid SSR issues
      const mermaid = (await import("mermaid")).default;

      // Initialize with theme-aware config
      const config = getMermaidConfig(isDarkTheme);
      mermaid.initialize(config);

      // Render the diagram
      const { svg } = await mermaid.render(diagramId, code.trim());

      // Post-process SVG to fix subgraph label clipping
      // Mermaid calculates viewBox before rendering subgraph titles, causing them to be cut off
      // We expand the viewBox to ensure all content is visible
      const fixedSvg = fixMermaidViewBox(svg);

      setRenderState({ status: "success", svg: fixedSvg });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to render diagram";
      setRenderState({ status: "error", message });
    }
  }, [code, diagramId, isDarkTheme]);

  // Re-render when code or theme changes
  useEffect(() => {
    void renderDiagram();
  }, [renderDiagram]);

  // Handle copy code
  const handleCopyCode = useCallback(() => {
    void copyToClipboard(code);
    toast.success({
      title: "Copied",
      description: "Mermaid code copied to clipboard",
    });
  }, [code, copyToClipboard]);

  // Handle download SVG
  const handleDownloadSvg = useCallback(() => {
    if (renderState.status === "success") {
      const blob = new Blob([renderState.svg], { type: "image/svg+xml" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "diagram.svg";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success({
        title: "Downloaded",
        description: "SVG file downloaded",
      });
    }
  }, [renderState]);

  // Handle fullscreen toggle
  const toggleFullscreen = useCallback(() => {
    setIsFullscreen((prev) => !prev);
  }, []);

  // Handle escape key to exit fullscreen
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isFullscreen) {
        setIsFullscreen(false);
      }
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [isFullscreen]);

  return (
    <>
      {/* Fullscreen overlay */}
      {isFullscreen && (
        <div
          className={`
            fixed inset-0 z-50 flex items-center justify-center bg-background/95
            backdrop-blur-xs
          `}
          onClick={toggleFullscreen}
        >
          <div
            className={`
              relative flex h-[90vh] w-[90vw] flex-col overflow-hidden
              rounded-lg border bg-card shadow-2xl
            `}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header with close button */}
            <div className="flex items-center justify-end border-b px-4 py-3">
              <Button
                aria-label="Exit fullscreen"
                onClick={toggleFullscreen}
                size="icon"
                variant="ghost"
              >
                <Minimize2 className="size-4" />
              </Button>
            </div>
            {/* Diagram container */}
            {renderState.status === "success" && (
              <div
                className={`
                  mermaid-diagram-fullscreen flex flex-1 items-center
                  justify-center overflow-auto p-8
                `}
                dangerouslySetInnerHTML={{ __html: renderState.svg }}
              />
            )}
          </div>
        </div>
      )}

      {/* Main container */}
      <Card
        className={cn(
          `
            group relative mb-4 overflow-hidden transition-all duration-200

            dark:bg-stone-950
          `,
          className,
        )}
      >
        {/* Toolbar */}
        <div
          className={`
            absolute right-2 top-2 z-10 flex gap-1 opacity-0 transition-opacity

            group-hover:opacity-100
          `}
        >
          {showCodeToggle && (
            <Tooltip content={showCode ? "Show diagram" : "Show code"}>
              <Button
                aria-label={showCode ? "Show diagram" : "Show code"}
                className="bg-secondary/80"
                onClick={() => setShowCode(!showCode)}
                size="icon"
                variant="ghost"
              >
                <Code2
                  className={`
                    size-4 text-muted-foreground

                    hover:text-foreground
                  `}
                />
              </Button>
            </Tooltip>
          )}
          <Tooltip content="Copy code">
            <Button
              aria-label="Copy code"
              className="bg-secondary/80"
              onClick={handleCopyCode}
              size="icon"
              variant="ghost"
            >
              <Copy
                className={`
                  size-4 text-muted-foreground

                  hover:text-foreground
                `}
              />
            </Button>
          </Tooltip>
          {renderState.status === "success" && (
            <>
              <Tooltip content="Download SVG">
                <Button
                  aria-label="Download SVG"
                  className="bg-secondary/80"
                  onClick={handleDownloadSvg}
                  size="icon"
                  variant="ghost"
                >
                  <Download
                    className={`
                      size-4 text-muted-foreground

                      hover:text-foreground
                    `}
                  />
                </Button>
              </Tooltip>
              <Tooltip content="Fullscreen">
                <Button
                  aria-label="View fullscreen"
                  className="bg-secondary/80"
                  onClick={toggleFullscreen}
                  size="icon"
                  variant="ghost"
                >
                  <Maximize2
                    className={`
                      size-4 text-muted-foreground

                      hover:text-foreground
                    `}
                  />
                </Button>
              </Tooltip>
            </>
          )}
        </div>

        {/* Content */}
        <div className="p-4" ref={containerRef}>
          {showCode ? (
            // Code view
            <pre
              className={`
                overflow-x-auto rounded-md bg-muted p-4 font-berkeley text-sm
                text-foreground
              `}
            >
              <code>{code}</code>
            </pre>
          ) : renderState.status === "loading" ? (
            // Loading skeleton
            <div
              className={`
              flex flex-col items-center justify-center gap-4 py-8
            `}
            >
              <Skeleton className="h-32 w-full max-w-md" />
              <Skeleton className="h-4 w-48" />
            </div>
          ) : renderState.status === "error" ? (
            // Error state
            <div className="flex flex-col items-center gap-4 py-8 text-center">
              <div
                className={`flex items-center gap-2 text-sm text-destructive`}
              >
                <AlertCircle className="size-4" />
                <span>Failed to render diagram</span>
              </div>
              <p className="max-w-md text-xs text-muted-foreground">
                {renderState.message}
              </p>
              <Button
                onClick={() => setShowCode(true)}
                size="sm"
                variant="outline"
              >
                View code
              </Button>
            </div>
          ) : (
            // Success - render SVG
            <div
              className={`
                mermaid-diagram flex items-center justify-center overflow-x-auto
                transition-opacity duration-300
              `}
              dangerouslySetInnerHTML={{ __html: renderState.svg }}
            />
          )}
        </div>

        {/* Caption */}
        {caption && (
          <div className="border-t px-4 py-2">
            <p className="text-center text-sm text-muted-foreground">
              {caption}
            </p>
          </div>
        )}
      </Card>
    </>
  );
}
