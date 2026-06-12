import type { ReactNode } from "react";
import { Link } from "@tanstack/react-router";
import { Palette } from "lucide-react";

import { Button, InferenceIcon, ThemeToggle } from "~/lib/ui";
import { isDesktopShell } from "~/desktop/desktopBridge";

/**
 * The single fixed window header used by every page.
 *
 * The outer div is the ElectroBun window drag region (titleBarStyle
 * hiddenInset); interactive children must live inside the no-drag wrapper.
 * In the desktop shell the left cell stays empty — it's the macOS traffic
 * lights' zone and the wordmark moves below it into WorkspaceNav. In a plain
 * browser there are no traffic lights, so the wordmark lives here.
 */
export function AppHeader({
  actions,
  description,
  icon,
  status,
  title,
}: {
  actions?: ReactNode;
  description?: string;
  icon?: ReactNode;
  status?: ReactNode;
  title: string;
}) {
  return (
    <div className="electrobun-webkit-app-region-drag fixed inset-x-0 top-0 z-40 grid h-14 select-none grid-cols-[14rem_minmax(0,1fr)]">
      {isDesktopShell() ? (
        <div className="h-14 border-r border-border/50 bg-sidebar" />
      ) : (
        <div className="flex h-14 items-center border-r border-border/50 bg-sidebar px-5">
          <Link
            className="electrobun-webkit-app-region-no-drag"
            search={{} as never}
            to="/traces"
          >
            <InferenceIcon height={20} width={120} />
          </Link>
        </div>
      )}
      <div className="flex min-w-0 items-center justify-between gap-4 border-b border-border/50 bg-sidebar px-6">
        <div className="flex min-w-0 items-center gap-3">
          {icon ? (
            <div className="grid h-8 w-8 shrink-0 place-items-center rounded-md border border-subtle bg-card">
              {icon}
            </div>
          ) : null}
          <div className="min-w-0">
            <p className="text-xs font-medium text-muted-foreground">HALO</p>
            <p className="truncate text-sm font-semibold">{title}</p>
          </div>
          {description ? (
            <span className="hidden truncate text-xs text-muted-foreground md:block">
              {description}
            </span>
          ) : null}
        </div>

        <div className="electrobun-webkit-app-region-no-drag flex min-w-0 shrink-0 items-center gap-2">
          {status}
          {actions}
          <ThemeToggle
            trigger={
              <Button
                aria-label="Change theme"
                size="icon"
                variant="ghost"
              >
                <Palette className="h-4 w-4" />
              </Button>
            }
          />
        </div>
      </div>
    </div>
  );
}
