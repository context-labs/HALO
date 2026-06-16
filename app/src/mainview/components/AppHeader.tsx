import type { ReactNode } from "react";
import { Link } from "@tanstack/react-router";
import { BookOpen } from "lucide-react";

import { Button, InferenceIcon } from "~/lib/ui";
import { APP_DOCS_URL, APP_GITHUB_URL } from "../../desktop/commands";
import { isDesktopShell, openExternalUrl } from "~/desktop/desktopBridge";

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
          <HeaderResourceButton
            href={APP_DOCS_URL}
            icon={<BookOpen className="h-4 w-4" />}
            label="Read Docs"
          />
          <HeaderResourceButton
            href={APP_GITHUB_URL}
            icon={<GitHubMark className="h-4 w-4" />}
            label="GitHub"
          />
          {status}
          {actions}
        </div>
      </div>
    </div>
  );
}

function GitHubMark({ className }: { className?: string }) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="currentColor"
      viewBox="0 0 24 24"
    >
      <path d="M12 2C6.477 2 2 6.486 2 12.02c0 4.43 2.865 8.185 6.839 9.513.5.092.682-.217.682-.482 0-.237-.009-.866-.014-1.7-2.782.605-3.369-1.343-3.369-1.343-.455-1.158-1.11-1.467-1.11-1.467-.908-.62.069-.608.069-.608 1.004.071 1.532 1.033 1.532 1.033.892 1.53 2.341 1.088 2.91.832.091-.647.35-1.088.636-1.338-2.221-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.987 1.03-2.687-.103-.253-.447-1.27.098-2.647 0 0 .84-.27 2.75 1.026A9.56 9.56 0 0 1 12 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.026 2.747-1.026.546 1.377.202 2.394.1 2.647.64.7 1.028 1.594 1.028 2.687 0 3.848-2.338 4.695-4.566 4.943.36.31.68.923.68 1.86 0 1.343-.012 2.427-.012 2.757 0 .267.18.578.688.48C19.138 20.2 22 16.448 22 12.02 22 6.486 17.523 2 12 2Z" />
    </svg>
  );
}

function HeaderResourceButton({
  href,
  icon,
  label,
}: {
  href: string;
  icon: ReactNode;
  label: string;
}) {
  return (
    <Button
      className="gap-1.5"
      onClick={() => void openExternalUrl(href)}
      size="sm"
      type="button"
      variant="outline"
    >
      {icon}
      {label}
    </Button>
  );
}
