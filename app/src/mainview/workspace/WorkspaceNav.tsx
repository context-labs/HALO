import { Link } from "@tanstack/react-router";
import { Activity, BrainCircuit, DownloadCloud, Settings } from "lucide-react";
import type { ReactNode } from "react";

import { InferenceIcon, cn } from "~/lib/ui";
import { isDesktopShell } from "~/desktop/desktopBridge";

export type WorkspaceSection = "traces" | "analysis" | "imports" | "settings";

const navItems: Array<{
  id: WorkspaceSection;
  icon: ReactNode;
  label: string;
  to: "/traces" | "/analysis" | "/imports" | "/settings";
}> = [
  {
    id: "traces",
    icon: <Activity className="h-4 w-4" strokeWidth={1.5} />,
    label: "Traces",
    to: "/traces",
  },
  {
    id: "analysis",
    icon: <BrainCircuit className="h-4 w-4" strokeWidth={1.5} />,
    label: "Analysis",
    to: "/analysis",
  },
  {
    id: "imports",
    icon: <DownloadCloud className="h-4 w-4" strokeWidth={1.5} />,
    label: "Imports",
    to: "/imports",
  },
  {
    id: "settings",
    icon: <Settings className="h-4 w-4" strokeWidth={1.5} />,
    label: "Settings",
    to: "/settings",
  },
];

export function WorkspaceNav({ active }: { active: WorkspaceSection }) {
  return (
    <aside className="flex flex-col border-r border-border/50 bg-sidebar">
      {/* In the desktop shell the brand sits below the macOS traffic lights
          (the empty header strip above), aligned with the nav item icons. In
          a browser the wordmark stays in the AppHeader instead. */}
      {isDesktopShell() ? (
        <div className="flex-none px-6 pb-3">
          <Link
            className="electrobun-webkit-app-region-no-drag inline-flex"
            search={{} as never}
            to="/traces"
          >
            <InferenceIcon height={20} width={120} />
          </Link>
        </div>
      ) : null}
      <nav className="relative flex min-h-0 flex-1 overflow-y-auto pb-2">
        <ul className="w-full">
          {navItems.map((item) => (
            <WorkspaceNavLink
              active={active === item.id}
              icon={item.icon}
              key={item.id}
              label={item.label}
              to={item.to}
            />
          ))}
        </ul>
      </nav>
    </aside>
  );
}

function WorkspaceNavLink({
  active,
  icon,
  label,
  to,
}: {
  active: boolean;
  icon: ReactNode;
  label: string;
  to: "/traces" | "/analysis" | "/imports" | "/settings";
}) {
  return (
    <li className="px-3 py-px">
      <Link
        className={cn(
          "electrobun-webkit-app-region-no-drag flex h-9 items-center gap-3 rounded-md px-3 text-sm font-medium text-foreground hover:bg-accent hover:text-foreground",
          active && "bg-accent text-foreground",
        )}
        search={{} as never}
        to={to}
      >
        <span className="shrink-0">{icon}</span>
        <span className="truncate">{label}</span>
      </Link>
    </li>
  );
}
