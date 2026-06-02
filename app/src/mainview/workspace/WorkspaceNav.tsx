import { Link } from "@tanstack/react-router";
import { Activity, BrainCircuit, MessageSquare, Settings } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "~/lib/ui";

export type WorkspaceSection = "traces" | "sessions" | "analysis" | "settings";

const navItems: Array<{
  id: WorkspaceSection;
  icon: ReactNode;
  label: string;
  to: "/traces" | "/sessions" | "/analysis" | "/settings";
}> = [
  {
    id: "traces",
    icon: <Activity className="h-4 w-4" />,
    label: "Traces",
    to: "/traces",
  },
  {
    id: "sessions",
    icon: <MessageSquare className="h-4 w-4" />,
    label: "Sessions",
    to: "/sessions",
  },
  {
    id: "analysis",
    icon: <BrainCircuit className="h-4 w-4" />,
    label: "Analysis",
    to: "/analysis",
  },
  {
    id: "settings",
    icon: <Settings className="h-4 w-4" />,
    label: "Settings",
    to: "/settings",
  },
];

export function WorkspaceNav({ active }: { active: WorkspaceSection }) {
  return (
    <aside className="border-r border-subtle bg-background">
      <nav className="flex h-full flex-col gap-2 p-3">
        {navItems.map((item) => (
          <WorkspaceNavLink
            active={active === item.id}
            icon={item.icon}
            key={item.id}
            label={item.label}
            to={item.to}
          />
        ))}
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
  to: "/traces" | "/sessions" | "/analysis" | "/settings";
}) {
  return (
    <Link
      className={cn(
        "electrobun-webkit-app-region-no-drag flex min-h-16 flex-col items-center justify-center gap-1 rounded-md border border-transparent px-2 py-3 text-xs font-medium text-muted-foreground transition hover:border-subtle hover:bg-muted hover:text-foreground",
        active && "border-detail-brand/30 bg-detail-brand/10 text-detail-brand",
      )}
      search={{} as never}
      to={to}
    >
      {icon}
      {label}
    </Link>
  );
}
