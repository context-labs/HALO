import type { ReactNode } from "react";
import {
  Bell,
  Component as ComponentIcon,
  GalleryVerticalEnd,
  Palette,
  Search,
} from "lucide-react";
import { Link } from "@tanstack/react-router";

import {
  Button,
  InferenceIcon,
  Input,
  Row,
  ThemeToggle,
  toast,
} from "~/lib/ui";

import {
  componentCount,
  gallerySections,
  type IconComponent,
  type SectionId,
} from "./GalleryPage";
import { useGallerySearch } from "./search";

export type GalleryRoutePath =
  | "/components"
  | "/components/foundations"
  | "/components/inputs"
  | "/components/overlays"
  | "/components/navigation"
  | "/components/custom";

const galleryRoutePaths = [
  "/components",
  "/components/foundations",
  "/components/inputs",
  "/components/overlays",
  "/components/navigation",
  "/components/custom",
] satisfies GalleryRoutePath[];

export function toGalleryRoutePath(pathname: string): GalleryRoutePath {
  return galleryRoutePaths.includes(pathname as GalleryRoutePath)
    ? (pathname as GalleryRoutePath)
    : "/components";
}

const sectionPaths: Record<SectionId, GalleryRoutePath> = {
  custom: "/components/custom",
  foundations: "/components/foundations",
  inputs: "/components/inputs",
  navigation: "/components/navigation",
  overlays: "/components/overlays",
};

export function GalleryShell({ children }: { children: ReactNode }) {
  const { query, setQuery } = useGallerySearch();

  return (
    <main className="min-h-screen bg-background text-foreground">
      <div className="electrobun-webkit-app-region-drag grid h-14 select-none grid-cols-[76px_1fr] border-b border-subtle bg-background/95 backdrop-blur">
        <div className="border-r border-subtle" />
        <div className="flex min-w-0 items-center justify-between gap-4 px-5 pl-20">
          <Row className="min-w-0 items-center gap-3">
            <div className="grid h-7 w-7 shrink-0 place-items-center rounded-md border border-subtle bg-card">
              <ComponentIcon className="h-4 w-4" />
            </div>
            <div className="min-w-0">
              <p className="text-xs font-semibold text-muted-foreground">
                HALO
              </p>
              <p className="truncate text-sm font-semibold">
                Tailwind UI Gallery
              </p>
            </div>
          </Row>

          <Row className="electrobun-webkit-app-region-no-drag min-w-0 items-center gap-2">
            <Input
              aria-label="Search components"
              className="h-9"
              containerClassname="hidden w-64 md:block xl:w-80"
              icon={<Search className="h-4 w-4 text-muted-foreground" />}
              onChange={(event) => setQuery(event.currentTarget.value)}
              placeholder="Search components"
              value={query}
            />
            <ThemeToggle
              trigger={
                <Button size="sm" variant="outline">
                  <Palette className="mr-2 h-4 w-4" />
                  Theme
                </Button>
              }
            />
            <Button
              size="sm"
              variant="secondary"
              onClick={() =>
                toast.info({
                  title: "Gallery ready",
                  description: "The copied UI library is rendering locally.",
                })
              }
            >
              <Bell className="mr-2 h-4 w-4" />
              Ping
            </Button>
          </Row>
        </div>
      </div>

      <div className="grid min-h-[calc(100vh-3.5rem)] grid-cols-[76px_1fr] lg:grid-cols-[260px_1fr]">
        <aside className="border-r border-subtle bg-sidebar">
          <div className="flex h-full flex-col gap-4 p-3">
            <div className="hidden px-2 pt-2 lg:block">
              <InferenceIcon width={172} height={24} />
            </div>
            <nav className="space-y-1">
              <GalleryNavLink
                icon={GalleryVerticalEnd}
                label="All"
                to="/components"
              />
              {gallerySections.map((section) => (
                <GalleryNavLink
                  icon={section.icon}
                  key={section.id}
                  label={section.label}
                  to={sectionPaths[section.id]}
                />
              ))}
            </nav>
            <div className="mt-auto hidden rounded-xl border border-subtle bg-background p-3 lg:block">
              <p className="text-sm font-semibold">{componentCount} exports</p>
              <p className="mt-1 text-sm text-muted-foreground">
                File-based routes powered by TanStack Start.
              </p>
            </div>
          </div>
        </aside>

        <section className="min-w-0 overflow-auto">{children}</section>
      </div>
    </main>
  );
}

function GalleryNavLink({
  icon: Icon,
  label,
  to,
}: {
  icon: IconComponent;
  label: string;
  to: GalleryRoutePath;
}) {
  return (
    <Link
      activeOptions={{ exact: to === "/components" }}
      activeProps={{
        className: "border-subtle bg-background text-foreground shadow-sm",
      }}
      className="electrobun-webkit-app-region-no-drag flex h-11 w-full items-center justify-center rounded-lg border text-sm font-medium transition lg:justify-start lg:gap-3 lg:px-3"
      inactiveProps={{
        className:
          "border-transparent text-muted-foreground hover:bg-background/70 hover:text-foreground",
      }}
      to={to}
    >
      <Icon className="h-4 w-4" />
      <span className="hidden lg:inline">{label}</span>
    </Link>
  );
}
