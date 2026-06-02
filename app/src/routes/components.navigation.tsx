import { createFileRoute } from "@tanstack/react-router";

import { GalleryPage } from "~/gallery/GalleryPage";
import { GalleryRouteShell } from "~/gallery/GalleryRouteShell";

export const Route = createFileRoute("/components/navigation")({
  component: NavigationRoute,
});

function NavigationRoute() {
  return (
    <GalleryRouteShell>
      <GalleryPage sectionId="navigation" />
    </GalleryRouteShell>
  );
}
