import { createFileRoute } from "@tanstack/react-router";

import { GalleryPage } from "~/gallery/GalleryPage";
import { GalleryRouteShell } from "~/gallery/GalleryRouteShell";

export const Route = createFileRoute("/components")({
  component: ComponentsRoute,
});

function ComponentsRoute() {
  return (
    <GalleryRouteShell>
      <GalleryPage />
    </GalleryRouteShell>
  );
}
