import { createFileRoute } from "@tanstack/react-router";

import { GalleryPage } from "~/gallery/GalleryPage";
import { GalleryRouteShell } from "~/gallery/GalleryRouteShell";

export const Route = createFileRoute("/components/overlays")({
  component: OverlaysRoute,
});

function OverlaysRoute() {
  return (
    <GalleryRouteShell>
      <GalleryPage sectionId="overlays" />
    </GalleryRouteShell>
  );
}
