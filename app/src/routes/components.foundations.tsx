import { createFileRoute } from "@tanstack/react-router";

import { GalleryPage } from "~/gallery/GalleryPage";
import { GalleryRouteShell } from "~/gallery/GalleryRouteShell";

export const Route = createFileRoute("/components/foundations")({
  component: FoundationsRoute,
});

function FoundationsRoute() {
  return (
    <GalleryRouteShell>
      <GalleryPage sectionId="foundations" />
    </GalleryRouteShell>
  );
}
