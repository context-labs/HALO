import { createFileRoute } from "@tanstack/react-router";

import { GalleryPage } from "~/gallery/GalleryPage";
import { GalleryRouteShell } from "~/gallery/GalleryRouteShell";

export const Route = createFileRoute("/components/custom")({
  component: CustomRoute,
});

function CustomRoute() {
  return (
    <GalleryRouteShell>
      <GalleryPage sectionId="custom" />
    </GalleryRouteShell>
  );
}
