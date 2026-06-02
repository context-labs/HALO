import { createFileRoute } from "@tanstack/react-router";

import { GalleryPage } from "~/gallery/GalleryPage";
import { GalleryRouteShell } from "~/gallery/GalleryRouteShell";

export const Route = createFileRoute("/components/inputs")({
  component: InputsRoute,
});

function InputsRoute() {
  return (
    <GalleryRouteShell>
      <GalleryPage sectionId="inputs" />
    </GalleryRouteShell>
  );
}
