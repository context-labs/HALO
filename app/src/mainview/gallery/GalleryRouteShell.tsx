import { useState, type ReactNode } from "react";

import { GalleryShell } from "./GalleryShell";
import { GallerySearchProvider } from "./search";

export function GalleryRouteShell({ children }: { children: ReactNode }) {
  const [query, setQuery] = useState("");

  return (
    <GallerySearchProvider
      value={{
        clearQuery: () => setQuery(""),
        query,
        setQuery,
      }}
    >
      <GalleryShell>{children}</GalleryShell>
    </GallerySearchProvider>
  );
}
