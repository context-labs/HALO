import {
  createContext,
  useContext,
  type ReactNode,
} from "react";

export type GallerySearch = {
  q?: string;
};

type GallerySearchContextValue = {
  clearQuery: () => void;
  query: string;
  setQuery: (query: string) => void;
};

const GallerySearchContext =
  createContext<GallerySearchContextValue | null>(null);

export function normalizeGallerySearch(
  search: Record<string, unknown>,
): GallerySearch {
  const q = typeof search.q === "string" ? search.q.trim() : "";

  return q ? { q } : {};
}

export function GallerySearchProvider({
  children,
  value,
}: {
  children: ReactNode;
  value: GallerySearchContextValue;
}) {
  return (
    <GallerySearchContext.Provider value={value}>
      {children}
    </GallerySearchContext.Provider>
  );
}

export function useGallerySearch(): GallerySearchContextValue {
  const context = useContext(GallerySearchContext);

  if (!context) {
    throw new Error("useGallerySearch must be used within GallerySearchProvider");
  }

  return context;
}
