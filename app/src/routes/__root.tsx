/// <reference types="vite/client" />

import { useEffect, useState, type ReactNode } from "react";
import {
  createRootRoute,
  HeadContent,
  Outlet,
  Scripts,
} from "@tanstack/react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  createWSClient,
  httpBatchLink,
  splitLink,
  wsLink,
} from "@trpc/client";

import { InferenceIcon, ThemeProvider, Toaster } from "~/lib/ui";
import { trpc } from "~/trpc";

import appCss from "../mainview/styles.css?url";

const TRPC_HTTP_URL =
  import.meta.env.VITE_TRPC_HTTP_URL ?? "http://127.0.0.1:8799/trpc";
const TRPC_WS_URL = import.meta.env.VITE_TRPC_WS_URL ?? "ws://127.0.0.1:8800";

export const Route = createRootRoute({
  head: () => ({
    links: [
      { rel: "stylesheet", href: appCss },
      {
        rel: "icon",
        href: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%230a0a0a'/%3E%3Cpath d='M9 22V10M16 22V10M23 22V10' stroke='%23ffffff' stroke-width='3' stroke-linecap='round'/%3E%3Cpath d='M9 10v12' stroke='%2353B1FD' stroke-width='3' stroke-linecap='round'/%3E%3Cpath d='M16 10v12' stroke='%23FAC515' stroke-width='3' stroke-linecap='round'/%3E%3Cpath d='M23 10v12' stroke='%23FF4405' stroke-width='3' stroke-linecap='round'/%3E%3C/svg%3E",
      },
    ],
    meta: [
      { charSet: "utf-8" },
      {
        name: "viewport",
        content: "width=device-width, initial-scale=1",
      },
      { title: "HALO" },
    ],
  }),
  component: RootComponent,
});

const safeThemeStorage: Storage = {
  get length() {
    return typeof window === "undefined" ? 0 : window.localStorage.length;
  },
  clear() {
    if (typeof window !== "undefined") {
      window.localStorage.clear();
    }
  },
  getItem(key) {
    return typeof window === "undefined" ? null : window.localStorage.getItem(key);
  },
  key(index) {
    return typeof window === "undefined" ? null : window.localStorage.key(index);
  },
  removeItem(key) {
    if (typeof window !== "undefined") {
      window.localStorage.removeItem(key);
    }
  },
  setItem(key, value) {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(key, value);
    }
  },
};

function RootComponent() {
  const [queryClient] = useState(() => new QueryClient());
  const [trpcClient] = useState(() => {
    const wsClient = createWSClient({
      lazy: {
        closeMs: 10_000,
        enabled: true,
      },
      url: TRPC_WS_URL,
    });
    return trpc.createClient({
      links: [
        splitLink({
          condition: (op) => op.type === "subscription",
          false: httpBatchLink({
            url: TRPC_HTTP_URL,
          }),
          true: wsLink({
            client: wsClient,
          }),
        }),
      ],
    });
  });

  return (
    <RootDocument>
      <trpc.Provider client={trpcClient} queryClient={queryClient}>
        <QueryClientProvider client={queryClient}>
          <ThemeProvider storage={safeThemeStorage} storageKey="halo-canvas-theme">
            <StartupTransition>
              <Outlet />
            </StartupTransition>
            <Toaster />
          </ThemeProvider>
        </QueryClientProvider>
      </trpc.Provider>
    </RootDocument>
  );
}

const SPLASH_VISIBLE_MS = 500;
const SPLASH_EXIT_MS = 220;
const DASHBOARD_ENTER_MS = 260;

type StartupPhase = "splash" | "exiting" | "entering" | "visible";
type StartupWindow = Window & {
  __haloReactSplashSeen?: boolean;
};

function StartupTransition({ children }: Readonly<{ children: ReactNode }>) {
  const [phase, setPhase] = useState<StartupPhase>("splash");

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const startupWindow = window as StartupWindow;
    if (startupWindow.__haloReactSplashSeen) {
      setPhase("visible");
      return;
    }

    startupWindow.__haloReactSplashSeen = true;
    const exitTimer = setTimeout(() => setPhase("exiting"), SPLASH_VISIBLE_MS);
    const enterTimer = setTimeout(
      () => setPhase("entering"),
      SPLASH_VISIBLE_MS + SPLASH_EXIT_MS,
    );
    const visibleTimer = setTimeout(
      () => setPhase("visible"),
      SPLASH_VISIBLE_MS + SPLASH_EXIT_MS + DASHBOARD_ENTER_MS,
    );

    return () => {
      clearTimeout(exitTimer);
      clearTimeout(enterTimer);
      clearTimeout(visibleTimer);
    };
  }, []);

  const showSplash = phase === "splash" || phase === "exiting";

  return (
    <div
      aria-busy={showSplash}
      className="halo-startup-shell"
    >
      <div
        aria-hidden={showSplash ? true : undefined}
        className={
          phase === "entering"
            ? "halo-startup-content halo-startup-content-entering"
            : phase === "visible"
              ? "halo-startup-content halo-startup-content-visible"
              : "halo-startup-content halo-startup-content-pending"
        }
      >
        {children}
      </div>
      {showSplash ? (
        <div
          className={
            phase === "exiting"
              ? "halo-react-splash halo-react-splash-exiting"
              : "halo-react-splash"
          }
          role="status"
        >
          <div className="halo-react-splash-logo dark">
            <InferenceIcon height="auto" width="100%" />
          </div>
        </div>
      ) : null}
    </div>
  );
}

function RootDocument({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <head>
        <HeadContent />
      </head>
      <body>
        {children}
        <Scripts />
      </body>
    </html>
  );
}
