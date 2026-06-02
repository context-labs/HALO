import {
  createHashHistory,
  createMemoryHistory,
  createRouter,
} from "@tanstack/react-router";

import { routeTree } from "./routeTree.gen";

export function getRouter() {
  return createRouter({
    defaultPreload: "intent",
    history:
      typeof window === "undefined"
        ? createMemoryHistory({ initialEntries: ["/"] })
        : createHashHistory(),
    routeTree,
    scrollRestoration: true,
  });
}

declare module "@tanstack/react-router" {
  interface Register {
    router: ReturnType<typeof getRouter>;
  }
}
