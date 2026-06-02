import { createFileRoute } from "@tanstack/react-router";

import { SessionsPage } from "~/tracing/TraceMonitorPage";

export const Route = createFileRoute("/sessions")({
  component: SessionsRoute,
  validateSearch: (search) => ({
    sessionId:
      typeof search.sessionId === "string" ? search.sessionId : undefined,
  }),
});

function SessionsRoute() {
  const navigate = Route.useNavigate();
  const search = Route.useSearch();

  return (
    <SessionsPage
      onSelectSession={(sessionId) => {
        void navigate({
          search: {
            sessionId: sessionId ?? undefined,
          },
        });
      }}
      selectedSessionId={search.sessionId}
    />
  );
}
