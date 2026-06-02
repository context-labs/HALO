import { createFileRoute } from "@tanstack/react-router";

import { TraceMonitorPage } from "~/tracing/TraceMonitorPage";

export const Route = createFileRoute("/")({
  component: TraceMonitorRoute,
  validateSearch: (search) => ({
    followLatest:
      search.followLatest === 1 || search.followLatest === "1" ? 1 : undefined,
    traceId: typeof search.traceId === "string" ? search.traceId : undefined,
  }),
});

function TraceMonitorRoute() {
  const navigate = Route.useNavigate();
  const search = Route.useSearch();

  return (
    <TraceMonitorPage
      followLatest={search.followLatest === 1}
      onFollowLatestChange={(followLatest) => {
        void navigate({
          search: {
            followLatest: followLatest ? 1 : undefined,
            traceId: search.traceId,
          },
        });
      }}
      onSelectLatestTrace={(traceId) => {
        void navigate({
          replace: true,
          search: {
            followLatest: 1,
            traceId,
          },
        });
      }}
      onSelectTrace={(traceId) => {
        void navigate({
          search: {
            followLatest: undefined,
            traceId: traceId ?? undefined,
          },
        });
      }}
      selectedTraceId={search.traceId}
    />
  );
}
