import { createFileRoute } from "@tanstack/react-router";

import { AnalysisPage } from "~/halo/HaloPages";

export const Route = createFileRoute("/analysis")({
  component: AnalysisRoute,
  validateSearch: (search) => ({
    runId: typeof search.runId === "string" ? search.runId : undefined,
  }),
});

function AnalysisRoute() {
  const navigate = Route.useNavigate();
  const search = Route.useSearch();

  return (
    <AnalysisPage
      onSelectRun={(runId) => {
        void navigate({
          search: {
            runId: runId ?? undefined,
          },
        });
      }}
      selectedRunId={search.runId}
    />
  );
}
