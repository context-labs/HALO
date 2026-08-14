import { useCallback, useState } from "react";

import { trpc } from "~/trpc";
import { WorkspaceNav } from "~/workspace/WorkspaceNav";
import { AppHeader } from "~/components/AppHeader";
import { openExternalUrl } from "~/desktop/desktopBridge";
import { APP_DOCS_URL } from "../../desktop/commands";
import { ImportDataScreen, LocalAgentSetupDialog } from "./ImportDataScreen";
import { LangfuseImportDialog } from "./langfuse/LangfuseImportDialog";
import { PhoenixImportDialog } from "./phoenix/PhoenixImportDialog";
import { FileImportDialog } from "./fileimport/FileImportDialog";
import { DemoTracesImportDialog } from "./DemoTracesImportDialog";

const DEFAULT_INGEST_URL = "http://127.0.0.1:8799/v1/traces";

export function ImportDataRoutePage() {
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [phoenixDialogOpen, setPhoenixDialogOpen] = useState(false);
  const [fileDialogOpen, setFileDialogOpen] = useState(false);
  const [demoDialogOpen, setDemoDialogOpen] = useState(false);
  const [localAgentSetupOpen, setLocalAgentSetupOpen] = useState(false);
  const utils = trpc.useUtils();
  const infoQuery = trpc.telemetry.info.useQuery();

  const ingestUrl = infoQuery.data?.ingestUrl ?? DEFAULT_INGEST_URL;
  const inferenceEnvLine = `INFERENCE_OTLP_ENDPOINT=${ingestUrl}`;

  const refreshTelemetry = useCallback(() => {
    void infoQuery.refetch();
    void utils.traces.facets.invalidate();
    void utils.traces.list.invalidate();
    void utils.traces.search.invalidate();
    void utils.sessions.facets.invalidate();
    void utils.sessions.list.invalidate();
    void utils.sessions.search.invalidate();
  }, [infoQuery, utils]);
  const handleReadDocumentation = useCallback(() => {
    void openExternalUrl(APP_DOCS_URL);
  }, []);

  trpc.live.workspace.useSubscription(undefined, {
    onData() {
      void utils.telemetry.info.invalidate();
    },
  });

  return (
    <main className="h-screen overflow-hidden bg-background text-foreground">
      <AppHeader title="Import data" />

      <div className="grid h-full min-h-0 grid-cols-[14rem_minmax(0,1fr)] pt-14">
        <WorkspaceNav active="imports" />
        <section className="relative min-h-0 min-w-0 overflow-y-auto">
          <ImportDataScreen
            onConnectLocalAgent={() => setLocalAgentSetupOpen(true)}
            onImportJsonl={() => setFileDialogOpen(true)}
            onImportLangfuse={() => setImportDialogOpen(true)}
            onImportPhoenix={() => setPhoenixDialogOpen(true)}
            onLoadDemoTraces={() => setDemoDialogOpen(true)}
            onReadDocumentation={handleReadDocumentation}
          />
        </section>
      </div>

      <LangfuseImportDialog
        onImported={refreshTelemetry}
        onOpenChange={setImportDialogOpen}
        open={importDialogOpen}
      />
      <PhoenixImportDialog
        onImported={refreshTelemetry}
        onOpenChange={setPhoenixDialogOpen}
        open={phoenixDialogOpen}
      />
      <FileImportDialog
        onImported={refreshTelemetry}
        onOpenChange={setFileDialogOpen}
        open={fileDialogOpen}
      />
      <DemoTracesImportDialog
        onImported={refreshTelemetry}
        onOpenChange={setDemoDialogOpen}
        open={demoDialogOpen}
      />
      <LocalAgentSetupDialog
        envLine={inferenceEnvLine}
        ingestUrl={ingestUrl}
        onOpenChange={setLocalAgentSetupOpen}
        open={localAgentSetupOpen}
      />
    </main>
  );
}
