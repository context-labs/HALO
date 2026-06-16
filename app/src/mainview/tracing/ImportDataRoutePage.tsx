import { useCallback, useState } from "react";
import { Link } from "@tanstack/react-router";
import {
  ArrowLeft,
  MoreHorizontal,
  RefreshCcw,
} from "lucide-react";

import {
  Button,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  cn,
} from "~/lib/ui";
import { trpc } from "~/trpc";
import { WorkspaceNav } from "~/workspace/WorkspaceNav";
import { AppHeader } from "~/components/AppHeader";
import { ImportDataScreen, LocalAgentSetupDialog } from "./ImportDataScreen";
import { LangfuseImportDialog } from "./langfuse/LangfuseImportDialog";
import { PhoenixImportDialog } from "./phoenix/PhoenixImportDialog";
import { FileImportDialog } from "./fileimport/FileImportDialog";
import { LiveStatusBadge, type LiveStatus } from "./TraceTitleBar";

const DEFAULT_INGEST_URL = "http://127.0.0.1:8799/v1/traces";

export function ImportDataRoutePage() {
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [phoenixDialogOpen, setPhoenixDialogOpen] = useState(false);
  const [fileDialogOpen, setFileDialogOpen] = useState(false);
  const [localAgentSetupOpen, setLocalAgentSetupOpen] = useState(false);
  const [liveStatus, setLiveStatus] = useState<LiveStatus>("connecting");
  const utils = trpc.useUtils();
  const infoQuery = trpc.telemetry.info.useQuery();

  const ingestUrl = infoQuery.data?.ingestUrl ?? DEFAULT_INGEST_URL;
  const catalystEnvLine = `CATALYST_OTLP_ENDPOINT=${ingestUrl}`;
  const isRefreshing = infoQuery.isFetching;

  const refreshTelemetry = useCallback(() => {
    void infoQuery.refetch();
    void utils.traces.facets.invalidate();
    void utils.traces.list.invalidate();
    void utils.traces.search.invalidate();
    void utils.sessions.facets.invalidate();
    void utils.sessions.list.invalidate();
    void utils.sessions.search.invalidate();
  }, [infoQuery, utils]);

  trpc.live.workspace.useSubscription(undefined, {
    onComplete() {
      setLiveStatus("offline");
    },
    onData() {
      setLiveStatus("live");
      void utils.telemetry.info.invalidate();
    },
    onError() {
      setLiveStatus("reconnecting");
    },
    onStarted() {
      setLiveStatus("live");
    },
  });

  return (
    <main className="h-screen overflow-hidden bg-background text-foreground">
      <AppHeader
        status={
          <LiveStatusBadge
            health={infoQuery.data?.lastBatch?.status ?? "waiting"}
            liveStatus={liveStatus}
            liveUrl={infoQuery.data?.liveUrl ?? "ws://127.0.0.1:8800"}
          />
        }
        title="Import data"
        actions={
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button aria-label="More actions" size="icon" variant="ghost">
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={refreshTelemetry}>
                <RefreshCcw
                  className={cn("mr-2 h-4 w-4", isRefreshing && "animate-spin")}
                />
                Refresh
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        }
      />

      <div className="grid h-full min-h-0 grid-cols-[14rem_minmax(0,1fr)] pt-14">
        <WorkspaceNav active="imports" />
        <section className="relative min-h-0 min-w-0 overflow-y-auto">
          <div className="absolute left-8 top-4 z-10">
            <Link
              className="inline-flex items-center gap-1.5 text-sm text-muted-foreground transition hover:text-foreground"
              to="/imports"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              Imports
            </Link>
          </div>
          <ImportDataScreen
            onConnectLocalAgent={() => setLocalAgentSetupOpen(true)}
            onImportJsonl={() => setFileDialogOpen(true)}
            onImportLangfuse={() => setImportDialogOpen(true)}
            onImportPhoenix={() => setPhoenixDialogOpen(true)}
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
      <LocalAgentSetupDialog
        envLine={catalystEnvLine}
        ingestUrl={ingestUrl}
        onOpenChange={setLocalAgentSetupOpen}
        open={localAgentSetupOpen}
      />
    </main>
  );
}
