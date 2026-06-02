import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  Ban,
  CheckCircle2,
  Clock3,
  DatabaseZap,
  DownloadCloud,
  KeyRound,
  Loader2,
  Play,
  RefreshCcw,
  Trash2,
  XCircle,
} from "lucide-react";

import {
  Badge,
  Button,
  Dialog,
  Input,
  Separator,
  cn,
  toast,
} from "~/lib/ui";
import { trpc } from "~/trpc";
import type {
  LangfuseDiscovery,
  LangfuseImportStatus,
} from "../../server/langfuse/types";

type DialogStep = "connect" | "select" | "import" | "done";
type DatePreset = "24h" | "7d" | "30d" | "all";

const DEFAULT_LANGFUSE_URL = "http://localhost:3001";

export function LangfuseImportDialog({
  onImported,
  onOpenChange,
  open,
}: {
  onImported: () => void;
  onOpenChange: (open: boolean) => void;
  open: boolean;
}) {
  const utils = trpc.useUtils();
  const [step, setStep] = useState<DialogStep>("connect");
  const [connectionId, setConnectionId] = useState<string | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeFacets, setActiveFacets] = useState<
    LangfuseDiscovery["facets"] | null
  >(null);
  const [connectionName, setConnectionName] = useState("Local Langfuse");
  const [baseUrl, setBaseUrl] = useState(DEFAULT_LANGFUSE_URL);
  const [publicKey, setPublicKey] = useState("");
  const [secretKey, setSecretKey] = useState("");
  const [datePreset, setDatePreset] = useState<DatePreset>("30d");
  const [environment, setEnvironment] = useState("");
  const [traceName, setTraceName] = useState("");
  const [tag, setTag] = useState("");
  const [userId, setUserId] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [version, setVersion] = useState("");
  const [release, setRelease] = useState("");

  const connectionsQuery = trpc.langfuse.connections.list.useQuery(undefined, {
    enabled: open,
  });
  const jobsQuery = trpc.langfuse.imports.list.useQuery(
    { limit: 8 },
    { enabled: open },
  );
  const activeJobQuery = trpc.langfuse.imports.get.useQuery(
    { jobId: activeJobId ?? "" },
    {
      enabled: open && Boolean(activeJobId),
      refetchInterval: (query) => {
        const status = query.state.data?.status;
        return status === "queued" || status === "running" ? 1_500 : false;
      },
    },
  );

  const saveAndDiscover = trpc.langfuse.connections.saveAndDiscover.useMutation({
    onError(error) {
      toast.error({
        title: "Could not connect to Langfuse",
        description: error.message,
      });
    },
    async onSuccess(result) {
      setConnectionId(result.connection.id);
      setActiveFacets(result.discovery.facets);
      setBaseUrl(result.connection.baseUrl);
      setConnectionName(result.connection.name);
      setPublicKey(result.connection.publicKey);
      setSecretKey("");
      resetFacetSelections();
      setStep("select");
      await utils.langfuse.connections.list.invalidate();
      toast.success({
        title: "Langfuse connected",
        description: `${result.discovery.traces.totalItems} traces discovered.`,
      });
    },
  });

  const startImport = trpc.langfuse.imports.start.useMutation({
    onError(error) {
      toast.error({
        title: "Could not start import",
        description: error.message,
      });
    },
    async onSuccess(job) {
      setActiveJobId(job.id);
      setStep("import");
      await utils.langfuse.imports.list.invalidate();
      toast.info({
        title: "Langfuse import queued",
        description: "The import will keep running if this dialog is closed.",
      });
    },
  });

  const cancelImport = trpc.langfuse.imports.cancel.useMutation({
    async onSuccess(job) {
      await utils.langfuse.imports.get.invalidate({ jobId: job.id });
      await utils.langfuse.imports.list.invalidate();
      toast.warning({
        title: "Import cancelled",
        description: "The current Langfuse import has been stopped.",
      });
    },
  });
  const deleteConnection = trpc.langfuse.connections.delete.useMutation({
    async onSuccess() {
      await utils.langfuse.connections.list.invalidate();
    },
  });

  trpc.live.importJob.useSubscription(
    { jobId: activeJobId ?? "" },
    {
      enabled: open && Boolean(activeJobId),
      onData(eventEnvelope) {
        const event = eventEnvelope.data;
        if (event.payload.type !== "import.job.updated") return;
        const snapshot = event.payload.job;
        utils.langfuse.imports.get.setData(
          { jobId: snapshot.id },
          (current) =>
            current
              ? {
                  ...current,
                  ...snapshot,
                  status: snapshot.status as LangfuseImportStatus,
                }
              : current,
        );
        void utils.langfuse.imports.list.invalidate();
        if (snapshot.status === "completed") {
          setStep("done");
          onImported();
        }
      },
    },
  );

  const latestJob = activeJobQuery.data;
  const discovery = useMemo(() => {
    const connection = connectionsQuery.data?.find((item) => item.id === connectionId);
    return activeFacets ?? connection?.discoveredFacets;
  }, [activeFacets, connectionId, connectionsQuery.data]);

  useEffect(() => {
    if (!open || activeJobId) return;
    const running = jobsQuery.data?.find((job) =>
      ["queued", "running"].includes(job.status),
    );
    if (running) {
      setActiveJobId(running.id);
      setStep("import");
    }
  }, [activeJobId, jobsQuery.data, open]);

  useEffect(() => {
    if (!latestJob) return;
    if (latestJob.status === "completed") {
      setStep("done");
      onImported();
    }
  }, [latestJob, onImported]);

  const connectWithCurrentValues = () => {
    saveAndDiscover.mutate({
      baseUrl,
      name: connectionName,
      publicKey,
      secretKey,
    });
  };

  const reconnectStored = (id: string) => {
    saveAndDiscover.mutate({ id });
  };

  const beginImport = () => {
    if (!connectionId) return;
    startImport.mutate({
      connectionId,
      filters: buildFilters({
        datePreset,
        environment,
        release,
        sessionId,
        tag,
        traceName,
        userId,
        version,
      }),
    });
  };

  const canStartImport = Boolean(connectionId) && !startImport.isPending;

  return (
    <Dialog
      className="!w-[min(800px,92vw)] !max-w-[92vw] sm:!max-w-[800px] md:!w-[800px]"
      dialogDescription="Bring historical Langfuse traces into the local HALO timeline."
      dialogTitle={
        <span className="flex items-center gap-2">
          <DownloadCloud className="h-5 w-5 text-detail-brand" />
          Import Data
        </span>
      }
      maxWidth={800}
      footer={
        <div className="flex items-center justify-between gap-3 border-t border-subtle px-6 py-4">
          <StepRail step={step} />
          <div className="flex items-center gap-2">
            {step !== "connect" ? (
              <Button onClick={() => setStep("connect")} variant="secondary">
                Back
              </Button>
            ) : null}
            <Button onClick={() => onOpenChange(false)} variant="ghost">
              Close
            </Button>
            {step === "select" ? (
              <Button disabled={!canStartImport} onClick={beginImport}>
                {startImport.isPending ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Play className="mr-2 h-4 w-4" />
                )}
                Start import
              </Button>
            ) : null}
          </div>
        </div>
      }
      hideConfirmButton
      onConfirm={() => undefined}
      onOpenChange={onOpenChange}
      open={open}
    >
      <div className="space-y-5">
        {step === "connect" ? (
          <ConnectStep
            baseUrl={baseUrl}
            connectionName={connectionName}
            connections={connectionsQuery.data ?? []}
            isConnecting={saveAndDiscover.isPending}
            onBaseUrlChange={setBaseUrl}
            onConnect={connectWithCurrentValues}
            onConnectionNameChange={setConnectionName}
            onDeleteConnection={(id) => deleteConnection.mutate({ id })}
            onPublicKeyChange={setPublicKey}
            onReconnectStored={reconnectStored}
            onSecretKeyChange={setSecretKey}
            publicKey={publicKey}
            secretKey={secretKey}
          />
        ) : null}

        {step === "select" ? (
          <SelectStep
            datePreset={datePreset}
            discovery={discovery}
            environment={environment}
            onDatePresetChange={setDatePreset}
            onEnvironmentChange={setEnvironment}
            onReleaseChange={setRelease}
            onSessionIdChange={setSessionId}
            onTagChange={setTag}
            onTraceNameChange={setTraceName}
            onUserIdChange={setUserId}
            onVersionChange={setVersion}
            release={release}
            sessionId={sessionId}
            tag={tag}
            traceName={traceName}
            userId={userId}
            version={version}
          />
        ) : null}

        {step === "import" || step === "done" ? (
          <ImportProgressStep
            job={latestJob}
            onCancel={() => {
              if (latestJob) cancelImport.mutate({ jobId: latestJob.id });
            }}
            onNewImport={() => {
              setActiveJobId(null);
              setStep("select");
            }}
          />
        ) : null}

        <RecentJobs
          jobs={jobsQuery.data ?? []}
          onSelect={(jobId) => {
            setActiveJobId(jobId);
            setStep("import");
          }}
        />
      </div>
    </Dialog>
  );

  function resetFacetSelections() {
    setDatePreset("30d");
    setEnvironment("");
    setTraceName("");
    setTag("");
    setUserId("");
    setSessionId("");
    setVersion("");
    setRelease("");
  }
}

function ConnectStep({
  baseUrl,
  connectionName,
  connections,
  isConnecting,
  onBaseUrlChange,
  onConnect,
  onConnectionNameChange,
  onDeleteConnection,
  onPublicKeyChange,
  onReconnectStored,
  onSecretKeyChange,
  publicKey,
  secretKey,
}: {
  baseUrl: string;
  connectionName: string;
  connections: Array<{
    baseUrl: string;
    id: string;
    lastStatus: string;
    name: string;
    projectName: string | null;
    publicKey: string;
    updatedAt: string;
  }>;
  isConnecting: boolean;
  onBaseUrlChange: (value: string) => void;
  onConnect: () => void;
  onConnectionNameChange: (value: string) => void;
  onDeleteConnection: (id: string) => void;
  onPublicKeyChange: (value: string) => void;
  onReconnectStored: (id: string) => void;
  onSecretKeyChange: (value: string) => void;
  publicKey: string;
  secretKey: string;
}) {
  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1.1fr)_320px]">
      <div className="space-y-4">
        <StatusPanel
          icon={<KeyRound className="h-4 w-4" />}
          title="Langfuse credentials"
        >
          <p className="text-sm text-muted-foreground">
            Use a project API key pair. HALO stores it locally and uses it
            to import trace history over the Langfuse public API.
          </p>
        </StatusPanel>
        <div className="grid gap-3">
          <Input
            label="Connection name"
            onChange={(event) => onConnectionNameChange(event.currentTarget.value)}
            placeholder="Local Langfuse"
            value={connectionName}
          />
          <Input
            label="API URL"
            onChange={(event) => onBaseUrlChange(event.currentTarget.value)}
            placeholder={DEFAULT_LANGFUSE_URL}
            value={baseUrl}
          />
          <Input
            label="Public key"
            onChange={(event) => onPublicKeyChange(event.currentTarget.value)}
            placeholder="lf_pk_..."
            value={publicKey}
          />
          <Input
            label="Secret key"
            onChange={(event) => onSecretKeyChange(event.currentTarget.value)}
            placeholder="lf_sk_..."
            type="password"
            value={secretKey}
          />
        </div>
        <Button className="w-full" disabled={isConnecting} onClick={onConnect}>
          {isConnecting ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <RefreshCcw className="mr-2 h-4 w-4" />
          )}
          Connect and discover
        </Button>
      </div>

      <div className="rounded-lg border border-subtle bg-background-muted p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold">Saved connections</h3>
          <Badge variant="outline">{connections.length}</Badge>
        </div>
        <div className="mt-3 space-y-2">
          {connections.length === 0 ? (
            <p className="rounded-md border border-dashed border-subtle p-4 text-sm text-muted-foreground">
              No Langfuse connections saved yet.
            </p>
          ) : (
            connections.map((connection) => (
              <div
                className="rounded-md border border-subtle bg-background p-3"
                key={connection.id}
              >
                <div className="flex min-w-0 items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{connection.name}</p>
                    <p className="truncate text-xs text-muted-foreground">
                      {connection.projectName ?? connection.baseUrl}
                    </p>
                  </div>
                  <Badge
                    variant={
                      connection.lastStatus === "connected"
                        ? "status-success"
                        : "outline"
                    }
                  >
                    {connection.lastStatus}
                  </Badge>
                </div>
                <div className="mt-3 flex items-center gap-2">
                  <Button
                    className="flex-1"
                    onClick={() => onReconnectStored(connection.id)}
                    size="sm"
                    variant="secondary"
                  >
                    Use
                  </Button>
                  <Button
                    aria-label="Delete Langfuse connection"
                    onClick={() => onDeleteConnection(connection.id)}
                    size="icon"
                    variant="ghost"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

function SelectStep({
  datePreset,
  discovery,
  environment,
  onDatePresetChange,
  onEnvironmentChange,
  onReleaseChange,
  onSessionIdChange,
  onTagChange,
  onTraceNameChange,
  onUserIdChange,
  onVersionChange,
  release,
  sessionId,
  tag,
  traceName,
  userId,
  version,
}: {
  datePreset: DatePreset;
  discovery:
    | {
        environments: Facet[];
        releases: Facet[];
        sessions: Facet[];
        tags: Facet[];
        traceNames: Facet[];
        users: Facet[];
        versions: Facet[];
      }
    | undefined;
  environment: string;
  onDatePresetChange: (value: DatePreset) => void;
  onEnvironmentChange: (value: string) => void;
  onReleaseChange: (value: string) => void;
  onSessionIdChange: (value: string) => void;
  onTagChange: (value: string) => void;
  onTraceNameChange: (value: string) => void;
  onUserIdChange: (value: string) => void;
  onVersionChange: (value: string) => void;
  release: string;
  sessionId: string;
  tag: string;
  traceName: string;
  userId: string;
  version: string;
}) {
  return (
    <div className="space-y-5">
      <div className="grid gap-3 md:grid-cols-3">
        <StatusPanel
          icon={<DatabaseZap className="h-4 w-4" />}
          title="Trace import"
        >
          <p className="text-sm text-muted-foreground">
            Import traces and observations into the local SQLite span store.
          </p>
        </StatusPanel>
        <MetricCard label="Trace names" value={discovery?.traceNames.length ?? 0} />
        <MetricCard label="Tags" value={discovery?.tags.length ?? 0} />
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <NativeSelect
          label="Time window"
          onChange={(value) => onDatePresetChange(value as DatePreset)}
          options={[
            { label: "Last 24 hours", value: "24h" },
            { label: "Last 7 days", value: "7d" },
            { label: "Last 30 days", value: "30d" },
            { label: "All time", value: "all" },
          ]}
          value={datePreset}
        />
        <NativeSelect
          label="Environment"
          onChange={onEnvironmentChange}
          options={facetOptions(discovery?.environments, "Any environment")}
          value={environment}
        />
        <NativeSelect
          label="Trace name"
          onChange={onTraceNameChange}
          options={facetOptions(discovery?.traceNames, "Any trace name")}
          value={traceName}
        />
        <NativeSelect
          label="Tag"
          onChange={onTagChange}
          options={facetOptions(discovery?.tags, "Any tag")}
          value={tag}
        />
        <NativeSelect
          label="User"
          onChange={onUserIdChange}
          options={facetOptions(discovery?.users, "Any user")}
          value={userId}
        />
        <NativeSelect
          label="Session"
          onChange={onSessionIdChange}
          options={facetOptions(discovery?.sessions, "Any session")}
          value={sessionId}
        />
        <NativeSelect
          label="Version"
          onChange={onVersionChange}
          options={facetOptions(discovery?.versions, "Any version")}
          value={version}
        />
        <NativeSelect
          label="Release"
          onChange={onReleaseChange}
          options={facetOptions(discovery?.releases, "Any release")}
          value={release}
        />
      </div>
    </div>
  );
}

function ImportProgressStep({
  job,
  onCancel,
  onNewImport,
}: {
  job:
    | {
        currentTraceName: string | null;
        errorMessage: string | null;
        failedTraces: number;
        importedObservations: number;
        importedTraces: number;
        progress: number;
        status: string;
        totalObservations: number;
        totalTraces: number;
        updatedAt: string;
      }
    | null
    | undefined;
  onCancel: () => void;
  onNewImport: () => void;
}) {
  if (!job) {
    return (
      <div className="grid min-h-64 place-items-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }
  const active = job.status === "queued" || job.status === "running";
  const failed = job.status === "failed" || job.status === "interrupted";
  const cancelled = job.status === "cancelled";

  return (
    <div className="space-y-5">
      <div className="rounded-lg border border-subtle bg-background-muted p-5">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <ImportStatusIcon status={job.status} />
              <h3 className="text-lg font-semibold">{statusTitle(job.status)}</h3>
            </div>
            <p className="mt-1 text-sm text-muted-foreground">
              {job.currentTraceName
                ? `Importing ${job.currentTraceName}`
                : `Updated ${relativeTime(job.updatedAt)}`}
            </p>
          </div>
          <Badge variant={failed ? "status-failure" : active ? "status-running" : "outline"}>
            {job.status}
          </Badge>
        </div>

        <div className="mt-5 h-2 overflow-hidden rounded-full bg-muted">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-500",
              failed
                ? "bg-detail-failure"
                : cancelled
                  ? "bg-muted-foreground"
                  : "bg-detail-brand",
            )}
            style={{ width: `${Math.max(2, job.progress)}%` }}
          />
        </div>

        {job.errorMessage ? (
          <p className="mt-3 rounded-md border border-detail-failure/30 bg-detail-failure/10 p-3 text-sm text-detail-failure">
            {job.errorMessage}
          </p>
        ) : null}

        <div className="mt-5 grid gap-3 sm:grid-cols-4">
          <MetricCard label="Traces" value={`${job.importedTraces}/${job.totalTraces}`} />
          <MetricCard label="Observations" value={job.importedObservations} />
          <MetricCard label="Known total" value={job.totalObservations} />
          <MetricCard label="Failures" value={job.failedTraces} />
        </div>
      </div>

      <div className="flex items-center justify-end gap-2">
        {active ? (
          <Button onClick={onCancel} variant="secondary">
            <Ban className="mr-2 h-4 w-4" />
            Cancel import
          </Button>
        ) : (
          <Button onClick={onNewImport} variant="secondary">
            Start another import
          </Button>
        )}
      </div>
    </div>
  );
}

function RecentJobs({
  jobs,
  onSelect,
}: {
  jobs: Array<{
    id: string;
    connectionName: string | null;
    importedTraces: number;
    progress: number;
    status: string;
    updatedAt: string;
  }>;
  onSelect: (jobId: string) => void;
}) {
  if (jobs.length === 0) return null;
  return (
    <div>
      <Separator className="my-5" />
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Recent imports</h3>
        <Badge variant="outline">{jobs.length}</Badge>
      </div>
      <div className="mt-3 grid gap-2 md:grid-cols-2">
        {jobs.map((job) => (
          <button
            className="rounded-md border border-subtle bg-background-muted p-3 text-left transition hover:bg-muted"
            key={job.id}
            onClick={() => onSelect(job.id)}
            type="button"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="truncate text-sm font-medium">
                {job.connectionName ?? "Langfuse"}
              </span>
              <Badge size="sm" variant="outline">
                {job.status}
              </Badge>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              {job.importedTraces} traces, {job.progress}% complete,
              {" "}
              {relativeTime(job.updatedAt)}
            </p>
          </button>
        ))}
      </div>
    </div>
  );
}

function StepRail({ step }: { step: DialogStep }) {
  const steps: DialogStep[] = ["connect", "select", "import", "done"];
  const activeIndex = steps.indexOf(step);
  return (
    <div className="hidden items-center gap-2 md:flex">
      {steps.map((item, index) => (
        <div className="flex items-center gap-2" key={item}>
          <span
            className={cn(
              "grid h-6 min-w-6 place-items-center rounded-full border text-[11px]",
              index <= activeIndex
                ? "border-detail-brand bg-detail-brand/15 text-detail-brand"
                : "border-subtle text-muted-foreground",
            )}
          >
            {index + 1}
          </span>
          <span className="text-xs capitalize text-muted-foreground">{item}</span>
          {index < steps.length - 1 ? <span className="h-px w-5 bg-border" /> : null}
        </div>
      ))}
    </div>
  );
}

function StatusPanel({
  children,
  icon,
  title,
}: {
  children: ReactNode;
  icon: ReactNode;
  title: string;
}) {
  return (
    <div className="rounded-lg border border-subtle bg-background-muted p-4">
      <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
        {icon}
        {title}
      </div>
      {children}
    </div>
  );
}

function NativeSelect({
  label,
  onChange,
  options,
  value,
}: {
  label: string;
  onChange: (value: string) => void;
  options: Array<{ label: string; value: string }>;
  value: string;
}) {
  return (
    <label className="space-y-2">
      <span className="text-xs font-semibold uppercase text-muted-foreground">
        {label}
      </span>
      <select
        className="h-10 w-full rounded-md border border-subtle bg-background px-3 text-sm"
        onChange={(event) => onChange(event.currentTarget.value)}
        value={value}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function MetricCard({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="rounded-md border border-subtle bg-background px-3 py-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-1 text-lg font-semibold">{value}</p>
    </div>
  );
}

function ImportStatusIcon({ status }: { status: string }) {
  if (status === "completed") return <CheckCircle2 className="h-5 w-5 text-detail-success" />;
  if (status === "failed" || status === "interrupted") {
    return <XCircle className="h-5 w-5 text-detail-failure" />;
  }
  if (status === "cancelled") return <Ban className="h-5 w-5 text-muted-foreground" />;
  if (status === "queued") return <Clock3 className="h-5 w-5 text-detail-brand" />;
  return <Loader2 className="h-5 w-5 animate-spin text-detail-brand" />;
}

type Facet = { count: number; label: string; value: string };

function facetOptions(facets: Facet[] | undefined, emptyLabel: string) {
  return [
    { label: emptyLabel, value: "" },
    ...(facets ?? []).map((facet) => ({
      label: `${facet.label} (${facet.count})`,
      value: facet.value,
    })),
  ];
}

function buildFilters(input: {
  datePreset: DatePreset;
  environment: string;
  release: string;
  sessionId: string;
  tag: string;
  traceName: string;
  userId: string;
  version: string;
}) {
  const fromTimestamp = fromTimestampForPreset(input.datePreset);
  return {
    environment: input.environment || undefined,
    fromTimestamp,
    release: input.release || undefined,
    sessionId: input.sessionId || undefined,
    tag: input.tag || undefined,
    traceName: input.traceName || undefined,
    userId: input.userId || undefined,
    version: input.version || undefined,
  };
}

function fromTimestampForPreset(preset: DatePreset) {
  if (preset === "all") return undefined;
  const hours = preset === "24h" ? 24 : preset === "7d" ? 24 * 7 : 24 * 30;
  return new Date(Date.now() - hours * 60 * 60 * 1000).toISOString();
}

function statusTitle(status: string) {
  if (status === "completed") return "Import complete";
  if (status === "failed") return "Import failed";
  if (status === "cancelled") return "Import cancelled";
  if (status === "interrupted") return "Import interrupted";
  if (status === "queued") return "Import queued";
  return "Importing traces";
}

function relativeTime(iso: string) {
  const delta = Date.now() - Date.parse(iso);
  if (!Number.isFinite(delta)) return "just now";
  if (delta < 60_000) return "just now";
  if (delta < 3_600_000) return `${Math.floor(delta / 60_000)}m ago`;
  if (delta < 86_400_000) return `${Math.floor(delta / 3_600_000)}h ago`;
  return `${Math.floor(delta / 86_400_000)}d ago`;
}
