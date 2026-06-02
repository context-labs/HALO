import { useDeferredValue, useMemo, useState, type ReactNode } from "react";
import {
  AlertCircle,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  Copy,
  DownloadCloud,
  Loader2,
  Palette,
  Play,
  RefreshCcw,
  Save,
  Settings,
  Square,
  Trash2,
} from "lucide-react";

import {
  Badge,
  Button,
  Input,
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  Textarea,
  ThemeToggle,
  toast,
  cn,
} from "~/lib/ui";
import { trpc } from "~/trpc";
import { WorkspaceNav } from "~/workspace/WorkspaceNav";
import type { HaloRun, HaloRunTargetType } from "../../server/halo/types";
import type { FacetOption, TelemetryFilters } from "../../server/telemetry/types";
import { HaloRunActivity } from "./HaloRunActivity.tsx";

type HaloRunView = Omit<HaloRun, "filters"> & { filters: unknown };

type DateRange = "1h" | "24h" | "7d" | "all";
type StatusFilter = "all" | "ok" | "error";
type SourceFilter = "all" | "local" | "langfuse";

const DEFAULT_PROMPT =
  "Analyze these traces. Identify the most important failures, latency bottlenecks, confusing tool behavior, and concrete improvements for the developer.";

export function AnalysisPage({
  onSelectRun,
  selectedRunId,
}: {
  onSelectRun: (runId: string | null) => void;
  selectedRunId?: string;
}) {
  const utils = trpc.useUtils();
  const [targetType, setTargetType] = useState<HaloRunTargetType>("session_group");
  const [dateRange, setDateRange] = useState<DateRange>("24h");
  const [status, setStatus] = useState<StatusFilter>("all");
  const [source, setSource] = useState<SourceFilter>("all");
  const [serviceName, setServiceName] = useState("all");
  const [agentName, setAgentName] = useState("all");
  const [modelName, setModelName] = useState("all");
  const [providerId, setProviderId] = useState("");
  const [title, setTitle] = useState("");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [maxDepth, setMaxDepth] = useState(1);
  const [maxTurns, setMaxTurns] = useState(8);
  const [maxParallel, setMaxParallel] = useState(2);
  const deferredPrompt = useDeferredValue(prompt);

  const filters = useMemo<TelemetryFilters>(
    () => ({
      agents: agentName === "all" ? undefined : [agentName],
      llmModelNames: modelName === "all" ? undefined : [modelName],
      serviceNames: serviceName === "all" ? undefined : [serviceName],
      sources: source === "all" ? undefined : [source],
      startDate: startDateForRange(dateRange),
      status: status === "all" ? undefined : status,
    }),
    [agentName, dateRange, modelName, serviceName, source, status],
  );

  const engineQuery = trpc.halo.engine.status.useQuery();
  const providersQuery = trpc.halo.providers.list.useQuery();
  const runsQuery = trpc.halo.runs.list.useQuery({ limit: 50 });
  const sessionFacetsQuery = trpc.sessions.facets.useQuery(
    {
      facetIds: ["agent_name", "llm_model_name", "service_name", "source", "status"],
    },
    { enabled: targetType === "session_group" },
  );
  const traceFacetsQuery = trpc.traces.facets.useQuery(
    {
      facetIds: ["agent_name", "llm_model_name", "service_name", "source", "status"],
    },
    { enabled: targetType === "trace_group" },
  );
  const facets = targetType === "session_group" ? sessionFacetsQuery.data : traceFacetsQuery.data;
  const previewQuery = trpc.halo.runs.preview.useQuery({
    filters,
    targetType,
  });

  const startMutation = trpc.halo.runs.start.useMutation({
    async onSuccess(run) {
      toast.success({ title: "HALO run queued" });
      onSelectRun(run.id);
      await Promise.all([
        utils.halo.runs.list.invalidate(),
        utils.halo.runs.get.invalidate({ runId: run.id }),
      ]);
    },
    onError(error) {
      toast.error({ title: "Could not start HALO run", description: error.message });
    },
  });
  const cancelMutation = trpc.halo.runs.cancel.useMutation({
    async onSuccess(run) {
      toast.success({ title: "HALO run cancelled" });
      await Promise.all([
        utils.halo.runs.list.invalidate(),
        utils.halo.runs.get.invalidate({ runId: run.id }),
      ]);
    },
  });

  const providers = providersQuery.data ?? [];
  const selectedProvider = providers.find((provider) => provider.id === providerId);
  const canStart =
    Boolean(providerId || selectedProvider) &&
    deferredPrompt.trim().length > 0 &&
    previewQuery.data != null &&
    previewQuery.data.spanCount > 0 &&
    engineQuery.data?.status === "installed";
  const runs = runsQuery.data ?? [];

  return (
    <main className="min-h-screen bg-background text-foreground">
      <WorkspaceHeader
        description="Run local HALO analysis over filtered trace and session groups."
        icon={<BrainCircuit className="h-4 w-4 text-detail-brand" />}
        title="Analysis"
      />
      <div className="grid min-h-[calc(100vh-3.5rem)] grid-cols-[112px_minmax(0,1fr)] pt-14">
        <WorkspaceNav active="analysis" />
        <section className="min-w-0 overflow-auto">
          <div className="mx-auto flex max-w-7xl flex-col gap-5 p-6">
            <SetupBanner
              engineInstalled={engineQuery.data?.status === "installed"}
              providerCount={providers.length}
            />

            <div className="grid gap-5 xl:grid-cols-[minmax(420px,0.92fr)_minmax(0,1.08fr)]">
              <div className="border border-subtle bg-card">
                <div className="border-b border-subtle p-5">
                  <h1 className="text-xl font-semibold">New HALO run</h1>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Select a filtered group, preview the local export, and stream
                    the analysis back into this workspace.
                  </p>
                </div>
                <div className="space-y-5 p-5">
                  <div className="grid grid-cols-2 gap-2">
                    <SegmentButton
                      active={targetType === "session_group"}
                      label="Session group"
                      onClick={() => setTargetType("session_group")}
                    />
                    <SegmentButton
                      active={targetType === "trace_group"}
                      label="Trace group"
                      onClick={() => setTargetType("trace_group")}
                    />
                  </div>

                  <div className="grid gap-3 sm:grid-cols-2">
                    <FilterSelect
                      label="Window"
                      onChange={(value) => setDateRange(value as DateRange)}
                      options={[
                        { label: "Last hour", value: "1h" },
                        { label: "Last 24 hours", value: "24h" },
                        { label: "Last 7 days", value: "7d" },
                        { label: "All time", value: "all" },
                      ]}
                      value={dateRange}
                    />
                    <FilterSelect
                      label="Status"
                      onChange={(value) => setStatus(value as StatusFilter)}
                      options={[
                        { label: "Any status", value: "all" },
                        { label: "OK", value: "ok" },
                        { label: "Errors", value: "error" },
                      ]}
                      value={status}
                    />
                    <FilterSelect
                      label="Source"
                      onChange={(value) => setSource(value as SourceFilter)}
                      options={toFacetOptions(facets?.categorical.source, "Any source")}
                      value={source}
                    />
                    <FilterSelect
                      label="Service"
                      onChange={setServiceName}
                      options={toFacetOptions(
                        facets?.categorical.service_name,
                        "Any service",
                      )}
                      value={serviceName}
                    />
                    <FilterSelect
                      label="Agent"
                      onChange={setAgentName}
                      options={toFacetOptions(
                        facets?.categorical.agent_name,
                        "Any agent",
                      )}
                      value={agentName}
                    />
                    <FilterSelect
                      label="Model"
                      onChange={setModelName}
                      options={toFacetOptions(
                        facets?.categorical.llm_model_name,
                        "Any model",
                      )}
                      value={modelName}
                    />
                  </div>

                  <PreviewCard loading={previewQuery.isLoading} preview={previewQuery.data} />

                  <div className="grid gap-3 sm:grid-cols-2">
                    <label className="space-y-2">
                      <span className="text-xs font-semibold uppercase text-muted-foreground">
                        Provider
                      </span>
                      <select
                        className="h-10 w-full rounded-md border border-subtle bg-background px-3 text-sm"
                        onChange={(event) => setProviderId(event.currentTarget.value)}
                        value={providerId}
                      >
                        <option value="">Choose provider</option>
                        {providers.map((provider) => (
                          <option key={provider.id} value={provider.id}>
                            {provider.name} · {provider.model}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="space-y-2">
                      <span className="text-xs font-semibold uppercase text-muted-foreground">
                        Title
                      </span>
                      <Input
                        onChange={(event) => setTitle(event.currentTarget.value)}
                        placeholder="Optional run title"
                        value={title}
                      />
                    </label>
                  </div>

                  <label className="space-y-2">
                    <span className="text-xs font-semibold uppercase text-muted-foreground">
                      Analysis prompt
                    </span>
                    <Textarea
                      className="min-h-32 resize-y"
                      onChange={(event) => setPrompt(event.currentTarget.value)}
                      value={prompt}
                    />
                  </label>

                  <div className="grid grid-cols-3 gap-3">
                    <NumberField label="Depth" onChange={setMaxDepth} value={maxDepth} />
                    <NumberField label="Turns" onChange={setMaxTurns} value={maxTurns} />
                    <NumberField
                      label="Parallel"
                      onChange={setMaxParallel}
                      value={maxParallel}
                    />
                  </div>

                  <Button
                    className="w-full"
                    disabled={!canStart || startMutation.isPending}
                    onClick={() =>
                      startMutation.mutate({
                        filters,
                        maxDepth,
                        maxParallel,
                        maxTurns,
                        prompt,
                        providerId,
                        targetType,
                        title: title || undefined,
                      })
                    }
                  >
                    {startMutation.isPending ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <Play className="mr-2 h-4 w-4" />
                    )}
                    Start HALO run
                  </Button>
                </div>
              </div>

              <div className="border border-subtle bg-card">
                <div className="flex items-center justify-between border-b border-subtle p-5">
                  <div>
                    <h2 className="text-lg font-semibold">Runs</h2>
                    <p className="mt-1 text-sm text-muted-foreground">
                      Standalone local results, newest first.
                    </p>
                  </div>
                  <Button
                    onClick={() => void runsQuery.refetch()}
                    size="icon"
                    variant="ghost"
                  >
                    <RefreshCcw
                      className={cn("h-4 w-4", runsQuery.isFetching && "animate-spin")}
                    />
                  </Button>
                </div>
                <RunList
                  activeRunId={selectedRunId}
                  isLoading={runsQuery.isLoading}
                  onCancel={(runId) => cancelMutation.mutate({ runId })}
                  onSelect={onSelectRun}
                  runs={runs}
                />
              </div>
            </div>
          </div>
        </section>
      </div>
      <HaloRunSheet
        onOpenChange={(open) => {
          if (!open) onSelectRun(null);
        }}
        open={Boolean(selectedRunId)}
        runId={selectedRunId}
      />
    </main>
  );
}

export function SettingsPage() {
  const utils = trpc.useUtils();
  const engineQuery = trpc.halo.engine.status.useQuery();
  const providersQuery = trpc.halo.providers.list.useQuery();
  const telemetryInfoQuery = trpc.telemetry.info.useQuery();
  const [providerType, setProviderType] = useState<"openai" | "anthropic_compat" | "custom">("openai");
  const [name, setName] = useState("OpenAI");
  const [baseUrl, setBaseUrl] = useState("https://api.openai.com/v1");
  const [model, setModel] = useState("gpt-4.1-mini");
  const [apiKey, setApiKey] = useState("");

  const installMutation = trpc.halo.engine.installOrUpdate.useMutation({
    async onSuccess() {
      toast.success({ title: "HALO engine is ready" });
      await utils.halo.engine.status.invalidate();
    },
    onError(error) {
      toast.error({ title: "HALO install failed", description: error.message });
    },
  });
  const saveProviderMutation = trpc.halo.providers.save.useMutation({
    async onSuccess() {
      toast.success({ title: "Provider saved" });
      setApiKey("");
      await utils.halo.providers.list.invalidate();
    },
    onError(error) {
      toast.error({ title: "Could not save provider", description: error.message });
    },
  });
  const testProviderMutation = trpc.halo.providers.test.useMutation({
    async onSuccess() {
      toast.success({ title: "Provider connected" });
      await utils.halo.providers.list.invalidate();
    },
    onError(error) {
      toast.error({ title: "Provider test failed", description: error.message });
    },
  });
  const deleteProviderMutation = trpc.halo.providers.delete.useMutation({
    async onSuccess() {
      await utils.halo.providers.list.invalidate();
    },
  });

  const providers = providersQuery.data ?? [];
  const status = engineQuery.data;
  const telemetryInfo = telemetryInfoQuery.data;

  return (
    <main className="min-h-screen bg-background text-foreground">
      <WorkspaceHeader
        description="Configure the local HALO engine and model providers."
        icon={<Settings className="h-4 w-4 text-detail-brand" />}
        title="Settings"
      />
      <div className="grid min-h-[calc(100vh-3.5rem)] grid-cols-[112px_minmax(0,1fr)] pt-14">
        <WorkspaceNav active="settings" />
        <section className="min-w-0 overflow-auto">
          <div className="mx-auto grid max-w-6xl gap-5 p-6 xl:grid-cols-[minmax(0,1fr)_420px]">
            <div className="space-y-5">
              <section className="border border-subtle bg-card">
                <div className="flex items-start gap-3 border-b border-subtle p-5">
                  <div className="grid h-9 w-9 shrink-0 place-items-center rounded-md border border-subtle bg-background-muted">
                    <Palette className="h-4 w-4 text-detail-brand" />
                  </div>
                  <div className="min-w-0">
                    <h1 className="text-xl font-semibold">Workspace</h1>
                    <p className="mt-1 text-sm text-muted-foreground">
                      Appearance and local runtime details for this desktop app.
                    </p>
                  </div>
                </div>
                <div className="grid gap-4 p-5 lg:grid-cols-[minmax(0,1fr)_220px]">
                  <div className="grid gap-3 sm:grid-cols-2">
                    <InfoLine
                      label="Database path"
                      value={telemetryInfo?.dbPath ?? "data/halo-canvas.sqlite"}
                    />
                    <InfoLine
                      label="Ingest endpoint"
                      value={telemetryInfo?.ingestUrl ?? "http://127.0.0.1:8799/v1/traces"}
                    />
                    <InfoLine
                      label="Live socket"
                      value={telemetryInfo?.liveUrl ?? "ws://127.0.0.1:8800"}
                    />
                    <InfoLine
                      label="Stored telemetry"
                      value={`${telemetryInfo?.traceCount ?? 0} traces · ${telemetryInfo?.spanCount ?? 0} spans`}
                    />
                  </div>
                  <div className="rounded-md border border-subtle bg-background-muted p-3">
                    <p className="text-xs uppercase text-muted-foreground">
                      Theme
                    </p>
                    <p className="mt-1 text-sm text-muted-foreground">
                      Change the app appearance from Settings instead of the
                      workspace toolbar.
                    </p>
                    <ThemeToggle
                      trigger={
                        <Button className="mt-3 w-full justify-start" variant="outline">
                          <Palette className="mr-2 h-4 w-4" />
                          Choose theme
                        </Button>
                      }
                    />
                  </div>
                </div>
              </section>

              <section className="border border-subtle bg-card">
                <div className="flex items-start justify-between gap-4 border-b border-subtle p-5">
                  <div>
                    <h1 className="text-xl font-semibold">HALO engine</h1>
                    <p className="mt-1 text-sm text-muted-foreground">
                      HALO clones the engine locally and runs it with uv.
                    </p>
                  </div>
                  <StatusBadge status={status?.status ?? "not_installed"} />
                </div>
                <div className="grid gap-4 p-5 md:grid-cols-2">
                  <InfoLine label="Install path" value={status?.installPath ?? "data/halo-engine"} />
                  <InfoLine label="Repo" value={status?.repoUrl ?? "https://github.com/context-labs/HALO"} />
                  <InfoLine label="Commit" value={status?.commitSha ?? "not installed"} />
                  <InfoLine label="Python" value={status?.checks.python ?? "missing"} />
                  <InfoLine label="uv" value={status?.checks.uv ?? "missing"} />
                  <InfoLine label="git" value={status?.checks.git ?? "missing"} />
                  {status?.lastError ? (
                    <div className="md:col-span-2 rounded-md border border-destructive-border bg-destructive/5 p-3 text-sm text-destructive">
                      {status.lastError}
                    </div>
                  ) : null}
                </div>
                <div className="flex items-center justify-between border-t border-subtle p-5">
                  <p className="text-sm text-muted-foreground">
                    Requires git, uv, and Python 3.12. The engine may still call
                    the configured model provider.
                  </p>
                  <Button
                    disabled={installMutation.isPending}
                    onClick={() => installMutation.mutate()}
                  >
                    {installMutation.isPending ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <DownloadCloud className="mr-2 h-4 w-4" />
                    )}
                    Install / update HALO
                  </Button>
                </div>
              </section>

              <section className="border border-subtle bg-card">
                <div className="border-b border-subtle p-5">
                  <h2 className="text-lg font-semibold">Model providers</h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Keys are stored in the local SQLite database and masked in
                    the UI.
                  </p>
                </div>
                <div className="divide-y divide-subtle">
                  {providers.length === 0 ? (
                    <div className="p-5 text-sm text-muted-foreground">
                      No providers saved yet.
                    </div>
                  ) : (
                    providers.map((provider) => (
                      <div
                        className="flex items-center justify-between gap-4 p-5"
                        key={provider.id}
                      >
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <p className="truncate font-medium">{provider.name}</p>
                            <StatusBadge status={provider.lastStatus} />
                          </div>
                          <p className="mt-1 truncate text-xs text-muted-foreground">
                            {provider.baseUrl} · {provider.model} ·{" "}
                            {provider.apiKeyMasked}
                          </p>
                          {provider.lastError ? (
                            <p className="mt-1 text-xs text-destructive">
                              {provider.lastError}
                            </p>
                          ) : null}
                        </div>
                        <div className="flex shrink-0 items-center gap-2">
                          <Button
                            disabled={testProviderMutation.isPending}
                            onClick={() => testProviderMutation.mutate({ id: provider.id })}
                            size="sm"
                            variant="outline"
                          >
                            Test
                          </Button>
                          <Button
                            onClick={() =>
                              deleteProviderMutation.mutate({ id: provider.id })
                            }
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
              </section>
            </div>

            <section className="h-fit border border-subtle bg-card">
              <div className="border-b border-subtle p-5">
                <h2 className="text-lg font-semibold">Add provider</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  HALO expects an OpenAI-compatible endpoint.
                </p>
              </div>
              <div className="space-y-4 p-5">
                <FilterSelect
                  label="Preset"
                  onChange={(value) => {
                    const next = value as typeof providerType;
                    setProviderType(next);
                    if (next === "openai") {
                      setName("OpenAI");
                      setBaseUrl("https://api.openai.com/v1");
                      setModel("gpt-4.1-mini");
                    } else if (next === "anthropic_compat") {
                      setName("Anthropic compatible");
                      setBaseUrl("https://api.anthropic.com/v1");
                      setModel("claude-sonnet-4-20250514");
                    } else {
                      setName("Custom provider");
                      setBaseUrl("");
                      setModel("");
                    }
                  }}
                  options={[
                    { label: "OpenAI", value: "openai" },
                    { label: "Anthropic compatible", value: "anthropic_compat" },
                    { label: "Custom OpenAI-compatible", value: "custom" },
                  ]}
                  value={providerType}
                />
                <Input
                  onChange={(event) => setName(event.currentTarget.value)}
                  placeholder="Provider name"
                  value={name}
                />
                <Input
                  onChange={(event) => setBaseUrl(event.currentTarget.value)}
                  placeholder="https://api.openai.com/v1"
                  value={baseUrl}
                />
                <Input
                  onChange={(event) => setModel(event.currentTarget.value)}
                  placeholder="Model id"
                  value={model}
                />
                <Input
                  onChange={(event) => setApiKey(event.currentTarget.value)}
                  placeholder="API key"
                  type="password"
                  value={apiKey}
                />
                <Button
                  className="w-full"
                  disabled={
                    saveProviderMutation.isPending ||
                    !name.trim() ||
                    !baseUrl.trim() ||
                    !model.trim() ||
                    !apiKey.trim()
                  }
                  onClick={() =>
                    saveProviderMutation.mutate({
                      apiKey,
                      baseUrl,
                      model,
                      name,
                      providerType,
                    })
                  }
                >
                  {saveProviderMutation.isPending ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Save className="mr-2 h-4 w-4" />
                  )}
                  Save provider
                </Button>
              </div>
            </section>
          </div>
        </section>
      </div>
    </main>
  );
}

function WorkspaceHeader({
  description,
  icon,
  title,
}: {
  description: string;
  icon: ReactNode;
  title: string;
}) {
  return (
    <div className="electrobun-webkit-app-region-drag fixed inset-x-0 top-0 z-40 grid h-14 select-none grid-cols-[112px_minmax(0,1fr)] border-b border-subtle bg-background/95 backdrop-blur">
      <div className="border-r border-subtle" />
      <div className="flex min-w-0 items-center justify-between gap-4 px-4">
        <div className="flex min-w-0 items-center gap-3">
          <div className="grid h-8 w-8 shrink-0 place-items-center rounded-md border border-subtle bg-card">
            {icon}
          </div>
          <div className="min-w-0">
            <p className="text-xs font-medium text-muted-foreground">HALO</p>
            <p className="truncate text-sm font-semibold">{title}</p>
          </div>
          <span className="hidden truncate text-xs text-muted-foreground md:block">
            {description}
          </span>
        </div>
      </div>
    </div>
  );
}

function SetupBanner({
  engineInstalled,
  providerCount,
}: {
  engineInstalled: boolean;
  providerCount: number;
}) {
  if (engineInstalled && providerCount > 0) return null;
  return (
    <div className="flex items-center justify-between gap-4 border border-detail-warning/40 bg-detail-warning/10 p-4">
      <div className="flex min-w-0 items-start gap-3">
        <AlertCircle className="mt-0.5 h-5 w-5 text-detail-warning" />
        <div>
          <p className="font-medium">HALO needs setup before analysis can run.</p>
          <p className="mt-1 text-sm text-muted-foreground">
            {engineInstalled ? "Add a model provider in Settings." : "Install the HALO engine in Settings."}
          </p>
        </div>
      </div>
      <Button asChild variant="outline">
        <a href="#/settings">Open Settings</a>
      </Button>
    </div>
  );
}

function PreviewCard({
  loading,
  preview,
}: {
  loading: boolean;
  preview?: {
    sessionCount: number;
    spanCount: number;
    traceCount: number;
    warnings: string[];
  };
}) {
  return (
    <div className="grid grid-cols-3 border border-subtle bg-background-muted">
      <PreviewTile label="Traces" loading={loading} value={preview?.traceCount ?? 0} />
      <PreviewTile label="Sessions" loading={loading} value={preview?.sessionCount ?? 0} />
      <PreviewTile label="Spans" loading={loading} value={preview?.spanCount ?? 0} />
      {preview?.warnings.length ? (
        <div className="col-span-3 border-t border-subtle p-3 text-xs text-detail-warning">
          {preview.warnings.join(" ")}
        </div>
      ) : null}
    </div>
  );
}

function PreviewTile({
  label,
  loading,
  value,
}: {
  label: string;
  loading: boolean;
  value: number;
}) {
  return (
    <div className="border-r border-subtle p-3 last:border-r-0">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-1 text-xl font-semibold">
        {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : value}
      </p>
    </div>
  );
}

function SegmentButton({
  active,
  label,
  onClick,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <Button
      className={cn(active && "border-detail-brand/60 text-detail-brand")}
      onClick={onClick}
      type="button"
      variant={active ? "secondary" : "outline"}
    >
      {label}
    </Button>
  );
}

function FilterSelect({
  label,
  onChange,
  options,
  value,
}: {
  label: string;
  onChange: (value: string) => void;
  options: Array<{ label: string; value: string; count?: number }>;
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
            {option.count == null ? option.label : `${option.label} (${option.count})`}
          </option>
        ))}
      </select>
    </label>
  );
}

function NumberField({
  label,
  onChange,
  value,
}: {
  label: string;
  onChange: (value: number) => void;
  value: number;
}) {
  return (
    <label className="space-y-2">
      <span className="text-xs font-semibold uppercase text-muted-foreground">
        {label}
      </span>
      <Input
        min={label === "Depth" ? 0 : 1}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
        type="number"
        value={String(value)}
      />
    </label>
  );
}

function RunList({
  activeRunId,
  isLoading,
  onCancel,
  onSelect,
  runs,
}: {
  activeRunId?: string;
  isLoading: boolean;
  onCancel: (runId: string) => void;
  onSelect: (runId: string) => void;
  runs: HaloRunView[];
}) {
  if (isLoading) {
    return (
      <div className="grid min-h-80 place-items-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }
  if (runs.length === 0) {
    return (
      <div className="grid min-h-80 place-items-center p-8 text-center text-sm text-muted-foreground">
        No HALO runs yet.
      </div>
    );
  }
  return (
    <div className="divide-y divide-subtle">
      {runs.map((run) => {
        const active = ["queued", "exporting", "running"].includes(run.status);
        return (
          <div
            className={cn(
              "grid grid-cols-[minmax(0,1fr)_120px_96px] items-center gap-3 p-4 transition hover:bg-muted/40",
              activeRunId === run.id && "bg-muted",
            )}
            key={run.id}
          >
            <button
              className="min-w-0 text-left"
              onClick={() => onSelect(run.id)}
              type="button"
            >
              <div className="flex min-w-0 items-center gap-2">
                <StatusBadge status={run.status} />
                <p className="truncate font-medium">{run.title}</p>
              </div>
              <p className="mt-1 truncate text-xs text-muted-foreground">
                {targetLabel(run.targetType)} · {run.traceCount} traces ·{" "}
                {run.spanCount} spans · {run.providerName || "provider"}
              </p>
            </button>
            <ProgressBar value={run.progress} />
            <div className="flex justify-end gap-1">
              {active ? (
                <Button
                  onClick={() => onCancel(run.id)}
                  size="icon"
                  variant="ghost"
                >
                  <Square className="h-4 w-4" />
                </Button>
              ) : null}
              <Button onClick={() => onSelect(run.id)} size="sm" variant="outline">
                Open
              </Button>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function HaloRunSheet({
  onOpenChange,
  open,
  runId,
}: {
  onOpenChange: (open: boolean) => void;
  open: boolean;
  runId?: string;
}) {
  const utils = trpc.useUtils();
  const runQuery = trpc.halo.runs.get.useQuery(
    { runId: runId ?? "" },
    { enabled: open && Boolean(runId) },
  );
  const eventsQuery = trpc.halo.runs.getEvents.useQuery(
    { limit: 1000, runId: runId ?? "" },
    { enabled: open && Boolean(runId) },
  );
  trpc.live.haloRun.useSubscription(
    { runId: runId ?? "" },
    {
      enabled: open && Boolean(runId),
      onData(eventEnvelope) {
        const event = eventEnvelope.data;
        if (
          event.payload.type === "halo.run.updated" ||
          event.payload.type === "halo.run.event" ||
          event.payload.type === "halo.run.completed" ||
          event.payload.type === "halo.run.failed"
        ) {
          void utils.halo.runs.get.invalidate({ runId: event.payload.run.id });
          void utils.halo.runs.getEvents.invalidate({ limit: 1000, runId: event.payload.run.id });
          void utils.halo.runs.list.invalidate();
        }
      },
    },
  );

  const run = runQuery.data ?? null;
  const events = eventsQuery.data ?? [];
  const streamedText = events
    .filter((event) => event.eventType === "delta")
    .map((event) => String(event.payload.text_delta ?? ""))
    .join("");
  const agentSteps = events.filter((event) => event.eventType === "agent_step");

  return (
    <Sheet onOpenChange={onOpenChange} open={open}>
      <SheetContent
        className="flex w-[80vw] max-w-[80vw] flex-col overflow-hidden p-0 max-md:w-[92vw] max-md:max-w-[92vw] sm:max-w-[80vw]"
        side="right"
      >
        <SheetHeader className="sticky top-0 z-10 border-b border-subtle bg-background/95 px-6 py-5 pr-12 backdrop-blur">
          <div className="flex min-w-0 items-start justify-between gap-4">
            <div className="min-w-0">
              <SheetTitle className="truncate text-lg font-semibold">
                {run?.title ?? "HALO run"}
              </SheetTitle>
              <SheetDescription className="mt-1 truncate">
                {run
                  ? `${targetLabel(run.targetType)} · ${run.traceCount} traces · ${run.spanCount} spans`
                  : "Loading run"}
              </SheetDescription>
            </div>
            {run ? <StatusBadge status={run.status} /> : null}
          </div>
          {run ? (
            <>
              <div className="mt-4">
                <ProgressBar value={run.progress} />
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
                <span>{run.providerName || "Provider"}</span>
                <span>{run.model || "Model"}</span>
                <span>{run.startedAt ? formatTimestamp(run.startedAt) : "Queued"}</span>
                <span>{run.progress}% complete</span>
              </div>
            </>
          ) : null}
        </SheetHeader>
        {!run ? (
          <div className="grid flex-1 place-items-center">
            <Loader2 className="h-7 w-7 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <div className="min-h-0 flex-1 overflow-auto bg-background-muted/30">
            <div className="mx-auto grid max-w-7xl gap-6 p-6 xl:grid-cols-[minmax(0,1fr)_340px]">
              <div className="min-w-0 space-y-5">
                {run.errorMessage ? (
                  <div className="rounded-md border border-destructive-border bg-destructive/5 p-4 text-sm text-destructive">
                    {run.errorMessage}
                  </div>
                ) : null}
                <RunAnswerPanel
                  answer={run.finalAnswer || streamedText}
                  isStreaming={!run.finalAnswer && streamedText.length > 0}
                />
                <HaloRunActivity events={agentSteps} />
              </div>
              <aside className="space-y-5 xl:sticky xl:top-6 xl:self-start">
                <RunDetailsPanel run={run} />
                <Panel title="Prompt">
                  <pre className="whitespace-pre-wrap text-xs leading-relaxed text-muted-foreground">
                    {run.prompt}
                  </pre>
                </Panel>
                <Panel title="Artifacts">
                  <div className="space-y-2 text-xs text-muted-foreground">
                    <CopyLine label="Export" value={run.exportPath} />
                    <CopyLine label="Result" value={run.resultPath} />
                  </div>
                </Panel>
              </aside>
            </div>
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
}

function RunAnswerPanel({
  answer,
  isStreaming,
}: {
  answer: string;
  isStreaming: boolean;
}) {
  return (
    <section className="border border-subtle bg-card">
      <div className="flex items-center justify-between gap-3 border-b border-subtle px-5 py-4">
        <div>
          <h2 className="text-base font-semibold">Final answer</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            HALO's analysis, streamed into a local run record.
          </p>
        </div>
        {isStreaming ? (
          <Badge size="sm" variant="status-running">
            Streaming
          </Badge>
        ) : null}
      </div>
      <div className="px-5 py-5">
        <pre className="whitespace-pre-wrap text-[0.94rem] leading-7">
          {answer || "Waiting for HALO output..."}
        </pre>
      </div>
    </section>
  );
}

function Panel({ children, title }: { children: ReactNode; title: string }) {
  return (
    <section className="border border-subtle bg-card">
      <div className="border-b border-subtle px-4 py-3 text-sm font-semibold">
        {title}
      </div>
      <div className="p-4">{children}</div>
    </section>
  );
}

function RunDetailsPanel({ run }: { run: HaloRunView }) {
  return (
    <Panel title="Run details">
      <div className="space-y-3">
        <DetailRow label="Target" value={targetLabel(run.targetType)} />
        <DetailRow label="Progress" value={`${run.progress}%`} />
        <DetailRow label="Provider" value={run.providerName || "provider"} />
        <DetailRow label="Model" value={run.model || "model"} />
        <DetailRow
          label="Started"
          value={run.startedAt ? formatTimestamp(run.startedAt) : "queued"}
        />
        <DetailRow label="Traces" value={String(run.traceCount)} />
        {run.sessionCount > 0 ? (
          <DetailRow label="Sessions" value={String(run.sessionCount)} />
        ) : null}
        <DetailRow label="Spans" value={String(run.spanCount)} />
      </div>
    </Panel>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex min-w-0 items-center justify-between gap-3 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="min-w-0 truncate text-right font-medium">{value}</span>
    </div>
  );
}

function CopyLine({ label, value }: { label: string; value: string | null }) {
  if (!value) return null;
  return (
    <div className="flex items-center justify-between gap-3">
      <span>{label}</span>
      <button
        className="flex min-w-0 items-center gap-2 font-mono text-foreground"
        onClick={async () => {
          await navigator.clipboard.writeText(value);
          toast.success({ title: "Copied" });
        }}
        type="button"
      >
        <span className="truncate">{value}</span>
        <Copy className="h-3.5 w-3.5 shrink-0" />
      </button>
    </div>
  );
}

function InfoLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-md border border-subtle bg-background-muted p-3">
      <p className="text-xs uppercase text-muted-foreground">{label}</p>
      <p className="mt-1 truncate font-mono text-xs">{value}</p>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const active = ["queued", "exporting", "running", "installing"].includes(status);
  const ok = ["completed", "installed", "connected"].includes(status);
  const bad = ["failed", "error", "cancelled", "interrupted"].includes(status);
  return (
    <Badge
      className="gap-1.5"
      variant={bad ? "status-failure" : ok ? "status-success" : active ? "status-running" : "outline"}
    >
      {active ? (
        <Clock3 className="h-3 w-3" />
      ) : ok ? (
        <CheckCircle2 className="h-3 w-3" />
      ) : bad ? (
        <AlertCircle className="h-3 w-3" />
      ) : null}
      {status.replaceAll("_", " ")}
    </Badge>
  );
}

function ProgressBar({ value }: { value: number }) {
  return (
    <div className="h-2 overflow-hidden rounded-full bg-muted">
      <div
        className="h-full rounded-full bg-detail-brand transition-[width]"
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  );
}

function toFacetOptions(options: FacetOption[] | undefined, allLabel: string) {
  return [
    { label: allLabel, value: "all" },
    ...(options ?? []).map((option) => ({
      count: option.count,
      label: option.label || option.value,
      value: option.value,
    })),
  ];
}

function startDateForRange(range: DateRange) {
  if (range === "all") return undefined;
  const ms =
    range === "1h"
      ? 60 * 60 * 1000
      : range === "24h"
        ? 24 * 60 * 60 * 1000
        : 7 * 24 * 60 * 60 * 1000;
  return new Date(Date.now() - ms);
}

function targetLabel(targetType: HaloRunTargetType) {
  return targetType === "session_group" ? "Session group" : "Trace group";
}

function formatTimestamp(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "unknown";
  return new Intl.DateTimeFormat(undefined, {
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    month: "short",
  }).format(date);
}
