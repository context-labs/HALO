import { useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import {
  Copy,
  DownloadCloud,
  Loader2,
  Palette,
  Save,
  Settings,
  Trash2,
} from "lucide-react";

import {
  Button,
  Dialog,
  Input,
  ThemeToggle,
  cn,
  toast,
} from "~/lib/ui";
import { trpc } from "~/trpc";
import { WorkspaceNav } from "~/workspace/WorkspaceNav";
import { AppHeader } from "~/components/AppHeader";
import { FilterSelect } from "~/components/FilterSelect";
import { StatusBadge } from "~/components/StatusBadge";

export function SettingsPage() {
  const utils = trpc.useUtils();
  const navigate = useNavigate();
  const engineQuery = trpc.halo.engine.status.useQuery();
  const providersQuery = trpc.halo.providers.list.useQuery();
  const telemetryInfoQuery = trpc.telemetry.info.useQuery();
  const [providerType, setProviderType] = useState<"openai" | "anthropic_compat" | "custom">("openai");
  const [name, setName] = useState("OpenAI");
  const [baseUrl, setBaseUrl] = useState("https://api.openai.com/v1");
  const [model, setModel] = useState("gpt-4.1-mini");
  const [apiKey, setApiKey] = useState("");
  const [providerPendingDelete, setProviderPendingDelete] = useState<
    { id: string; name: string } | null
  >(null);

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
  const replayOnboardingMutation = trpc.onboarding.reset.useMutation({
    async onSuccess() {
      await utils.onboarding.get.invalidate();
      void navigate({ to: "/welcome" });
    },
  });
  const deleteProviderMutation = trpc.halo.providers.delete.useMutation({
    async onSuccess() {
      setProviderPendingDelete(null);
      await utils.halo.providers.list.invalidate();
    },
  });

  const providers = providersQuery.data ?? [];
  const status = engineQuery.data;
  const telemetryInfo = telemetryInfoQuery.data;

  return (
    <main className="min-h-screen bg-background text-foreground">
      <AppHeader
        icon={<Settings className="h-4 w-4 text-detail-brand" />}
        title="Settings"
      />
      <div className="grid min-h-[calc(100vh-3.5rem)] grid-cols-[14rem_minmax(0,1fr)] pt-14">
        <WorkspaceNav active="settings" />
        <section className="min-w-0 overflow-auto">
          <div className="mx-auto flex max-w-6xl flex-col gap-6 p-8">
            <div>
              <h1 className="text-2xl tracking-normal">Settings</h1>
              <p className="mt-1 text-sm text-muted-foreground">
                Configure the local HALO engine and model providers.
              </p>
            </div>
            <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_420px]">
            <div className="space-y-5">
              <section className="rounded-xl border border-subtle bg-card">
                <div className="flex items-start gap-3 border-b border-subtle p-5">
                  <div className="grid h-9 w-9 shrink-0 place-items-center rounded-md border border-subtle bg-background-muted">
                    <Palette className="h-4 w-4 text-detail-brand" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <h1 className="text-xl font-semibold">Workspace</h1>
                    <p className="mt-1 text-sm text-muted-foreground">
                      Appearance and local runtime details for this desktop app.
                    </p>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <Button
                      disabled={replayOnboardingMutation.isPending}
                      onClick={() => replayOnboardingMutation.mutate()}
                      size="sm"
                      variant="outline"
                    >
                      Replay onboarding
                    </Button>
                    <ThemeToggle
                      trigger={
                        <Button size="sm" variant="outline">
                          <Palette className="mr-2 h-4 w-4" />
                          Theme
                        </Button>
                      }
                    />
                  </div>
                </div>
                <div className="divide-y divide-subtle px-5">
                  <DefinitionRow
                    copyable
                    label="Database path"
                    value={telemetryInfo?.dbPath ?? "data/halo-canvas.sqlite"}
                  />
                  <DefinitionRow
                    copyable
                    label="Ingest endpoint"
                    value={telemetryInfo?.ingestUrl ?? "http://127.0.0.1:8799/v1/traces"}
                  />
                  <DefinitionRow
                    copyable
                    label="Live socket"
                    value={telemetryInfo?.liveUrl ?? "ws://127.0.0.1:8800"}
                  />
                  <DefinitionRow
                    label="Stored telemetry"
                    value={`${telemetryInfo?.traceCount ?? 0} traces · ${telemetryInfo?.spanCount ?? 0} spans`}
                  />
                </div>
              </section>

              <section className="rounded-xl border border-subtle bg-card">
                <div className="flex items-start justify-between gap-4 border-b border-subtle p-5">
                  <div>
                    <h1 className="text-xl font-semibold">HALO engine</h1>
                    <p className="mt-1 text-sm text-muted-foreground">
                      HALO clones the engine locally and runs it with uv.
                    </p>
                  </div>
                  <StatusBadge status={status?.status ?? "not_installed"} />
                </div>
                <div className="divide-y divide-subtle px-5">
                  <DefinitionRow
                    copyable
                    label="Install path"
                    value={status?.installPath ?? "data/halo-engine"}
                  />
                  <DefinitionRow
                    copyable
                    label="Repo"
                    value={status?.repoUrl ?? "https://github.com/context-labs/HALO"}
                  />
                  <DefinitionRow label="Commit" value={status?.commitSha ?? "not installed"} />
                  <DefinitionRow label="Python" value={status?.checks.python ?? "missing"} />
                  <DefinitionRow label="uv" value={status?.checks.uv ?? "missing"} />
                  <DefinitionRow label="git" value={status?.checks.git ?? "missing"} />
                </div>
                {status?.lastError ? (
                  <div className="mx-5 mb-4 rounded-md border border-destructive-border bg-destructive/5 p-3 text-sm text-destructive">
                    {status.lastError}
                  </div>
                ) : null}
                <div className="flex items-center justify-between gap-4 border-t border-subtle p-5">
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

              <section className="rounded-xl border border-subtle bg-card">
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
                            aria-label={`Delete provider ${provider.name}`}
                            onClick={() =>
                              setProviderPendingDelete({
                                id: provider.id,
                                name: provider.name,
                              })
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

            <section className="h-fit rounded-xl border border-subtle bg-card">
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
                  label="Name"
                  onChange={(event) => setName(event.currentTarget.value)}
                  placeholder="Provider name"
                  value={name}
                />
                <Input
                  hint="OpenAI-compatible /v1 endpoint."
                  label="Base URL"
                  onChange={(event) => setBaseUrl(event.currentTarget.value)}
                  placeholder="https://api.openai.com/v1"
                  value={baseUrl}
                />
                <Input
                  label="Model"
                  onChange={(event) => setModel(event.currentTarget.value)}
                  placeholder="Model id"
                  value={model}
                />
                <Input
                  hint="Stored locally in SQLite. It never leaves this machine."
                  label="API key"
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
          </div>
        </section>
      </div>

      <Dialog
        cancelTitle="Cancel"
        confirmButtonVariant="destructive"
        confirmTitle="Delete provider"
        dialogDescription={`This removes "${providerPendingDelete?.name ?? ""}" and its stored API key from the local database.`}
        dialogTitle="Delete this provider?"
        disabled={deleteProviderMutation.isPending}
        loading={deleteProviderMutation.isPending}
        onConfirm={() => {
          if (providerPendingDelete) {
            deleteProviderMutation.mutate({ id: providerPendingDelete.id });
          }
        }}
        onOpenChange={(open) => {
          if (!open) setProviderPendingDelete(null);
        }}
        open={Boolean(providerPendingDelete)}
      />
    </main>
  );
}

function DefinitionRow({
  copyable,
  label,
  value,
}: {
  copyable?: boolean;
  label: string;
  value: string;
}) {
  return (
    <div className="flex min-w-0 items-center justify-between gap-4 py-3">
      <span className="shrink-0 text-sm text-muted-foreground">{label}</span>
      <span className="flex min-w-0 items-center gap-1.5">
        <span className="truncate font-mono text-xs" title={value}>
          {value}
        </span>
        {copyable ? (
          <Button
            aria-label={`Copy ${label}`}
            className={cn("h-6 w-6 shrink-0 text-muted-foreground")}
            onClick={async () => {
              await navigator.clipboard.writeText(value);
              toast.success({ title: `${label} copied` });
            }}
            size="icon"
            variant="ghost"
          >
            <Copy className="h-3 w-3" />
          </Button>
        ) : null}
      </span>
    </div>
  );
}
