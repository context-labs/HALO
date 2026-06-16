import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import {
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  DownloadCloud,
  FileUp,
  KeyRound,
  Loader2,
  Radio,
  RefreshCcw,
  Sparkles,
  XCircle,
} from "lucide-react";

import { Button, Input, cn, toast } from "~/lib/ui";
import { trpc } from "~/trpc";
import { LangfuseLogo, PhoenixLogo } from "~/tracing/ImportDataScreen";
import { APP_CATALYST_URL } from "../../desktop/commands";
import { openExternalUrl } from "../desktop/desktopBridge";

type OnboardingStep = "welcome" | "model" | "engine" | "traces";

const STEPS: Array<{ id: OnboardingStep; label: string }> = [
  { id: "welcome", label: "Welcome" },
  { id: "model", label: "Model" },
  { id: "engine", label: "Engine" },
  { id: "traces", label: "Traces" },
];

const PROVIDER_PRESETS = {
  anthropic_compat: {
    baseUrl: "https://api.anthropic.com/v1",
    keyPlaceholder: "sk-ant-…",
    model: "claude-sonnet-4-20250514",
    name: "Anthropic",
  },
  openai: {
    baseUrl: "https://api.openai.com/v1",
    keyPlaceholder: "sk-…",
    model: "gpt-4.1-mini",
    name: "OpenAI",
  },
} as const;

type PresetId = keyof typeof PROVIDER_PRESETS | "custom";

export function OnboardingPage() {
  const navigate = useNavigate();
  const utils = trpc.useUtils();
  const [step, setStep] = useState<OnboardingStep>("welcome");

  const engineQuery = trpc.halo.engine.status.useQuery(undefined, {
    refetchInterval: (query) =>
      query.state.data?.status === "installing" ? 1_500 : 5_000,
    // Users routinely switch away mid-onboarding (fetching an API key from
    // the browser) while the engine installs — keep the progress fresh even
    // when the window is hidden.
    refetchIntervalInBackground: true,
  });
  const providersQuery = trpc.halo.providers.list.useQuery();

  const installMutation = trpc.halo.engine.installOrUpdate.useMutation({
    async onSettled() {
      await utils.halo.engine.status.invalidate();
    },
  });

  // Download the engine in the background the moment onboarding opens, so the
  // install usually finishes while the user reads the welcome copy and enters
  // a key. Only auto-start from a clean slate — a previous error waits for an
  // explicit retry.
  const autoInstallStarted = useRef(false);
  const engineStatus = engineQuery.data?.status;
  useEffect(() => {
    if (autoInstallStarted.current) return;
    if (engineStatus !== "not_installed") return;
    autoInstallStarted.current = true;
    installMutation.mutate();
  }, [engineStatus, installMutation]);

  const completeMutation = trpc.onboarding.complete.useMutation({
    async onSuccess() {
      await utils.onboarding.get.invalidate();
    },
  });

  const finish = async (destination: "home" | "import-data") => {
    await completeMutation.mutateAsync();
    void navigate({ to: destination === "home" ? "/" : "/import-data" });
  };

  const stepIndex = STEPS.findIndex((item) => item.id === step);
  const goNext = () => {
    const next = STEPS[stepIndex + 1]?.id;
    if (next) setStep(next);
  };

  return (
    <main className="min-h-screen overflow-auto bg-background text-foreground">
      <div className="mx-auto flex min-h-screen w-full max-w-3xl flex-col px-8 py-10">
        <header className="flex items-center justify-between">
          <StepRail step={step} />
          <Button
            onClick={() => void finish("home")}
            size="sm"
            variant="ghost"
          >
            Skip setup
          </Button>
        </header>

        <div className="flex flex-1 flex-col justify-center py-10">
          {step === "welcome" ? <WelcomeStep onContinue={goNext} /> : null}
          {step === "model" ? (
            <ModelStep
              connectedProviderName={providersQuery.data?.[0]?.name ?? null}
              onContinue={goNext}
            />
          ) : null}
          {step === "engine" ? (
            <EngineStep
              installing={installMutation.isPending}
              onContinue={goNext}
              onRetry={() => installMutation.mutate()}
              status={engineQuery.data}
            />
          ) : null}
          {step === "traces" ? (
            <TracesStep onFinish={(destination) => void finish(destination)} />
          ) : null}
        </div>
      </div>
    </main>
  );
}

function WelcomeStep({ onContinue }: { onContinue: () => void }) {
  return (
    <div className="space-y-8">
      <div className="text-center">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-detail-brand">
          Welcome to HALO
        </p>
        <h1 className="mt-3 text-3xl font-semibold tracking-normal">
          See what your AI agents are really doing
        </h1>
        <p className="mx-auto mt-3 max-w-xl text-base text-muted-foreground">
          HALO is a local control room for AI agents. It collects your agent
          traces, lets you inspect every step, and runs AI analysis that finds
          the failures and bottlenecks for you. Everything stays on this
          machine.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <ValueCard
          description="Stream traces live from any OpenTelemetry or Catalyst exporter, or watch sessions unfold turn by turn."
          icon={<Radio className="h-5 w-5" />}
          title="Watch agents live"
        />
        <ValueCard
          description="Bring trace history in from Langfuse, Arize Phoenix, or a JSONL export. Nothing gets locked in."
          icon={<DownloadCloud className="h-5 w-5" />}
          title="Import from anywhere"
        />
        <ValueCard
          description="Analysis runs read your traces with your own model key and report failures, latency, and fixes."
          icon={<BrainCircuit className="h-5 w-5" />}
          title="Let HALO find issues"
        />
      </div>

      <div className="flex justify-center">
        <Button
          className="h-auto max-w-full whitespace-normal px-4 py-3 text-center leading-5"
          onClick={() => void openExternalUrl(APP_CATALYST_URL)}
          size="lg"
          type="button"
          variant="secondary"
        >
          Run HALO on Catalyst with $250 in free credits -&gt; Try now
        </Button>
      </div>

      <div className="rounded-xl border border-dashed border-border/60 p-5">
        <p className="text-sm font-medium">How it fits into your app</p>
        <div className="mt-4 flex flex-wrap items-center justify-center gap-2 text-sm text-muted-foreground">
          <FlowNode label="Your agent" />
          <FlowArrow />
          <FlowNode label="Traces into HALO" />
          <FlowArrow />
          <FlowNode label="HALO engine + your model key" />
          <FlowArrow />
          <FlowNode highlight label="Failures, bottlenecks, fixes" />
        </div>
      </div>

      <div className="flex justify-center">
        <Button onClick={onContinue} size="lg">
          Get started
          <ArrowRight className="ml-2 h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

function ModelStep({
  connectedProviderName,
  onContinue,
}: {
  connectedProviderName: string | null;
  onContinue: () => void;
}) {
  const utils = trpc.useUtils();
  const [preset, setPreset] = useState<PresetId>("openai");
  const [apiKey, setApiKey] = useState("");
  const [customBaseUrl, setCustomBaseUrl] = useState("");
  const [customModel, setCustomModel] = useState("");
  const [saving, setSaving] = useState(false);

  const saveMutation = trpc.halo.providers.save.useMutation();
  const testMutation = trpc.halo.providers.test.useMutation();

  const presetConfig = preset === "custom" ? null : PROVIDER_PRESETS[preset];
  const baseUrl = presetConfig?.baseUrl ?? customBaseUrl;
  const model = presetConfig?.model ?? customModel;
  const canSave = Boolean(apiKey.trim() && baseUrl.trim() && model.trim());

  const saveAndContinue = async () => {
    setSaving(true);
    try {
      const provider = await saveMutation.mutateAsync({
        apiKey: apiKey.trim(),
        baseUrl: baseUrl.trim(),
        model: model.trim(),
        name: presetConfig?.name ?? "Custom provider",
        providerType: preset === "custom" ? "custom" : preset,
      });
      await utils.halo.providers.list.invalidate();
      try {
        await testMutation.mutateAsync({ id: provider.id });
        toast.success({ title: `${provider.name} connected` });
      } catch (error) {
        // Some keys lack permission for the /models probe yet work fine for
        // completions, so a failed test warns instead of blocking.
        toast.warning({
          title: "Saved, but the connection test failed",
          description:
            error instanceof Error
              ? error.message
              : "You can re-test it any time in Settings.",
        });
      }
      onContinue();
    } catch (error) {
      toast.error({
        title: "Could not save the provider",
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      <StepHeading
        eyebrow="Step 2 of 4"
        title="Connect a model"
        description="HALO's analysis engine uses your API key to reason over your traces. The key is stored in the local database and never leaves this machine."
      />

      {connectedProviderName ? (
        <div className="flex items-center gap-2.5 rounded-md border border-detail-success/30 bg-detail-success/5 px-3 py-2.5 text-sm">
          <CheckCircle2 className="h-4 w-4 shrink-0 text-detail-success" />
          <span>
            <span className="font-medium">{connectedProviderName}</span> is
            already connected. You can add another key or continue.
          </span>
        </div>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2">
        <ProviderCard
          description="GPT models via api.openai.com"
          label="OpenAI"
          onSelect={() => setPreset("openai")}
          selected={preset === "openai"}
        />
        <ProviderCard
          description="Claude models via api.anthropic.com"
          label="Anthropic"
          onSelect={() => setPreset("anthropic_compat")}
          selected={preset === "anthropic_compat"}
        />
      </div>

      <div className="space-y-3">
        {preset === "custom" ? (
          <>
            <Input
              label="Base URL"
              hint="Any OpenAI-compatible /v1 endpoint."
              onChange={(event) => setCustomBaseUrl(event.currentTarget.value)}
              placeholder="https://my-gateway.example.com/v1"
              value={customBaseUrl}
            />
            <Input
              label="Model"
              onChange={(event) => setCustomModel(event.currentTarget.value)}
              placeholder="Model id"
              value={customModel}
            />
          </>
        ) : null}
        <Input
          label={`${presetConfig?.name ?? "Provider"} API key`}
          hint={
            presetConfig
              ? `Uses ${presetConfig.model} at ${presetConfig.baseUrl}. Both can be changed later in Settings.`
              : undefined
          }
          onChange={(event) => setApiKey(event.currentTarget.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && canSave && !saving) {
              void saveAndContinue();
            }
          }}
          placeholder={presetConfig?.keyPlaceholder ?? "API key"}
          type="password"
          value={apiKey}
        />
        <button
          className="text-xs text-muted-foreground underline-offset-4 transition hover:text-foreground hover:underline"
          onClick={() => setPreset(preset === "custom" ? "openai" : "custom")}
          type="button"
        >
          {preset === "custom"
            ? "Use an OpenAI or Anthropic preset instead"
            : "Use a custom OpenAI-compatible endpoint instead"}
        </button>
      </div>

      <div className="flex items-center justify-center gap-3">
        <Button disabled={!canSave || saving} onClick={() => void saveAndContinue()}>
          {saving ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <KeyRound className="mr-2 h-4 w-4" />
          )}
          Save and continue
        </Button>
        <Button onClick={onContinue} variant="ghost">
          {connectedProviderName ? "Continue" : "Skip for now"}
        </Button>
      </div>
    </div>
  );
}

const ENGINE_PHASES = [
  "Downloading the HALO repository…",
  "Installing Python dependencies…",
  "Verifying the engine…",
] as const;

function EngineStep({
  installing,
  onContinue,
  onRetry,
  status,
}: {
  installing: boolean;
  onContinue: () => void;
  onRetry: () => void;
  status:
    | {
        status: "not_installed" | "installing" | "installed" | "error";
        statusDetail: string | null;
        lastError: string | null;
        commitSha: string | null;
        checks: { git: string | null; uv: string | null; python: string | null };
      }
    | undefined;
}) {
  const state = status?.status ?? "not_installed";
  const busy = state === "installing" || installing;

  return (
    <div className="space-y-6">
      <StepHeading
        eyebrow="Step 3 of 4"
        title="The HALO engine"
        description="Analysis runs are powered by a local engine. HALO downloads it from GitHub and runs it on this machine with uv, so your traces never leave your computer."
      />

      <div className="rounded-xl border border-subtle bg-card p-6">
        {state === "installed" ? (
          <div className="flex flex-col items-center gap-3 py-4 text-center">
            <span className="grid h-12 w-12 place-items-center rounded-full bg-detail-success/10 text-detail-success">
              <CheckCircle2 className="h-6 w-6" />
            </span>
            <p className="text-lg font-semibold">The HALO engine is ready</p>
            <p className="text-sm text-muted-foreground">
              {status?.commitSha
                ? `Installed at commit ${status.commitSha}.`
                : "Installed and verified."}{" "}
              It started downloading the moment you opened this screen.
            </p>
          </div>
        ) : busy ? (
          <div className="space-y-4 py-2">
            <div className="flex items-center justify-center gap-2.5">
              <Loader2 className="h-5 w-5 animate-spin text-detail-brand" />
              <p className="text-lg font-semibold">Setting up the engine…</p>
            </div>
            <PhaseChecklist activeDetail={status?.statusDetail ?? null} />
            <p className="text-center text-xs text-muted-foreground">
              Usually takes a minute or two. You can keep going — this finishes
              in the background.
            </p>
          </div>
        ) : state === "error" ? (
          <div className="space-y-4">
            <div className="flex items-center gap-2.5">
              <XCircle className="h-5 w-5 shrink-0 text-detail-failure" />
              <p className="text-lg font-semibold">The install hit a problem</p>
            </div>
            {status?.lastError ? (
              <p className="max-h-28 overflow-auto rounded-md border border-detail-failure/30 bg-detail-failure/10 p-3 text-xs text-muted-foreground">
                {status.lastError}
              </p>
            ) : null}
            <PrerequisiteList checks={status?.checks} />
            <div className="flex justify-center">
              <Button onClick={onRetry} variant="secondary">
                <RefreshCcw className="mr-2 h-4 w-4" />
                Retry install
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 py-4 text-center">
            <p className="text-lg font-semibold">Engine not installed yet</p>
            <Button onClick={onRetry}>
              <DownloadCloud className="mr-2 h-4 w-4" />
              Download the engine
            </Button>
          </div>
        )}
      </div>

      <div className="flex items-center justify-center gap-3">
        <Button disabled={state !== "installed"} onClick={onContinue}>
          Continue
          <ArrowRight className="ml-2 h-4 w-4" />
        </Button>
        {state !== "installed" ? (
          <Button onClick={onContinue} variant="ghost">
            {busy ? "Continue while it installs" : "Skip for now"}
          </Button>
        ) : null}
      </div>
    </div>
  );
}

function PhaseChecklist({ activeDetail }: { activeDetail: string | null }) {
  // "Updating…" replaces "Downloading…" on re-installs; treat it as phase one.
  const normalizedDetail =
    activeDetail === "Updating the HALO repository…"
      ? ENGINE_PHASES[0]
      : activeDetail;
  const activeIndex = Math.max(
    0,
    ENGINE_PHASES.findIndex((phase) => phase === normalizedDetail),
  );
  return (
    <div className="mx-auto w-fit space-y-2">
      {ENGINE_PHASES.map((phase, index) => (
        <div
          className={cn(
            "flex items-center gap-2.5 text-sm",
            index < activeIndex
              ? "text-muted-foreground"
              : index === activeIndex
                ? "text-foreground"
                : "text-muted-foreground/50",
          )}
          key={phase}
        >
          {index < activeIndex ? (
            <CheckCircle2 className="h-4 w-4 text-detail-success" />
          ) : index === activeIndex ? (
            <Loader2 className="h-4 w-4 animate-spin text-detail-brand" />
          ) : (
            <span className="grid h-4 w-4 place-items-center">
              <span className="h-1.5 w-1.5 rounded-full bg-border" />
            </span>
          )}
          {phase.replace(/…$/, "")}
        </div>
      ))}
    </div>
  );
}

function PrerequisiteList({
  checks,
}: {
  checks: { git: string | null; uv: string | null; python: string | null } | undefined;
}) {
  const rows = [
    { hint: "https://git-scm.com", label: "git", value: checks?.git },
    { hint: "https://docs.astral.sh/uv", label: "uv", value: checks?.uv },
    { hint: "Python 3.12", label: "python", value: checks?.python },
  ];
  return (
    <div className="rounded-md border border-subtle bg-background-muted p-3">
      <p className="text-xs font-semibold uppercase text-muted-foreground">
        Prerequisites
      </p>
      <div className="mt-2 space-y-1.5">
        {rows.map((row) => (
          <div className="flex items-center gap-2 text-sm" key={row.label}>
            {row.value ? (
              <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-detail-success" />
            ) : (
              <XCircle className="h-3.5 w-3.5 shrink-0 text-detail-failure" />
            )}
            <span className="font-mono text-xs">{row.label}</span>
            <span className="truncate text-xs text-muted-foreground">
              {row.value ?? `missing — install from ${row.hint}`}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function TracesStep({
  onFinish,
}: {
  onFinish: (destination: "home" | "import-data") => void;
}) {
  return (
    <div className="space-y-6">
      <StepHeading
        eyebrow="Step 4 of 4"
        title="Bring in your traces"
        description="HALO comes alive once traces arrive. Import history from another tool, or point a live agent at the local endpoint."
      />

      <div className="grid gap-3 sm:grid-cols-3">
        <ImportOptionCard
          icon={<LangfuseLogo className="h-6 w-6" />}
          label="Langfuse"
          onClick={() => onFinish("import-data")}
        />
        <ImportOptionCard
          icon={<PhoenixLogo className="h-6 w-6" />}
          label="Phoenix"
          onClick={() => onFinish("import-data")}
        />
        <ImportOptionCard
          icon={<FileUp className="h-5 w-5 text-detail-brand" />}
          label="JSONL file"
          onClick={() => onFinish("import-data")}
        />
      </div>

      <div className="flex justify-center">
        <Button onClick={() => onFinish("home")} size="lg">
          <Sparkles className="mr-2 h-4 w-4" />
          Open HALO
        </Button>
      </div>
    </div>
  );
}

/**
 * Banner for analysis surfaces when run prerequisites are missing. Quietly
 * disappears once both the engine and a provider are configured.
 */
export function SetupNudgeBanner() {
  const engineQuery = trpc.halo.engine.status.useQuery();
  const providersQuery = trpc.halo.providers.list.useQuery();
  if (!engineQuery.data || !providersQuery.data) return null;

  const missingEngine = engineQuery.data.status !== "installed";
  const missingProvider = providersQuery.data.length === 0;
  if (!missingEngine && !missingProvider) return null;

  const missing =
    missingEngine && missingProvider
      ? "the HALO engine and a model API key"
      : missingEngine
        ? "the HALO engine"
        : "a model API key";

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-detail-brand/30 bg-detail-brand/5 px-4 py-3">
      <div className="flex min-w-0 items-center gap-2.5 text-sm">
        <Sparkles className="h-4 w-4 shrink-0 text-detail-brand" />
        <span>
          Analysis runs need {missing}. Finish setup to start finding failures
          in your traces.
        </span>
      </div>
      <Button asChild size="sm" variant="secondary">
        <Link to="/welcome">Finish setup</Link>
      </Button>
    </div>
  );
}

function StepHeading({
  description,
  eyebrow,
  title,
}: {
  description: string;
  eyebrow: string;
  title: string;
}) {
  return (
    <div className="text-center">
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-detail-brand">
        {eyebrow}
      </p>
      <h1 className="mt-2 text-2xl font-semibold tracking-normal">{title}</h1>
      <p className="mx-auto mt-2 max-w-xl text-sm leading-6 text-muted-foreground">
        {description}
      </p>
    </div>
  );
}

function ValueCard({
  description,
  icon,
  title,
}: {
  description: string;
  icon: ReactNode;
  title: string;
}) {
  return (
    <div className="rounded-[18px] border border-border/70 bg-card p-5">
      <span className="grid h-10 w-10 place-items-center rounded-xl bg-detail-brand/10 text-detail-brand">
        {icon}
      </span>
      <h2 className="mt-4 text-base font-semibold">{title}</h2>
      <p className="mt-1.5 text-sm leading-6 text-muted-foreground">{description}</p>
    </div>
  );
}

function FlowNode({ highlight, label }: { highlight?: boolean; label: string }) {
  return (
    <span
      className={cn(
        "rounded-full border px-3 py-1.5 text-xs",
        highlight
          ? "border-detail-brand/40 bg-detail-brand/10 text-detail-brand"
          : "border-subtle bg-background-muted",
      )}
    >
      {label}
    </span>
  );
}

function FlowArrow() {
  return <ArrowRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground/60" />;
}

function ProviderCard({
  description,
  label,
  onSelect,
  selected,
}: {
  description: string;
  label: string;
  onSelect: () => void;
  selected: boolean;
}) {
  return (
    <button
      className={cn(
        "rounded-[18px] border bg-card p-4 text-left transition hover:bg-card-hover/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        selected
          ? "border-detail-brand/50 ring-1 ring-detail-brand/30"
          : "border-border/70 hover:border-border",
      )}
      onClick={onSelect}
      type="button"
    >
      <div className="flex items-center justify-between gap-2">
        <p className="font-semibold">{label}</p>
        {selected ? (
          <CheckCircle2 className="h-4 w-4 shrink-0 text-detail-brand" />
        ) : null}
      </div>
      <p className="mt-1 text-sm text-muted-foreground">{description}</p>
    </button>
  );
}

function ImportOptionCard({
  icon,
  label,
  onClick,
}: {
  icon: ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      className="group flex items-center gap-3 rounded-[18px] border border-border/70 bg-card p-4 text-left transition hover:border-border hover:bg-card-hover/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      onClick={onClick}
      type="button"
    >
      <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-background-muted">
        {icon}
      </span>
      <span className="flex min-w-0 items-center gap-1.5 text-sm font-medium">
        {label}
        <ArrowRight className="h-3.5 w-3.5 -translate-x-1 text-muted-foreground opacity-0 transition group-hover:translate-x-0 group-hover:opacity-100" />
      </span>
    </button>
  );
}

function StepRail({ step }: { step: OnboardingStep }) {
  const activeIndex = STEPS.findIndex((item) => item.id === step);
  return (
    <div className="flex items-center gap-2">
      {STEPS.map((item, index) => (
        <div className="flex items-center gap-2" key={item.id}>
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
          <span className="hidden text-xs text-muted-foreground sm:inline">
            {item.label}
          </span>
          {index < STEPS.length - 1 ? (
            <span className="h-px w-5 bg-border/50" />
          ) : null}
        </div>
      ))}
    </div>
  );
}
