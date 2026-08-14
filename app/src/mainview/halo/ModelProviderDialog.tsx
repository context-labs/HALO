import { useState } from "react";
import { Loader2, Save } from "lucide-react";

import { FilterSelect } from "~/components/FilterSelect";
import { Button, Dialog, Input, toast } from "~/lib/ui";
import { trpc } from "~/trpc";

type ProviderType = "openai" | "anthropic_compat" | "orcarouter" | "custom";

const PROVIDER_BASE_URL_HINTS: Record<ProviderType, string> = {
  anthropic_compat: "Anthropic /v1 endpoint.",
  custom: "OpenAI-compatible /v1 endpoint.",
  openai: "OpenAI /v1 endpoint.",
  orcarouter: "OrcaRouter /v1 endpoint.",
};

export function ModelProviderDialog({
  onOpenChange,
  onSaved,
  open,
  submitLabel = "Save provider",
}: {
  onOpenChange: (open: boolean) => void;
  onSaved?: () => void | Promise<void>;
  open: boolean;
  submitLabel?: string;
}) {
  const utils = trpc.useUtils();
  const [providerType, setProviderType] = useState<ProviderType>("openai");
  const [name, setName] = useState("OpenAI");
  const [baseUrl, setBaseUrl] = useState("https://api.openai.com/v1");
  const [apiKey, setApiKey] = useState("");

  const saveProviderMutation = trpc.halo.providers.save.useMutation({
    async onSuccess() {
      toast.success({ title: "Provider saved" });
      setApiKey("");
      onOpenChange(false);
      await utils.halo.providers.list.invalidate();
      await onSaved?.();
    },
    onError(error) {
      toast.error({ title: "Could not save provider", description: error.message });
    },
  });

  return (
    <Dialog
      className="sm:!max-w-[520px] md:!w-[520px]"
      dialogDescription="HALO supports OpenAI and Anthropic-compatible endpoints."
      dialogTitle="Add provider"
      footer={
        <div className="flex justify-end border-t border-subtle px-6 py-4">
          <Button
            disabled={
              saveProviderMutation.isPending ||
              !name.trim() ||
              !baseUrl.trim() ||
              !apiKey.trim()
            }
            onClick={() =>
              saveProviderMutation.mutate({
                apiKey,
                baseUrl,
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
            {submitLabel}
          </Button>
        </div>
      }
      hideConfirmButton
      maxWidth={520}
      onConfirm={() => undefined}
      onOpenChange={onOpenChange}
      open={open}
    >
      <div className="space-y-4">
        <FilterSelect
          label="Provider"
          labelClassName="text-sm font-medium normal-case leading-none text-foreground"
          onChange={(value) => {
            const next = value as ProviderType;
            setProviderType(next);
            if (next === "openai") {
              setName("OpenAI");
              setBaseUrl("https://api.openai.com/v1");
            } else if (next === "anthropic_compat") {
              setName("Anthropic");
              setBaseUrl("https://api.anthropic.com/v1");
            } else if (next === "orcarouter") {
              setName("OrcaRouter");
              setBaseUrl("https://api.orcarouter.ai/v1");
            } else {
              setName("Custom provider");
              setBaseUrl("");
            }
          }}
          options={[
            { label: "OpenAI", value: "openai" },
            { label: "Anthropic", value: "anthropic_compat" },
            { label: "OrcaRouter", value: "orcarouter" },
            { label: "Custom OpenAI-compatible", value: "custom" },
          ]}
          triggerClassName="text-xs"
          value={providerType}
        />
        <Input
          label="Name"
          onChange={(event) => setName(event.currentTarget.value)}
          placeholder="Provider name"
          value={name}
        />
        <Input
          hint={PROVIDER_BASE_URL_HINTS[providerType]}
          label="Base URL"
          onChange={(event) => setBaseUrl(event.currentTarget.value)}
          placeholder="https://api.openai.com/v1"
          value={baseUrl}
        />
        <Input
          hint="Stored locally in SQLite. It never leaves this machine."
          label="API key"
          onChange={(event) => setApiKey(event.currentTarget.value)}
          placeholder="API key"
          type="password"
          value={apiKey}
        />
      </div>
    </Dialog>
  );
}
