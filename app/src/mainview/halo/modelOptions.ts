import type { HaloProviderType } from "../../server/halo/types";

type ModelProviderType = Exclude<HaloProviderType, "custom">;

export const PROVIDER_MODEL_OPTIONS: Record<
  ModelProviderType,
  Array<{ label: string; value: string }>
> = {
  anthropic_compat: [
    { label: "Claude Opus 4.8", value: "claude-opus-4-8" },
    { label: "Claude Opus 4.7", value: "claude-opus-4-7" },
    { label: "Claude Opus 4.6", value: "claude-opus-4-6" },
    { label: "Claude Sonnet 4.6", value: "claude-sonnet-4-6" },
    { label: "Claude Sonnet 4.5", value: "claude-sonnet-4-5" },
    { label: "Claude Sonnet 4", value: "claude-sonnet-4" },
    { label: "Claude 3.5 Sonnet", value: "claude-3-5-sonnet" },
    { label: "Claude Haiku 4.5", value: "claude-haiku-4-5" },
    { label: "Claude 3.5 Haiku", value: "claude-3-5-haiku" },
  ],
  openai: [
    { label: "GPT-5.2", value: "gpt-5.2" },
    { label: "GPT-5.2 Pro", value: "gpt-5.2-pro" },
    { label: "GPT-5.1", value: "gpt-5.1" },
    { label: "GPT-5", value: "gpt-5" },
    { label: "GPT-5 Mini", value: "gpt-5-mini" },
    { label: "GPT-4.1", value: "gpt-4.1" },
    { label: "GPT-4.1 Mini", value: "gpt-4.1-mini" },
    { label: "GPT-4o", value: "gpt-4o" },
    { label: "GPT-4o Mini", value: "gpt-4o-mini" },
    { label: "o4 Mini", value: "o4-mini" },
  ],
  orcarouter: [
    { label: "OrcaRouter Auto", value: "orcarouter/auto" },
    { label: "GPT-5.5", value: "openai/gpt-5.5" },
    { label: "GPT-5.4 Mini", value: "openai/gpt-5.4-mini" },
    { label: "Claude Sonnet 4.6", value: "anthropic/claude-sonnet-4.6" },
    { label: "Claude Haiku 4.5", value: "anthropic/claude-haiku-4.5" },
    { label: "DeepSeek Chat", value: "deepseek/deepseek-chat" },
  ],
};

export function defaultModelForProvider(providerType: HaloProviderType) {
  if (providerType === "openai") return "gpt-5.2";
  if (providerType === "anthropic_compat") return "claude-sonnet-4-5";
  if (providerType === "orcarouter") return "openai/gpt-5.5";
  return "";
}

export function modelOptionsForProvider(providerType: HaloProviderType) {
  if (providerType === "custom") return null;
  return PROVIDER_MODEL_OPTIONS[providerType];
}
