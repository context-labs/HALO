from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelProviderConfig(BaseModel):
    """Connection settings for the OpenAI-compatible HTTP endpoint backing a run.

    HALO targets the OpenAI-compatible chat-completions surface (the de facto
    2025 standard exposed by OpenAI itself, OpenRouter, Anthropic's compat
    layer, vLLM, Together, Groq, Ollama, LM Studio, Azure OpenAI, etc.).

    Each field is independent: when ``None`` the underlying ``AsyncOpenAI``
    client falls back to the matching env var (``OPENAI_BASE_URL`` /
    ``OPENAI_API_KEY``). Setting one and not the other is supported — e.g.
    point ``base_url`` at OpenRouter while letting ``OPENAI_API_KEY`` from
    the environment supply the credential.
    """

    model_config = ConfigDict(extra="forbid")

    base_url: str | None = None
    api_key: str | None = None
