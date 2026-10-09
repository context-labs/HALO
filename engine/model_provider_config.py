from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class ModelProviderConfig(BaseModel):
    """Connection settings for the OpenAI-compatible HTTP endpoint backing a run.

    ``auth_mode="chatgpt"`` selects HALO's independent OAuth session at
    ``auth_file`` (or its app config path), fixes the public Responses endpoint,
    and rejects API keys, alternate URLs and custom headers. The following
    environment fallback behavior applies to the default ``api_key`` mode.

    HALO targets the OpenAI-compatible chat-completions surface (the de facto
    2025 standard exposed by OpenAI itself, OpenRouter, Anthropic's compat
    layer, vLLM, Together, Groq, Ollama, LM Studio, Azure OpenAI, etc.).

    Each field is independent: when ``None`` the underlying ``AsyncOpenAI``
    client falls back to the matching env var (``OPENAI_BASE_URL`` /
    ``OPENAI_API_KEY``). When ``OPENAI_BASE_URL`` is unset, the endpoint is
    OpenAI's API base URL, ``https://api.openai.com/v1``. Setting one and not
    the other is supported — e.g. point ``base_url`` at OpenRouter while
    letting ``OPENAI_API_KEY`` from the environment supply the credential.
    """

    model_config = ConfigDict(extra="forbid")

    base_url: str | None = None
    api_key: str | None = None
    default_headers: dict[str, str] | None = None
    auth_mode: Literal["api_key", "chatgpt"] = "api_key"
    auth_file: Path | None = None

    @model_validator(mode="after")
    def validate_chatgpt(self) -> ModelProviderConfig:
        if self.auth_mode == "chatgpt" and (self.base_url or self.api_key or self.default_headers):
            raise ValueError(
                "ChatGPT mode cannot be combined with API keys, base URLs, or headers."
            )
        return self
