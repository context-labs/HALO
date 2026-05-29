from __future__ import annotations

from unittest.mock import MagicMock

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from engine.agents.openai_sdk_client import (
    build_async_openai_client,
    is_retriable_llm_error,
    omit,
)
from engine.model_provider_config import ModelProviderConfig


def test_build_async_openai_client_returns_async_openai() -> None:
    cfg = ModelProviderConfig(
        base_url="https://example.com/v1",
        api_key="sk-test",
        default_headers={"X-Test": "1"},
    )
    client = build_async_openai_client(cfg)
    assert isinstance(client, AsyncOpenAI)


def test_build_async_openai_client_with_empty_config() -> None:
    """Empty config is allowed — AsyncOpenAI falls back to env vars."""
    client = build_async_openai_client(ModelProviderConfig())
    assert isinstance(client, AsyncOpenAI)


def test_is_retriable_classifies_connection_error() -> None:
    exc = MagicMock(spec=APIConnectionError)
    assert is_retriable_llm_error(exc) is True


def test_is_retriable_classifies_timeout() -> None:
    exc = MagicMock(spec=APITimeoutError)
    assert is_retriable_llm_error(exc) is True


def test_is_retriable_classifies_rate_limit() -> None:
    exc = MagicMock(spec=RateLimitError)
    assert is_retriable_llm_error(exc) is True


def test_is_retriable_classifies_5xx_status_as_retriable() -> None:
    exc = MagicMock(spec=APIStatusError)
    exc.status_code = 503
    assert is_retriable_llm_error(exc) is True


def test_is_retriable_classifies_4xx_status_as_non_retriable() -> None:
    exc = MagicMock(spec=APIStatusError)
    exc.status_code = 400
    assert is_retriable_llm_error(exc) is False


def test_is_retriable_classifies_unrelated_exception_as_non_retriable() -> None:
    assert is_retriable_llm_error(RuntimeError("boom")) is False


def test_omit_is_reexported_openai_sentinel() -> None:
    """``omit`` is the openai SDK's "don't send this param" sentinel.
    Re-exporting it lets compactor.py keep its current behavior without
    importing from openai directly.
    """
    from openai import omit as openai_omit

    assert omit is openai_omit


def test_openai_provider_is_reexported_sdk_type() -> None:
    """``OpenAIProvider`` is re-exported so the engine can wire HALO's client
    into ``RunConfig.model_provider`` per call without importing the SDK
    provider module directly outside this boundary.
    """
    from agents.models.openai_provider import OpenAIProvider as SdkOpenAIProvider

    from engine.agents.openai_sdk_client import OpenAIProvider

    assert OpenAIProvider is SdkOpenAIProvider
