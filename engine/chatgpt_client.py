"""Enforce the SIWC HTTP contract at the wire, across all HALO model calls."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from agents import FunctionTool, Model, Tool, tool_namespace
from agents.models.openai_provider import OpenAIProvider
from agents.models.openai_responses import OpenAIResponsesModel
from openai import APIConnectionError, AsyncOpenAI
from openai.types.responses import ResponseStreamEvent

from engine.chatgpt_auth import RESOURCE, ChatGPTAuthError, CredentialStore, access_token
from engine.model_config import ModelConfig

NAMESPACE = "halo"
UNSUPPORTED_FIELDS = frozenset(
    "background conversation max_output_tokens max_tool_calls metadata moderation multi_agent "
    "prompt prompt_cache_retention safety_identifier temperature top_logprobs top_p truncation "
    "user previous_response_id".split()
)


class ChatGPTInferenceError(RuntimeError):
    """Terminal plan/protocol error. Never retry through a paid API-key fallback."""


def plan_request(body: dict[str, Any]) -> dict[str, Any]:
    """SDK defaults cannot opt this route back into storage or unsupported fields."""
    body = {key: value for key, value in body.items() if key not in UNSUPPORTED_FIELDS}
    body.update(store=False, stream=True)
    inputs = body.get("input")
    if not isinstance(inputs, list):
        raise ChatGPTInferenceError("ChatGPT plan requests require an input array.")
    body["input"] = [dict(item) for item in inputs]
    for item in body["input"]:
        if item.get("role") == "system":
            item["role"] = "developer"
        # HALO's portable local history intentionally stores bare function names.
        # Restore the namespace when replaying its tool turns over this route.
        if item.get("type") == "function_call":
            item.setdefault("namespace", NAMESPACE)
    for tool in body.get("tools", []):
        if tool.get("type") != "namespace":
            raise ChatGPTInferenceError("ChatGPT mode requires HALO tools in a namespace.")
        if any(inner.get("type") not in ("function", "custom") for inner in tool.get("tools", [])):
            raise ChatGPTInferenceError("Unsupported hosted tool in ChatGPT mode.")
    return body


class PlanTransport(httpx.AsyncBaseTransport):
    """Auth after SDK preparation; allow only the documented public endpoints."""

    def __init__(self, store: CredentialStore, transport: httpx.AsyncBaseTransport | None = None):
        self.store = store
        with store.lock:
            self.account = store.selected(store.read())["client_id"]
        self.transport = transport or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        allowed = (request.method, url.path) in {("POST", "/v1/responses"), ("GET", "/v1/models")}
        if (
            url.scheme != "https"
            or url.host != "api.openai.com"
            or url.port not in (None, 443)
            or not allowed
        ):
            raise ChatGPTInferenceError(
                "ChatGPT mode only permits the public models/Responses endpoints."
            )
        content = await request.aread()
        if request.method == "POST":
            content = json.dumps(plan_request(json.loads(content))).encode()
        headers = httpx.Headers(request.headers)
        headers["Authorization"] = "Bearer " + await asyncio.to_thread(
            access_token, self.store, self.account
        )
        # Never inherit API-project billing headers from an unrelated shell environment.
        for key in ("OpenAI-Organization", "OpenAI-Project", "Content-Length"):
            headers.pop(key, None)
        prepared = httpx.Request(
            request.method, url, headers=headers, content=content, extensions=request.extensions
        )
        response = await self.transport.handle_async_request(prepared)
        if response.status_code >= 400:
            await response.aread()
            try:
                error = response.json().get("error", {})
                code = error.get("code") if isinstance(error, dict) else None
            except (ValueError, AttributeError):
                code = None
            if code in {
                "subscription_sharing_usage_limit_exceeded",
                "subscription_sharing_unsupported_capability",
                "subscription_sharing_route_not_supported",
            }:
                await response.aclose()
                raise ChatGPTInferenceError(
                    f"ChatGPT request rejected: {code} (HTTP {response.status_code}; request ID {response.headers.get('x-request-id', 'unavailable')})."
                )
        return response

    async def aclose(self) -> None:
        await self.transport.aclose()


class ChatGPTClient(AsyncOpenAI):
    def __init__(
        self, store: CredentialStore, *, transport: httpx.AsyncBaseTransport | None = None
    ):
        super().__init__(
            api_key="halo-owned-oauth-session",
            base_url=RESOURCE,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=PlanTransport(store, transport), timeout=120),
        )

    async def request(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return await super().request(*args, **kwargs)
        except APIConnectionError as exc:
            # The SDK wraps all custom-transport exceptions as connection errors.
            # Restore terminal authorization/quota/contract errors before HALO's
            # retry classifier sees them and unnecessarily repeats the request.
            if isinstance(exc.__cause__, (ChatGPTAuthError, ChatGPTInferenceError)):
                raise exc.__cause__ from None
            raise


async def completed_stream(
    stream: AsyncIterator[ResponseStreamEvent],
) -> AsyncIterator[ResponseStreamEvent]:
    completed = False
    async for event in stream:
        kind = event.type
        if kind in ("response.failed", "response.incomplete", "error", "response.error"):
            response = getattr(event, "response", None)
            error = getattr(response, "error", None)
            code = getattr(error, "code", None) or getattr(event, "code", None)
            # Restrict diagnostic codes to identifiers, never echo server error prose.
            safe_code = code if isinstance(code, str) and code.replace("_", "").isalnum() else kind
            raise ChatGPTInferenceError(f"ChatGPT inference did not complete: {safe_code}.")
        if kind == "response.completed":
            completed = True
        yield event
    if not completed:
        raise ChatGPTInferenceError("ChatGPT stream ended without response.completed.")


class PlanResponsesModel(OpenAIResponsesModel):
    async def stream_response(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ResponseStreamEvent]:
        stream = super().stream_response(*args, **kwargs)
        try:
            async for event in completed_stream(stream):
                yield event
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()

    async def get_response(self, *args: Any, **kwargs: Any):
        raise ChatGPTInferenceError("ChatGPT mode requires streamed inference.")


class ChatGPTProvider(OpenAIProvider):
    def get_model(self, model_name: str | None) -> Model:
        if model_name is None:
            raise ChatGPTInferenceError("Select an account model with halo-chatgpt models.")
        return PlanResponsesModel(model=model_name, openai_client=self._get_client())


def plan_tools(tools: list[Tool]) -> list[Tool]:
    if any(not isinstance(tool, FunctionTool) for tool in tools):
        raise ChatGPTInferenceError(
            "HALO ChatGPT mode currently supports local function tools only."
        )
    functions = [tool for tool in tools if isinstance(tool, FunctionTool)]
    return list(
        tool_namespace(name=NAMESPACE, description="HALO trace analysis tools", tools=functions)
    )


async def plan_text(client: ChatGPTClient, model: ModelConfig, instructions: str, text: str) -> str:
    stream = await client.responses.create(
        model=model.name,
        instructions=instructions,
        input=[{"role": "user", "content": text}],
        store=False,
        stream=True,
    )
    chunks: list[str] = []
    try:
        async for event in completed_stream(stream):
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)
    finally:
        await stream.close()
    return "".join(chunks).strip()
