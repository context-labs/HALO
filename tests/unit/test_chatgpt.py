from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen

import httpx
import jwt
import pytest
from agents import Agent, RunConfig, Runner, function_tool
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import ValidationError
from typer.testing import CliRunner

import engine.chatgpt_auth as auth
from engine.agents.agent_context_items import AgentContextItem
from engine.agents.compactor import compact
from engine.chatgpt_client import (
    UNSUPPORTED_FIELDS,
    ChatGPTClient,
    ChatGPTInferenceError,
    ChatGPTProvider,
    PlanTransport,
    plan_request,
    plan_text,
    plan_tools,
)
from engine.model_config import ModelConfig
from engine.model_provider_config import ModelProviderConfig
from engine.tools.synthesis_tool import SynthesisTool, SynthesizeTracesArguments
from halo_cli.chatgpt import cli as auth_cli


@pytest.fixture
def store(tmp_path):
    result = auth.CredentialStore(tmp_path / "credentials" / "chatgpt.json")
    result.write(
        {
            "ext_agent_host_id": "urn:uuid:test",
            "active": "oaiapp_test",
            "accounts": {
                "oaiapp_test": {
                    "client_id": "oaiapp_test",
                    "subject": "user-1",
                    "issuer": auth.ISSUER,
                    "access_token": "test-access",
                    "refresh_token": "test-refresh",
                    "id_token": "test-id",
                    "expires_at": time.time() + 3600,
                    "scopes": [auth.PLAN_SCOPE],
                }
            },
        }
    )
    return result


@pytest.mark.parametrize(
    "params,client",
    [
        ({"state": "bad", "error": "access_denied"}, "dynamic_agent_client"),
        ({"state": "ok", "error": "access_denied"}, "dynamic_agent_client"),
        ({"state": "ok", "code": "c"}, "dynamic_agent_client"),
        ({"state": "ok", "code": "c", "client_id": "other"}, "oaiapp_test"),
    ],
)
def test_callback_rejects_unbound_or_denied_registration(params, client):
    with pytest.raises(auth.ChatGPTAuthError):
        auth.callback_client(params, state="ok", client_id=client)


def test_returning_callback_can_omit_client():
    assert auth.callback_client(
        {"state": "ok", "code": "c"}, state="ok", client_id="oaiapp_test"
    ) == ("oaiapp_test", "c")


@pytest.fixture
def signing_key(monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setattr(
        auth.jwt,
        "PyJWKClient",
        lambda *a, **kw: SimpleNamespace(
            get_signing_key_from_jwt=lambda token: SimpleNamespace(key=key.public_key())
        ),
    )
    return key


@pytest.mark.parametrize(
    "changed",
    [
        {"nonce": "wrong"},
        {"aud": "wrong"},
        {"iss": "https://other.test"},
        {"exp": 1},
        {"sub": ""},
    ],
)
def test_id_token_validation_rejects_wrong_claims(signing_key, changed):
    claims = dict(
        iss=auth.ISSUER, aud="oaiapp_test", exp=time.time() + 300, sub="user-1", nonce="expected"
    )
    claims.update(changed)
    token = jwt.encode(claims, signing_key, algorithm="RS256")
    with pytest.raises(auth.ChatGPTAuthError):
        auth._validated_identity(token, "oaiapp_test", "expected")


def test_id_token_rejects_bad_signature(signing_key):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = jwt.encode(
        dict(
            iss=auth.ISSUER,
            aud="oaiapp_test",
            exp=time.time() + 300,
            sub="user-1",
            nonce="expected",
        ),
        other,
        algorithm="RS256",
    )
    with pytest.raises(auth.ChatGPTAuthError):
        auth._validated_identity(token, "oaiapp_test", "expected")


def test_login_pkce_registration_and_returning_host(tmp_path, signing_key, monkeypatch):
    store = auth.CredentialStore(tmp_path / "chatgpt.json")
    queries, forms, threads = [], [], []

    def browser(url):
        query = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        queries.append(query)
        callback = (
            query["redirect_uri"]
            + "?"
            + urlencode(dict(state=query["state"], code="code", client_id="oaiapp_test"))
        )
        thread = threading.Thread(target=lambda: urlopen(callback, timeout=5).close())
        thread.start()
        threads.append(thread)
        return True

    def tokens(form):
        forms.append(form)
        token = jwt.encode(
            dict(
                iss=auth.ISSUER,
                aud="oaiapp_test",
                exp=time.time() + 300,
                sub="user-1",
                nonce=queries[-1]["nonce"],
            ),
            signing_key,
            algorithm="RS256",
        )
        return dict(
            access_token="test-access",
            refresh_token="test-refresh",
            id_token=token,
            token_type="Bearer",
            expires_in=3600,
            scope=auth.PLAN_SCOPE,
        )

    monkeypatch.setattr(auth.webbrowser, "open", browser)
    monkeypatch.setattr(auth, "_token_request", tokens)
    assert auth.login(store, timeout=5) == "oaiapp_test"
    assert auth.login(store, timeout=5) == "oaiapp_test"
    for thread in threads:
        thread.join()
    assert queries[0]["client_id"] == "dynamic_agent_client"
    assert queries[1]["client_id"] == "oaiapp_test"
    assert "agent_name_hint" not in queries[1]
    assert queries[0]["ext_agent_host_id"] == queries[1]["ext_agent_host_id"]
    assert queries[0]["state"] != queries[1]["state"]
    assert queries[1]["id_token_hint"]
    for query, form in zip(queries, forms):
        assert form["client_id"] == "oaiapp_test"
        assert form["redirect_uri"] == query["redirect_uri"]
        assert form["resource"] == auth.RESOURCE
        import base64
        import hashlib

        assert query["code_challenge"] == base64.urlsafe_b64encode(
            hashlib.sha256(form["code_verifier"].encode()).digest()
        ).decode().rstrip("=")
    if os.name != "nt":
        assert store.path.stat().st_mode & 0o777 == 0o600


def test_refresh_serialized_and_rotated(store, monkeypatch):
    data = store.read()
    data["accounts"]["oaiapp_test"]["expires_at"] = 0
    store.write(data)
    calls = []

    def tokens(form):
        calls.append(form)
        time.sleep(0.05)
        return dict(access_token="renewed", refresh_token="rotated", expires_in=3600)

    monkeypatch.setattr(auth, "_token_request", tokens)
    with ThreadPoolExecutor(4) as executor:
        assert (
            list(
                executor.map(
                    lambda _: auth.access_token(auth.CredentialStore(store.path)), range(4)
                )
            )
            == ["renewed"] * 4
        )
    assert len(calls) == 1
    assert "scope" not in calls[0]
    assert calls[0]["client_id"] == "oaiapp_test"
    assert store.read()["accounts"]["oaiapp_test"]["refresh_token"] == "rotated"


def test_no_plan_scope_fails_closed(store):
    data = store.read()
    data["accounts"]["oaiapp_test"]["scopes"] = ["openid"]
    store.write(data)
    with pytest.raises(auth.ChatGPTAuthError, match="not authorized"):
        auth.access_token(store)


def test_logout_preserves_registration_and_reports_remote_gap(store, monkeypatch):
    monkeypatch.setattr(
        auth.httpx, "get", lambda *a, **kw: (_ for _ in ()).throw(httpx.ConnectError("offline"))
    )
    assert auth.logout(store) is False
    record = store.read()["accounts"]["oaiapp_test"]
    assert record["client_id"] == "oaiapp_test"
    assert record["subject"] == "user-1"
    assert "access_token" not in record
    assert "refresh_token" not in record
    assert "id_token" not in record


def test_status_never_prints_credentials(store):
    result = CliRunner().invoke(auth_cli, ["--auth-file", str(store.path), "status"])
    assert result.exit_code == 0
    assert "plan_authorized=True" in result.output
    for token in ("test-access", "test-refresh", "test-id"):
        assert token not in result.output


def test_request_contract_and_local_tool_history():
    body = plan_request(
        {
            **{key: "unsupported" for key in UNSUPPORTED_FIELDS},
            "input": [
                {"type": "message", "role": "system", "content": "rules"},
                {"type": "function_call", "call_id": "c", "name": "lookup", "arguments": "{}"},
            ],
            "store": True,
            "stream": False,
        }
    )
    assert not (UNSUPPORTED_FIELDS & body.keys())
    assert body["store"] is False and body["stream"] is True
    assert body["input"][0]["role"] == "developer"
    assert body["input"][1]["namespace"] == "halo"


def response(output, status="completed", error=None):
    return dict(
        id="resp_test",
        object="response",
        created_at=0,
        status=status,
        error=error,
        incomplete_details=None,
        instructions=None,
        model="test-model",
        output=output,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        temperature=None,
        top_p=None,
    )


def sse(output=None, *, terminal="completed", error=None, text_delta=None):
    output = output or []
    events = []
    if text_delta:
        events.append(
            dict(
                type="response.output_text.delta",
                item_id="msg",
                output_index=0,
                content_index=0,
                delta=text_delta,
                sequence_number=0,
            )
        )
    for index, item in enumerate(output):
        events.append(
            dict(
                type="response.output_item.done",
                output_index=index,
                item=item,
                sequence_number=len(events),
            )
        )
    if terminal:
        events.append(
            dict(
                type=f"response.{terminal}",
                response=response(output, terminal, error),
                sequence_number=len(events),
            )
        )
    wire = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=wire)


async def test_sdk_tool_execution_and_replay_uses_plan_wire(store, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-paid-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://other.test/v1")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "unrelated-project")
    requests, called = [], []

    @function_tool
    def lookup(value: str) -> str:
        called.append(value)
        return "found"

    def handler(request):
        assert str(request.url) == f"{auth.RESOURCE}/responses"
        assert request.headers["authorization"] == "Bearer test-access"
        assert "openai-project" not in request.headers
        body = json.loads(request.content)
        requests.append(body)
        assert not (UNSUPPORTED_FIELDS & body.keys())
        assert body["store"] is False and body["stream"] is True
        assert body["tools"][0]["type"] == "namespace"
        assert body["tools"][0]["name"] == "halo"
        if len(requests) == 1:
            return sse(
                [
                    dict(
                        id="fc_1",
                        type="function_call",
                        call_id="call_1",
                        name="lookup",
                        namespace="halo",
                        arguments='{"value":"x"}',
                        status="completed",
                    )
                ]
            )
        assert any(
            item.get("type") == "function_call_output" and item["output"] == "found"
            for item in body["input"]
        )
        assert any(
            item.get("type") == "function_call" and item.get("namespace") == "halo"
            for item in body["input"]
        )
        return sse(
            [
                dict(
                    id="msg_1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[dict(type="output_text", text="done", annotations=[])],
                )
            ]
        )

    async with ChatGPTClient(store, transport=httpx.MockTransport(handler)) as client:
        agent = Agent(name="test", model="test-model", tools=plan_tools([lookup]))
        run = Runner.run_streamed(
            agent,
            input=[
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "look up x"},
            ],
            run_config=RunConfig(
                model_provider=ChatGPTProvider(
                    openai_client=client, use_responses=True, use_responses_websocket=False
                ),
                tracing_disabled=True,
            ),
        )
        async for _ in run.stream_events():
            pass
        assert run.final_output == "done"
    assert called == ["x"] and len(requests) == 2
    assert requests[0]["input"][0]["role"] == "developer"


@pytest.mark.parametrize(
    "terminal,error",
    [
        ("failed", {"code": "subscription_sharing_usage_limit_exceeded", "message": "quota"}),
        ("incomplete", None),
        (None, None),
    ],
)
async def test_partial_text_is_not_success(store, terminal, error):
    async with ChatGPTClient(
        store,
        transport=httpx.MockTransport(
            lambda r: sse(terminal=terminal, error=error, text_delta="partial")
        ),
    ) as client:
        with pytest.raises(ChatGPTInferenceError):
            await plan_text(client, ModelConfig(name="test"), "rules", "test")


async def test_compaction_and_synthesis_use_responses(store):
    requests = []

    def handler(request):
        assert request.url.path == "/v1/responses"
        body = json.loads(request.content)
        requests.append(body)
        assert body["instructions"]
        return sse(text_delta="summary")

    async with ChatGPTClient(store, transport=httpx.MockTransport(handler)) as client:
        assert (
            await compact(
                client=client,
                compaction_model=ModelConfig(name="test"),
                item=AgentContextItem(item_id="u", role="user", content="test"),
            )
            == "summary"
        )
        context = MagicMock()
        context.require_trace_store.return_value.render_trace.return_value = "trace content"
        result = await SynthesisTool(ModelConfig(name="test"), client).run(
            context, SynthesizeTracesArguments(trace_ids=["t"])
        )
        assert result.summary == "summary"
    assert len(requests) == 2


async def test_transport_rejects_other_hosts_before_auth(store):
    transport = PlanTransport(store, httpx.MockTransport(lambda r: pytest.fail("must not send")))
    with pytest.raises(ChatGPTInferenceError):
        await transport.handle_async_request(
            httpx.Request("POST", "https://other.test/v1/responses", json={})
        )
    await transport.aclose()


@pytest.mark.parametrize(
    "field,value",
    [
        ("api_key", "key"),
        ("base_url", "https://other.test"),
        ("default_headers", {"Authorization": "other"}),
    ],
)
def test_chatgpt_config_cannot_mix_billing_credentials(field, value):
    with pytest.raises(ValidationError):
        ModelProviderConfig(auth_mode="chatgpt", **{field: value})


def test_refresh_honors_server_earliest_time(store, monkeypatch):
    data = store.read()
    record = data["accounts"]["oaiapp_test"]
    record.update(expires_at=time.time() + 10, earliest_refresh_at=time.time() + 120)
    store.write(data)
    monkeypatch.setattr(auth, "_token_request", lambda form: pytest.fail("too early to refresh"))
    assert auth.access_token(store) == "test-access"
    record["expires_at"] = 0
    store.write(data)
    with pytest.raises(auth.ChatGPTAuthError, match="cannot refresh yet"):
        auth.access_token(store)


def test_invalid_refresh_clears_unusable_tokens_preserves_client(store, monkeypatch):
    data = store.read()
    data["accounts"]["oaiapp_test"]["expires_at"] = 0
    store.write(data)

    def expired(form):
        raise auth.SessionExpiredError("expired")

    monkeypatch.setattr(auth, "_token_request", expired)
    with pytest.raises(auth.SessionExpiredError):
        auth.access_token(store)
    record = store.read()["accounts"]["oaiapp_test"]
    assert record["client_id"] == "oaiapp_test" and record["subject"] == "user-1"
    assert "access_token" not in record and "refresh_token" not in record


async def test_running_client_stays_on_original_account(store):
    observed = []

    def handler(request):
        observed.append(request.headers["authorization"])
        return sse(text_delta="ok")

    async with ChatGPTClient(store, transport=httpx.MockTransport(handler)) as client:
        data = store.read()
        data["accounts"]["oaiapp_other"] = {
            **data["accounts"]["oaiapp_test"],
            "client_id": "oaiapp_other",
            "access_token": "other-account",
        }
        data["active"] = "oaiapp_other"
        store.write(data)
        assert await plan_text(client, ModelConfig(name="test"), "rules", "test") == "ok"
    assert observed == ["Bearer test-access"]


async def test_http_quota_error_is_terminal_and_single_request(store):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            429,
            json={"error": {"code": "subscription_sharing_usage_limit_exceeded"}},
            headers={"x-request-id": "test-request"},
        )

    async with ChatGPTClient(store, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ChatGPTInferenceError, match="test-request"):
            await plan_text(client, ModelConfig(name="test"), "rules", "test")
    assert len(calls) == 1


async def test_full_halo_root_subagent_and_final_answer(store, tmp_path, fixtures_dir, monkeypatch):
    import engine.main as main
    from engine.agents.agent_config import AgentConfig
    from engine.engine_config import EngineConfig
    from engine.models.messages import AgentMessage

    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        assert body["store"] is False and body["stream"] is True
        assert body["tools"][0]["type"] == "namespace"
        if len(bodies) == 1:
            return sse(
                [
                    dict(
                        id="fc_root",
                        type="function_call",
                        call_id="delegate",
                        namespace="halo",
                        name="call_subagent",
                        arguments='{"input":"analyze trace"}',
                        status="completed",
                    )
                ]
            )
        if len(bodies) == 2:
            assert not any(tool["name"] == "final_answer" for tool in body["tools"][0]["tools"])
            return sse(
                [
                    dict(
                        id="msg_child",
                        type="message",
                        role="assistant",
                        status="completed",
                        content=[dict(type="output_text", text="child finding", annotations=[])],
                    )
                ]
            )
        assert any(item.get("type") == "function_call_output" for item in body["input"])
        return sse(
            [
                dict(
                    id="fc_final",
                    type="function_call",
                    call_id="finish",
                    namespace="halo",
                    name="final_answer",
                    arguments='{"answer":"report"}',
                    status="completed",
                )
            ]
        )

    class WiredClient(ChatGPTClient):
        def __init__(self, credential_store):
            super().__init__(credential_store, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(main, "ChatGPTClient", WiredClient)
    model = ModelConfig(name="test-model")
    cfg = EngineConfig(
        root_agent=AgentConfig(name="root", model=model, maximum_turns=3),
        subagent=AgentConfig(name="child", model=model, maximum_turns=3),
        synthesis_model=model,
        compaction_model=model,
        maximum_depth=1,
        model_provider=ModelProviderConfig(auth_mode="chatgpt", auth_file=store.path),
    )
    traces = tmp_path / "trace.jsonl"
    traces.write_bytes((fixtures_dir / "tiny_traces.jsonl").read_bytes())
    output = await main.run_engine_async(
        [AgentMessage(role="user", content="analyze")], cfg, traces
    )
    assert any(item.depth == 1 and item.item.content == "child finding" for item in output)
    assert any(item.final and item.item.content == "report" for item in output)
    assert len(bodies) == 3
