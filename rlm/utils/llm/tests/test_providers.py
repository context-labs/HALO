"""Tests for the LiteLLM and local provider backends."""

import os
from unittest.mock import MagicMock, patch

import openai
import pytest


def test_complete_litellm_extracts_cost_from_header():
    """litellm provider should extract cost from x-litellm-response-cost header."""
    from utils.llm._providers import complete_litellm

    mock_message = MagicMock()
    mock_message.content = "Hello!"
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.completion_tokens_details = None

    mock_parsed = MagicMock()
    mock_parsed.choices = [mock_choice]
    mock_parsed.usage = mock_usage

    mock_raw_response = MagicMock()
    mock_raw_response.headers = {"x-litellm-response-cost": "0.000042"}
    mock_raw_response.parse.return_value = mock_parsed

    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create.return_value = mock_raw_response

    with patch("utils.llm._providers._get_litellm_client", return_value=mock_client):
        result = complete_litellm(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=100,
        )

    assert result["cost"] == pytest.approx(0.000042)
    assert result["content"] == "Hello!"
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 5
    assert result["error"] is None


def test_complete_litellm_raises_on_missing_api_key():
    """Should raise ValueError when LITELLM_API_KEY is not set."""
    import utils.llm._providers as providers_mod
    from utils.llm._providers import complete_litellm

    # Reset cached client so the function re-checks env
    old_client = providers_mod._litellm_client
    providers_mod._litellm_client = None
    try:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="LITELLM_API_KEY"):
                complete_litellm(
                    model="gpt-5.4",
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=0.0,
                    max_tokens=100,
                )
    finally:
        providers_mod._litellm_client = old_client


def test_complete_litellm_with_tools():
    """litellm provider should pass through tool calls."""
    from utils.llm._providers import complete_litellm

    mock_tc = MagicMock()
    mock_tc.id = "call_1"
    mock_tc.function.name = "get_weather"
    mock_tc.function.arguments = '{"city": "SF"}'

    mock_message = MagicMock()
    mock_message.content = None
    mock_message.tool_calls = [mock_tc]

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 20
    mock_usage.completion_tokens = 10
    mock_usage.completion_tokens_details = None

    mock_parsed = MagicMock()
    mock_parsed.choices = [mock_choice]
    mock_parsed.usage = mock_usage

    mock_raw_response = MagicMock()
    mock_raw_response.headers = {"x-litellm-response-cost": "0.001"}
    mock_raw_response.parse.return_value = mock_parsed

    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create.return_value = mock_raw_response

    with patch("utils.llm._providers._get_litellm_client", return_value=mock_client):
        result = complete_litellm(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice="auto",
            temperature=0.0,
            max_tokens=100,
        )

    assert result["tool_calls"] is not None
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# complete_local tests
# ---------------------------------------------------------------------------


def _make_mock_response(content="Hello from local!"):
    """Build a mock chat completion response for local tests."""
    mock_message = MagicMock()
    mock_message.content = content
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 8
    mock_usage.completion_tokens = 4
    mock_usage.completion_tokens_details = None

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    return mock_response


def test_complete_local_success():
    """complete_local should return cost=0.0 and correct content on success."""
    import utils.llm._providers as providers_mod

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response("local result")

    old_clients = providers_mod._local_clients.copy()
    providers_mod._local_clients.clear()
    providers_mod._local_clients["http://localhost:8000/v1"] = mock_client
    try:
        result = providers_mod.complete_local(
            model="local-model",
            messages=[{"role": "user", "content": "hi"}],
            endpoints=["http://localhost:8000/v1"],
        )
    finally:
        providers_mod._local_clients.clear()
        providers_mod._local_clients.update(old_clients)

    assert result["cost"] == 0.0
    assert result["content"] == "local result"
    assert result["error"] is None
    assert result["input_tokens"] == 8
    assert result["output_tokens"] == 4


def test_complete_local_round_robins_endpoints():
    """complete_local should round-robin across provided endpoints."""
    import utils.llm._providers as providers_mod

    mock_client_a = MagicMock()
    mock_client_a.chat.completions.create.return_value = _make_mock_response("A")

    mock_client_b = MagicMock()
    mock_client_b.chat.completions.create.return_value = _make_mock_response("B")

    endpoints = ["http://host-a:8000/v1", "http://host-b:8000/v1"]

    old_clients = providers_mod._local_clients.copy()
    providers_mod._local_clients.clear()
    providers_mod._local_clients["http://host-a:8000/v1"] = mock_client_a
    providers_mod._local_clients["http://host-b:8000/v1"] = mock_client_b
    try:
        r1 = providers_mod.complete_local(
            model="local-model",
            messages=[{"role": "user", "content": "hi"}],
            endpoints=endpoints,
        )
        r2 = providers_mod.complete_local(
            model="local-model",
            messages=[{"role": "user", "content": "hi"}],
            endpoints=endpoints,
        )
    finally:
        providers_mod._local_clients.clear()
        providers_mod._local_clients.update(old_clients)

    # Both clients should have been called exactly once
    assert (
        mock_client_a.chat.completions.create.call_count
        + mock_client_b.chat.completions.create.call_count
        == 2
    )
    # The two results should come from different endpoints
    assert {r1["content"], r2["content"]} == {"A", "B"}


def test_complete_local_retries_on_failure():
    """complete_local should retry on failure and succeed on the next attempt."""
    import utils.llm._providers as providers_mod

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        Exception("temporary failure"),
        _make_mock_response("recovered"),
    ]

    endpoint = "http://localhost:9000/v1"
    old_clients = providers_mod._local_clients.copy()
    providers_mod._local_clients.clear()
    providers_mod._local_clients[endpoint] = mock_client
    try:
        with patch("time.sleep"):  # skip actual sleeps
            result = providers_mod.complete_local(
                model="local-model",
                messages=[{"role": "user", "content": "hi"}],
                endpoints=[endpoint],
                max_retries=3,
            )
    finally:
        providers_mod._local_clients.clear()
        providers_mod._local_clients.update(old_clients)

    assert result["content"] == "recovered"
    assert result["cost"] == 0.0
    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.parametrize("error_cls", [openai.BadRequestError, openai.UnprocessableEntityError])
def test_complete_local_does_not_retry_per_request_errors(error_cls):
    """BadRequestError and UnprocessableEntityError should raise immediately, no retries."""
    import utils.llm._providers as providers_mod

    mock_response = MagicMock()
    mock_response.status_code = 400 if error_cls is openai.BadRequestError else 422
    mock_response.headers = {}
    mock_response.text = "error"
    mock_response.json.return_value = {}

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = error_cls(
        message="context_length_exceeded",
        response=mock_response,
        body=None,
    )

    endpoint = "http://localhost:9000/v1"
    old_clients = providers_mod._local_clients.copy()
    providers_mod._local_clients.clear()
    providers_mod._local_clients[endpoint] = mock_client
    try:
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(error_cls):
                providers_mod.complete_local(
                    model="local-model",
                    messages=[{"role": "user", "content": "hi"}],
                    endpoints=[endpoint],
                    max_retries=3,
                )
        mock_client.chat.completions.create.assert_called_once()
        mock_sleep.assert_not_called()
    finally:
        providers_mod._local_clients.clear()
        providers_mod._local_clients.update(old_clients)
