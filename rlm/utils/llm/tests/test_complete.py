"""Tests for the complete() entry point."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_provider_result(**overrides):
    base = {
        "content": "hi",
        "tool_calls": None,
        "message": {"role": "assistant", "content": "hi"},
        "input_tokens": 5,
        "output_tokens": 2,
        "thinking_tokens": 0,
        "cost": 0.001,
        "error": None,
    }
    base.update(overrides)
    return base


def test_complete_routes_to_litellm_by_default():
    from utils.llm._complete import complete

    with patch(
        "utils.llm._complete._providers.complete_litellm", return_value=_mock_provider_result()
    ) as mock_fn:
        result = complete("gpt-5.4", [{"role": "user", "content": "hi"}])

    mock_fn.assert_called_once()
    assert result.content == "hi"
    assert result.cost == 0.001
    assert result.error is None


def test_complete_routes_to_local_when_flag_set():
    from utils.llm._complete import complete

    with patch(
        "utils.llm._complete._providers.complete_local",
        return_value=_mock_provider_result(cost=0.0),
    ) as mock_fn:
        result = complete(
            "my-model",
            [{"role": "user", "content": "hi"}],
            local=True,
            endpoints=["http://10.0.1.28:8011/v1"],
        )

    mock_fn.assert_called_once()
    assert result.content == "hi"
    assert result.cost == 0.0


def test_complete_raises_when_local_without_endpoints():
    from utils.llm._complete import complete

    with pytest.raises(ValueError, match="endpoints"):
        complete("my-model", [{"role": "user", "content": "hi"}], local=True)


def test_complete_captures_per_request_error():
    import openai
    from utils.llm._complete import complete

    err = openai.BadRequestError(
        message="content_filter",
        response=MagicMock(status_code=400, headers={}, text="filtered", json=lambda: {}),
        body=None,
    )

    with patch("utils.llm._complete._providers.complete_litellm", side_effect=err):
        result = complete("gpt-5.4", [{"role": "user", "content": "bad"}])

    assert result.error is not None
    assert "content_filter" in result.error


def test_complete_captures_unprocessable_entity_error():
    """422 from LiteLLM proxy (e.g. context length exceeded) should be captured, not raised."""
    import openai
    from utils.llm._complete import complete

    err = openai.UnprocessableEntityError(
        message="context_length_exceeded",
        response=MagicMock(status_code=422, headers={}, text="too long", json=lambda: {}),
        body=None,
    )

    with patch("utils.llm._complete._providers.complete_litellm", side_effect=err):
        result = complete("gpt-5.4", [{"role": "user", "content": "x" * 100000}])

    assert result.error is not None
    assert "context_length_exceeded" in result.error
    assert result.content is None


def test_complete_raises_on_auth_error():
    import openai
    from utils.llm._complete import complete

    err = openai.AuthenticationError(
        message="invalid api key",
        response=MagicMock(status_code=401, headers={}, text="unauth", json=lambda: {}),
        body=None,
    )

    with patch("utils.llm._complete._providers.complete_litellm", side_effect=err):
        with pytest.raises(openai.AuthenticationError):
            complete("gpt-5.4", [{"role": "user", "content": "hi"}])


def test_complete_raises_on_not_found_error():
    import openai
    from utils.llm._complete import complete

    err = openai.NotFoundError(
        message="model not found",
        response=MagicMock(status_code=404, headers={}, text="not found", json=lambda: {}),
        body=None,
    )

    with patch("utils.llm._complete._providers.complete_litellm", side_effect=err):
        with pytest.raises(openai.NotFoundError):
            complete("nonexistent-model", [{"role": "user", "content": "hi"}])


def test_complete_raises_on_permission_denied_error():
    import openai
    from utils.llm._complete import complete

    err = openai.PermissionDeniedError(
        message="permission denied",
        response=MagicMock(status_code=403, headers={}, text="forbidden", json=lambda: {}),
        body=None,
    )

    with patch("utils.llm._complete._providers.complete_litellm", side_effect=err):
        with pytest.raises(openai.PermissionDeniedError):
            complete("gpt-5.4", [{"role": "user", "content": "hi"}])


def test_complete_calls_on_complete_callback():
    from utils.llm._complete import complete

    callback = MagicMock()

    with patch(
        "utils.llm._complete._providers.complete_litellm", return_value=_mock_provider_result()
    ):
        result = complete("gpt-5.4", [{"role": "user", "content": "hi"}], on_complete=callback)

    callback.assert_called_once_with(result)
