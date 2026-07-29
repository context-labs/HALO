"""Cached-input tokens reach the exported span attributes.

Cached tokens are billed at a fraction of fresh input, and a multi-turn HALO
run replays a byte-stable prefix the provider serves almost entirely from
cache. Dropping ``input_tokens_details.cached_tokens`` (which the SDK's span
usage dict already carries — ``model_usage_to_span_usage`` serializes it) made
every exported token count overstate what a run actually costs, and left no
way to verify caching is working at all.
"""

from __future__ import annotations

from engine.telemetry.local_processor import _generation_attrs, _response_attrs

RESPONSES_USAGE = {
    "requests": 1,
    "input_tokens": 21_000,
    "output_tokens": 180,
    "total_tokens": 21_180,
    "input_tokens_details": {"cached_tokens": 19_456},
    "output_tokens_details": {"reasoning_tokens": 0},
}

CHAT_USAGE = {
    "prompt_tokens": 3_659,
    "completion_tokens": 13,
    "total_tokens": 3_672,
    "prompt_tokens_details": {"cached_tokens": 3_328},
}


def test_response_span_exports_cached_tokens() -> None:
    attrs, projection = _response_attrs({"response_id": "resp_1", "usage": RESPONSES_USAGE})
    assert attrs["llm.token_count.prompt"] == 21_000
    assert attrs["llm.token_count.prompt_cached"] == 19_456
    assert projection["cached_input_tokens"] == 19_456


def test_generation_span_exports_cached_tokens_from_chat_shape() -> None:
    """Chat Completions nests the count under ``prompt_tokens_details``."""
    attrs, projection = _generation_attrs({"model": "m", "usage": CHAT_USAGE})
    assert attrs["llm.token_count.prompt_cached"] == 3_328
    assert projection["cached_input_tokens"] == 3_328


def test_absent_details_stay_absent_not_zero() -> None:
    """A provider that reports no cache detail must not fabricate a 0.

    ``_drop_none`` removes None-valued attrs, so "unknown" and "zero cache
    hits" remain distinguishable downstream.
    """
    attrs, projection = _response_attrs(
        {"response_id": "resp_2", "usage": {"input_tokens": 10, "output_tokens": 1}}
    )
    assert "llm.token_count.prompt_cached" not in attrs
    assert projection["cached_input_tokens"] is None


REASONING_USAGE = {
    "input_tokens": 1_000,
    "output_tokens": 900,
    "total_tokens": 1_900,
    "output_tokens_details": {"reasoning_tokens": 640},
}


def test_response_span_exports_reasoning_tokens() -> None:
    attrs, projection = _response_attrs({"response_id": "resp_3", "usage": REASONING_USAGE})
    assert attrs["llm.token_count.completion_reasoning"] == 640
    assert projection["reasoning_output_tokens"] == 640


def test_absent_reasoning_details_stay_absent() -> None:
    attrs, projection = _response_attrs(
        {"response_id": "resp_4", "usage": {"input_tokens": 10, "output_tokens": 1}}
    )
    assert "llm.token_count.completion_reasoning" not in attrs
    assert projection["reasoning_output_tokens"] is None


def test_generation_span_exports_reasoning_tokens_from_chat_shape() -> None:
    """Chat Completions nests the count under ``completion_tokens_details`` —
    the compactor and synthesis tool run on that surface."""
    usage = {
        "prompt_tokens": 500,
        "completion_tokens": 300,
        "total_tokens": 800,
        "completion_tokens_details": {"reasoning_tokens": 120},
    }
    attrs, projection = _generation_attrs({"model": "m", "usage": usage})
    assert attrs["llm.token_count.completion_reasoning"] == 120
    assert projection["reasoning_output_tokens"] == 120
