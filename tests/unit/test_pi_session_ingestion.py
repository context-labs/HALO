from __future__ import annotations

import asyncio
import json
from pathlib import Path

from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.models.trace_query_models import TraceFilters
from engine.traces.source_adapters import detect_trace_source, prepare_trace_input
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _minimal_session_rows(session_id: str = "session-1") -> list[dict]:
    return [
        {
            "type": "session",
            "version": 3,
            "id": session_id,
            "timestamp": "2026-04-30T12:00:00.000Z",
            "cwd": "/repo",
        },
        {
            "type": "message",
            "id": "u0000001",
            "parentId": None,
            "timestamp": "2026-04-30T12:00:01.000Z",
            "message": {
                "role": "user",
                "content": "please inspect this very long private prompt payload",
                "timestamp": 1777540801000,
            },
        },
        {
            "type": "message",
            "id": "a0000002",
            "parentId": "u0000001",
            "timestamp": "2026-04-30T12:00:02.000Z",
            "message": {
                "role": "assistant",
                "api": "responses",
                "provider": "openai",
                "model": "gpt-test",
                "usage": {"input": 11, "output": 7, "totalTokens": 18},
                "stopReason": "toolUse",
                "content": [
                    {"type": "thinking", "thinking": "hidden chain of thought should be bounded"},
                    {"type": "text", "text": "I will call a tool with bounded metadata."},
                    {
                        "type": "toolCall",
                        "id": "call_1",
                        "name": "bash",
                        "arguments": {
                            "command": "echo super-secret-value && sleep 1",
                            "timeout": 20,
                        },
                    },
                ],
                "timestamp": 1777540802000,
            },
        },
        {
            "type": "message",
            "id": "t0000003",
            "parentId": "a0000002",
            "timestamp": "2026-04-30T12:00:03.000Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "call_1",
                "toolName": "bash",
                "content": [
                    {"type": "text", "text": "large failing tool output with private details"}
                ],
                "isError": True,
                "timestamp": 1777540803000,
            },
        },
        {
            "type": "compaction",
            "id": "c0000004",
            "parentId": "t0000003",
            "timestamp": "2026-04-30T12:00:04.000Z",
            "summary": "compact old context with a bounded summary",
            "firstKeptEntryId": "a0000002",
            "tokensBefore": 50000,
        },
        {
            "type": "custom_message",
            "id": "m0000005",
            "parentId": "c0000004",
            "timestamp": "2026-04-30T12:00:05.000Z",
            "customType": "test-extension",
            "content": "extension-injected context should also be bounded",
            "display": True,
        },
        {
            "type": "branch_summary",
            "id": "b0000006",
            "parentId": "u0000001",
            "timestamp": "2026-04-30T12:00:06.000Z",
            "fromId": "c0000004",
            "summary": "abandoned branch summary",
        },
    ]


async def _load_prepared_store(source: Path, *, excerpt_chars: int = 12) -> TraceStore:
    prepared = prepare_trace_input(
        source,
        TraceIndexConfig(),
        source="pi-session",
        pi_session_excerpt_chars=excerpt_chars,
    )
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path=prepared.trace_path,
        config=prepared.config,
    )
    return TraceStore.load(trace_path=prepared.trace_path, index_path=index_path)


def test_detect_trace_source_accepts_pi_session_jsonl(tmp_path: Path) -> None:
    session_path = tmp_path / "session.jsonl"
    _write_jsonl(session_path, _minimal_session_rows())

    assert detect_trace_source(session_path) == "pi-session"


def test_pi_session_file_indexes_as_queryable_canonical_spans(tmp_path: Path) -> None:
    session_path = tmp_path / "session.jsonl"
    _write_jsonl(session_path, _minimal_session_rows())

    store = asyncio.run(_load_prepared_store(session_path))

    overview = store.get_overview(TraceFilters())
    assert overview.total_traces == 1
    assert overview.total_spans == 7
    assert overview.model_names == ["gpt-test"]
    assert overview.error_trace_count == 1
    assert overview.total_input_tokens == 11
    assert overview.total_output_tokens == 7

    trace = store.view_trace("session-1")
    assert [span.name for span in trace.spans] == [
        "session",
        "message.user",
        "message.assistant",
        "message.toolResult",
        "compaction",
        "custom_message",
        "branch_summary",
    ]
    assert trace.spans[3].status.code == "STATUS_CODE_ERROR"
    assert trace.spans[3].attributes["pi.message.isError"] is True


def test_pi_session_redaction_bounds_text_thinking_tool_args_and_results(tmp_path: Path) -> None:
    session_path = tmp_path / "session.jsonl"
    _write_jsonl(session_path, _minimal_session_rows())

    store = asyncio.run(_load_prepared_store(session_path, excerpt_chars=10))
    spans = {span.span_id: span for span in store.view_trace("session-1").spans}

    user_attrs = spans["u0000001"].attributes
    assert "pi.message.content.text" not in user_attrs
    assert user_attrs["pi.message.content.text.char_count"] > 10
    assert user_attrs["pi.message.content.text.excerpt"].startswith("please ins")
    assert "original" in user_attrs["pi.message.content.text.excerpt"]

    assistant_attrs = spans["a0000002"].attributes
    assert "pi.content.thinking" not in assistant_attrs
    assert assistant_attrs["pi.content.thinking.char_count"] > 10
    assert assistant_attrs["pi.content.thinking.excerpt"].startswith("hidden cha")
    assert "pi.tool_call.arguments" not in assistant_attrs
    assert assistant_attrs["pi.tool_call.argument_keys"] == {"bash": ["command", "timeout"]}
    assert assistant_attrs["pi.tool_call.arguments.json_char_count"] > 10
    assert "original" in assistant_attrs["pi.tool_call.arguments.excerpt"]

    tool_attrs = spans["t0000003"].attributes
    assert "pi.content.text" not in tool_attrs
    assert tool_attrs["pi.content.text.char_count"] > 10
    assert tool_attrs["pi.content.text.excerpt"].startswith("large fail")

    custom_attrs = spans["m0000005"].attributes
    assert "pi.custom_message.content.text" not in custom_attrs
    assert custom_attrs["pi.custom_message.customType"] == "test-extension"
    assert custom_attrs["pi.custom_message.display"] is True
    assert custom_attrs["pi.custom_message.content.text.char_count"] > 10
    assert custom_attrs["pi.custom_message.content.text.excerpt"].startswith("extension-")


def test_pi_session_directory_ingests_multiple_files(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    _write_jsonl(sessions_dir / "one.jsonl", _minimal_session_rows("session-one"))
    _write_jsonl(sessions_dir / "two.jsonl", _minimal_session_rows("session-two"))

    store = asyncio.run(_load_prepared_store(sessions_dir))

    overview = store.get_overview(TraceFilters())
    assert overview.total_traces == 2
    assert sorted(overview.sample_trace_ids) == ["session-one", "session-two"]
    assert store.count_traces(TraceFilters()).total == 2
