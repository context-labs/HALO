from __future__ import annotations

import asyncio
import json

from engine.traces.care_ai_langfuse import (
    convert_langfuse_export_to_spans,
    load_langfuse_export,
    write_halo_jsonl,
)
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.models.trace_query_models import TraceFilters
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore


def _care_ai_payload() -> dict:
    return {
        "trace": {
            "id": "lf-trace-1",
            "name": "orchestrator",
            "sessionId": "conv-123",
            "timestamp": "2026-05-21T14:01:00.123Z",
            "environment": "test",
            "release": "sha-abc",
            "version": "1.2.3",
            "tags": ["executor:airo-care-orchestrator", "env:test"],
            "metadata": {
                "conversationId": "conv-123",
                "ucid": "ucid-123",
                "sessionExecutorName": "airo-care-orchestrator",
            },
        },
        "observations": [
            {
                "id": "agent-span",
                "traceId": "lf-trace-1",
                "type": "SPAN",
                "name": "agent_execution",
                "startTime": "2026-05-21T14:01:00.123Z",
                "endTime": "2026-05-21T14:01:05.456Z",
                "input": "site down",
                "output": {"responseCount": 1, "toolCallCount": 1, "finalOutputLength": 18},
                "metadata": {
                    "traceKind": "agent_execution",
                    "agentName": "orchestrator",
                    "conversationId": "conv-123",
                    "ucid": "ucid-123",
                    "sessionExecutorName": "airo-care-orchestrator",
                    "conversationModel": "gpt-5.4",
                    "requestType": "send",
                    "requestId": "req-123",
                    "promptName": "c1/airo-care/orchestrator/orchestrator",
                    "promptVersion": 27,
                },
                "usage": {"promptTokens": 12, "completionTokens": 7, "totalTokens": 19},
            },
            {
                "id": "llm-1",
                "traceId": "lf-trace-1",
                "parentObservationId": "agent-span",
                "type": "GENERATION",
                "name": "llm_call_0",
                "model": "gpt-5.4",
                "startTime": "2026-05-21T14:01:01Z",
                "endTime": "2026-05-21T14:01:03Z",
                "input": {"role": "user", "content": "site down"},
                "output": [
                    {"type": "message", "content": "I will check the site."},
                    {"type": "function_call", "name": "http_probe", "call_id": "call-123"},
                ],
                "usage": {"promptTokens": 12, "completionTokens": 7, "totalTokens": 19},
                "usageDetails": {"input": 12, "output": 7, "total": 19},
                "metadata": {
                    "traceKind": "llm_generation",
                    "agentName": "orchestrator",
                    "responseId": "resp-123",
                },
            },
            {
                "id": "tool-1",
                "traceId": "lf-trace-1",
                "parentObservationId": "agent-span",
                "type": "SPAN",
                "name": "tool_call: http_probe",
                "startTime": "2026-05-21T14:01:03Z",
                "endTime": "2026-05-21T14:01:04Z",
                "input": {"url": "https://example.test"},
                "output": {"status": 503},
                "metadata": {
                    "traceKind": "tool_call",
                    "toolName": "http_probe",
                    "toolCallId": "call-123",
                    "status": "completed",
                    "parentAgent": "orchestrator",
                },
            },
        ],
    }


def test_convert_care_ai_langfuse_bundle_to_halo_spans() -> None:
    spans = convert_langfuse_export_to_spans(_care_ai_payload())

    assert [span.attributes["inference.observation_kind"] for span in spans] == [
        "CHAIN",
        "AGENT",
        "LLM",
        "TOOL",
    ]
    assert spans[0].attributes["care_ai.conversation_id"] == "conv-123"
    assert spans[1].attributes["inference.agent_name"] == "orchestrator"
    assert spans[1].attributes["care_ai.agent_reported_tool_call_count"] == 1
    assert spans[1].attributes["care_ai.agent_response_count"] == 1
    assert spans[1].attributes["care_ai.agent_final_output_length"] == 18
    assert spans[2].attributes["llm.model_name"] == "gpt-5.4"
    assert spans[2].attributes["inference.llm.input_tokens"] == 12
    assert spans[2].attributes["llm.response.id"] == "resp-123"
    assert spans[2].attributes["care_ai.llm_planned_tool_names"] == ["http_probe"]
    assert spans[2].attributes["care_ai.llm_planned_tool_count"] == 1
    assert spans[3].attributes["tool.name"] == "http_probe"
    assert spans[3].attributes["output.value"] == '{"status": 503}'
    assert spans[3].parent_span_id == spans[1].span_id


def test_converter_classifies_named_tool_spans_without_tracekind() -> None:
    payload = _care_ai_payload()
    tool = payload["observations"][2]
    tool["metadata"].pop("traceKind")
    tool["metadata"].pop("toolName")
    payload["observations"].append(
        {
            "id": "handoff-1",
            "traceId": "lf-trace-1",
            "parentObservationId": "agent-span",
            "type": "SPAN",
            "name": "transfer_to_DNS_Specialist",
            "startTime": "2026-05-21T14:01:04Z",
            "endTime": "2026-05-21T14:01:05Z",
            "metadata": {},
        }
    )

    spans = convert_langfuse_export_to_spans(payload)
    by_name = {span.name: span for span in spans}

    assert by_name["tool_call: http_probe"].attributes["inference.observation_kind"] == "TOOL"
    assert by_name["tool_call: http_probe"].attributes["tool.name"] == "http_probe"
    assert by_name["transfer_to_DNS_Specialist"].attributes["inference.observation_kind"] == "TOOL"
    assert by_name["transfer_to_DNS_Specialist"].attributes["tool.name"] == (
        "transfer_to_DNS_Specialist"
    )


def test_converter_reads_agent_execution_counts_from_json_string_output() -> None:
    payload = _care_ai_payload()
    payload["observations"][0]["output"] = json.dumps(
        {"responseCount": 2, "toolCallCount": 3, "finalOutputLength": 68}
    )

    spans = convert_langfuse_export_to_spans(payload)

    assert spans[1].attributes["care_ai.agent_response_count"] == 2
    assert spans[1].attributes["care_ai.agent_reported_tool_call_count"] == 3
    assert spans[1].attributes["care_ai.agent_final_output_length"] == 68


def test_converter_accepts_observation_list_exports() -> None:
    payload = _care_ai_payload()
    observations = payload["observations"]

    spans = convert_langfuse_export_to_spans(observations, project_id="care-ai-test")

    assert {span.trace_id for span in spans} == {spans[0].trace_id}
    assert spans[0].attributes["inference.project_id"] == "care-ai-test"
    assert {span.attributes["langfuse.trace_id"] for span in spans} == {"lf-trace-1"}


def test_converter_accepts_lf_body_wrappers() -> None:
    payload = _care_ai_payload()
    wrapped = {
        "trace": {"body": payload["trace"]},
        "observations": {"body": {"data": payload["observations"]}},
    }

    spans = convert_langfuse_export_to_spans(wrapped)

    assert len(spans) == 4
    assert spans[2].attributes["llm.model_name"] == "gpt-5.4"
    assert spans[3].attributes["tool.name"] == "http_probe"


def test_converter_uses_care_ai_conversation_model_when_observation_model_is_blank() -> None:
    payload = _care_ai_payload()
    payload["observations"][1]["model"] = ""
    payload["observations"][1]["metadata"].pop("model", None)
    payload["observations"][1]["metadata"]["conversationModel"] = "gpt-5.4"

    spans = convert_langfuse_export_to_spans(payload)

    assert spans[2].attributes["llm.model_name"] == "gpt-5.4"
    assert spans[2].attributes["inference.llm.model_name"] == "gpt-5.4"


def test_converter_extracts_planned_tool_names_from_json_string_output() -> None:
    payload = _care_ai_payload()
    payload["observations"][1]["output"] = json.dumps(
        [
            {"type": "message", "content": "Checking."},
            {"type": "function_call", "name": "domain_and_dns_tool"},
        ]
    )

    spans = convert_langfuse_export_to_spans(payload)

    assert spans[2].attributes["care_ai.llm_planned_tool_names"] == ["domain_and_dns_tool"]


def test_converted_spans_are_indexable_by_halo(tmp_path) -> None:
    trace_path = tmp_path / "care-ai-traces.jsonl"
    spans = convert_langfuse_export_to_spans(_care_ai_payload())
    write_halo_jsonl(spans, trace_path)

    index_path = asyncio.run(TraceIndexBuilder.ensure_index_exists(trace_path, TraceIndexConfig()))
    store = TraceStore.load(trace_path, index_path)
    overview = store.get_overview(TraceFilters())

    assert overview.total_traces == 1
    assert overview.total_spans == 4
    assert overview.service_names == ["care-ai-agents"]
    assert overview.model_names == ["gpt-5.4"]
    assert overview.agent_names == ["orchestrator"]
    assert overview.total_input_tokens == 12
    assert overview.total_output_tokens == 7


def test_load_json_and_jsonl_exports(tmp_path) -> None:
    json_path = tmp_path / "export.json"
    jsonl_path = tmp_path / "export.jsonl"
    payload = _care_ai_payload()
    json_path.write_text(json.dumps(payload))
    jsonl_path.write_text("\n".join(json.dumps(obs) for obs in payload["observations"]))

    assert load_langfuse_export(json_path)["trace"]["id"] == "lf-trace-1"
    assert len(load_langfuse_export(jsonl_path)) == 3
