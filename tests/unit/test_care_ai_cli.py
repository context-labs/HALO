from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from engine.traces.care_ai_langfuse import convert_langfuse_export_to_spans, write_halo_jsonl
from halo_cli.care_ai import (
    _GOCODE_OPENAI_BASE_URL,
    DiagnosticFocus,
    ExperimentPlanFormat,
    TraceToolMode,
    app,
    audit_halo_jsonl,
    build_candidate_decision,
    build_candidate_evaluation,
    build_candidate_handoff,
    build_candidate_local_pipeline,
    build_candidate_loop_status,
    build_candidate_pipeline_paths,
    build_candidate_preflight,
    build_candidate_prompt_artifact,
    build_candidate_review,
    build_candidate_runtime_check,
    build_candidate_runtime_plan,
    build_diagnosis_prompt_with_evidence,
    build_diagnostic_prompt,
    build_evidence_pack,
    build_experiment_plan,
    build_harness_context,
    build_lf_batch_recipe,
    build_pipeline_paths,
    build_prompt_evidence_summary,
    build_prompt_snapshot,
    care_ai_surface_for_executor,
    compare_audits,
    doctor_report,
    inspect_halo_jsonl,
    render_candidate_decision_markdown,
    render_candidate_handoff_markdown,
    render_candidate_loop_status_markdown,
    render_candidate_preflight_markdown,
    render_candidate_review_markdown,
    render_candidate_runtime_check_markdown,
    render_candidate_runtime_plan_markdown,
    render_experiment_plan_markdown,
    sanitize_halo_jsonl,
    trace_safety_report,
)


def _write_prompt_fixture(tmp_path: Path) -> Path:
    prompt_path = tmp_path / "prompt.json"
    prompt_path.write_text(
        json.dumps(
            {
                "ok": True,
                "status": 200,
                "body": {
                    "name": "c1/airo-care/orchestrator-test/orchestrator",
                    "type": "text",
                    "version": 42,
                    "labels": ["latest"],
                    "tags": ["care-ai"],
                    "commitMessage": "current test prompt",
                    "createdAt": "2026-05-21T12:00:00.000Z",
                    "updatedAt": "2026-05-21T12:05:00.000Z",
                    "prompt": "SYSTEM: secret routing prompt\nDo the thing.",
                    "config": {"model": "gpt-5.4"},
                },
            }
        )
    )
    return prompt_path


def _write_created_prompt_fixture(
    tmp_path: Path,
    *,
    candidate_body: str,
    version: int = 43,
    labels: list[str] | None = None,
) -> Path:
    created_path = tmp_path / "candidate.created.json"
    created_path.write_text(
        json.dumps(
            {
                "ok": True,
                "status": 200,
                "body": {
                    "name": "c1/airo-care/orchestrator-test/orchestrator",
                    "type": "text",
                    "version": version,
                    "labels": labels if labels is not None else ["latest", "halo-candidate"],
                    "tags": ["halo", "care-ai", "cost"],
                    "prompt": candidate_body,
                    "config": {"model": "gpt-5.4"},
                },
            }
        )
    )
    return created_path


def _write_runtime_check_fixture(
    tmp_path: Path,
    *,
    version: int = 43,
    created_at: str = "2026-05-22T04:41:00.460Z",
    passed: bool = True,
) -> Path:
    runtime_check_path = tmp_path / "runtime-check.json"
    runtime_check_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "care-ai-halo-candidate-runtime-check",
                "runtime_check_passed": passed,
                "prompt": {
                    "name": "c1/airo-care/orchestrator-test/orchestrator",
                    "version": version,
                    "labels": ["latest", "halo-candidate"],
                    "created_at": created_at,
                },
            }
        )
    )
    return runtime_check_path


def _write_trace_fixture(tmp_path: Path) -> Path:
    trace_path = tmp_path / "care-ai-traces.jsonl"
    write_halo_jsonl(convert_langfuse_export_to_spans(_langfuse_bundle()), trace_path)
    return trace_path


def _langfuse_bundle() -> dict:
    return {
        "trace": {
            "id": "trace-1",
            "name": "orchestrator-test",
            "sessionId": "conv-123",
            "timestamp": "2026-05-21T14:01:00.123Z",
            "metadata": {
                "conversationId": "conv-123",
                "ucid": "ucid-123",
                "sessionExecutorName": "orchestrator-test",
            },
        },
        "observations": [
            {
                "id": "agent-1",
                "traceId": "trace-1",
                "type": "SPAN",
                "name": "agent_execution",
                "startTime": "2026-05-21T14:01:00.123Z",
                "endTime": "2026-05-21T14:01:04.123Z",
                "metadata": {
                    "traceKind": "agent_execution",
                    "agentName": "orchestrator-test",
                    "conversationId": "conv-123",
                    "ucid": "ucid-123",
                    "sessionExecutorName": "orchestrator-test",
                    "conversationModel": "gpt-5.4",
                },
            },
            {
                "id": "llm-1",
                "traceId": "trace-1",
                "parentObservationId": "agent-1",
                "type": "GENERATION",
                "name": "llm_call_0",
                "model": "gpt-5.4",
                "startTime": "2026-05-21T14:01:01Z",
                "endTime": "2026-05-21T14:01:02Z",
                "input": "site down",
                "output": [
                    {"type": "message", "content": "checking"},
                    {"type": "function_call", "name": "http_probe", "call_id": "call-1"},
                ],
                "usageDetails": {"input": 11, "output": 3, "total": 14},
                "metadata": {
                    "traceKind": "llm_generation",
                    "agentName": "orchestrator-test",
                },
            },
            {
                "id": "tool-1",
                "traceId": "trace-1",
                "parentObservationId": "agent-1",
                "type": "SPAN",
                "name": "tool_call: http_probe",
                "startTime": "2026-05-21T14:01:02Z",
                "endTime": "2026-05-21T14:01:03Z",
                "input": {"url": "https://example.test"},
                "metadata": {
                    "traceKind": "tool_call",
                    "toolName": "http_probe",
                    "toolCallId": "call-1",
                    "status": "completed",
                    "parentAgent": "orchestrator-test",
                },
            },
        ],
    }


def _write_raw_lf_pair(tmp_path: Path) -> tuple[Path, Path]:
    bundle = _langfuse_bundle()
    trace_path = tmp_path / "trace.json"
    observations_path = tmp_path / "observations.json"
    trace_path.write_text(json.dumps({"ok": True, "status": 200, "body": bundle["trace"]}))
    observations_path.write_text(
        json.dumps({"ok": True, "status": 200, "body": {"data": bundle["observations"]}})
    )
    return trace_path, observations_path


def test_build_diagnostic_prompt_stays_trace_scoped() -> None:
    prompt = build_diagnostic_prompt(
        focus=DiagnosticFocus.routing,
        executor="airo-care-orchestrator",
        session_id="conv-123",
        extra_question="Did the agent transfer too early?",
    )

    assert "Stay inside the trace evidence" in prompt
    assert "care_ai.session_executor_name equals 'airo-care-orchestrator'" in prompt
    assert "care_ai.conversation_id equal to 'conv-123'" in prompt
    assert "GENERATION.input often contains only the latest user message" in prompt
    assert "care_ai.llm_planned_tool_names" in prompt
    assert (
        "do not describe token counts, hashes, ids, or byte counts as literal payload snippets"
        in prompt
    )
    assert "Expected answer shape" in prompt
    assert (
        "Include literal payload snippets only when raw payload text is actually present" in prompt
    )
    assert "Prompt Management API backed by Langfuse" in prompt
    assert "production `airo-care-orchestrator` manager prompt" in prompt
    assert "Did the agent transfer too early?" in prompt
    assert "misrouting" in prompt


def test_build_harness_context_includes_variant_tools_when_relevant() -> None:
    prod_context = build_harness_context(executor="airo-care-orchestrator")
    variant_context = build_harness_context(executor="orchestrator-test")
    sandbox_context = build_harness_context(executor="orchestrator-sandbox")

    assert "`domain_and_dns_tool`" in prod_context
    assert "`domain_lifecycle_tool`" in prod_context
    assert "case creation is disabled" in prod_context
    assert "`billing_tool`" not in prod_context
    assert "`billing_tool`" in variant_context
    assert "c1/airo-care/orchestrator-test/orchestrator" in variant_context
    assert "orchestrator-test/index.ts" in variant_context
    assert "Variant prompt names" in variant_context
    assert "`refund_agent`" not in variant_context
    assert "`refund_agent`" in sandbox_context


def test_care_ai_surface_for_executor_maps_known_variants() -> None:
    surface = care_ai_surface_for_executor("orchestrator-test")

    assert surface["prompt_name"] == "c1/airo-care/orchestrator-test/orchestrator"
    assert surface["route"] == "/v1/orchestrator-test"
    assert surface["session_executor_path"].endswith("orchestrator-test/index.ts")
    assert care_ai_surface_for_executor("unknown-executor") == {}


def test_build_diagnostic_prompt_has_focus_specific_tool_error_request() -> None:
    prompt = build_diagnostic_prompt(focus=DiagnosticFocus.tool_errors)

    assert "Analyze TOOL spans" in prompt
    assert "literal error strings" in prompt
    assert "If the trace was sanitized" in prompt
    assert "redaction fingerprints" in prompt
    assert "care_ai.tool_call_id" in prompt


def test_care_ai_cli_exposes_convert_and_diagnose_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "convert-langfuse" in result.output
    assert "convert-lf-pair" in result.output
    assert "lf-batch-recipe" in result.output
    assert "doctor" in result.output
    assert "sanitize" in result.output
    assert "inspect" in result.output
    assert "audit" in result.output
    assert "compare-audits" in result.output
    assert "evidence-pack" in result.output
    assert "experiment-plan" in result.output
    assert "candidate-handoff" in result.output
    assert "candidate-preflight" in result.output
    assert "candidate-runtime-check" in result.output
    assert "candidate-evaluate" in result.output
    assert "loop-status" in result.output
    assert "local-pipeline" in result.output
    assert "diagnose" in result.output


def test_diagnose_help_lists_focus_option() -> None:
    result = CliRunner().invoke(app, ["diagnose", "--help"])

    assert result.exit_code == 0
    assert "--focus" in result.output
    assert "--tool-mode" in result.output
    assert "not supported" in result.output
    assert "--output" in result.output
    assert "--events-jsonl" in result.output
    assert "metadata-only" in result.output
    assert "--evidence-top" in result.output
    assert "--print-prompt" in result.output
    assert "Care AI" in result.output


def test_diagnose_print_prompt_exits_without_running_halo(tmp_path: Path) -> None:
    trace_path = _write_trace_fixture(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "diagnose",
            str(trace_path),
            "--focus",
            "routing",
            "--executor",
            "orchestrator-test",
            "--print-prompt",
        ],
    )

    assert result.exit_code == 0
    assert "Deterministic local evidence summary" in result.output
    assert "Expected answer shape" in result.output
    assert "`billing_tool`" in result.output
    assert "misrouting" in result.output
    assert "site down" not in result.output
    assert "https://example.test" not in result.output


def test_diagnose_authentication_error_prints_short_remediation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = _write_trace_fixture(tmp_path)

    def raise_auth_error(**_kwargs):
        raise RuntimeError("invalid_api_key")

    monkeypatch.setattr("halo_cli.care_ai.run_trace", raise_auth_error)

    result = CliRunner().invoke(app, ["diagnose", str(trace_path)])

    assert result.exit_code == 1
    assert "could not authenticate" in result.stderr
    assert "doctor --check-model" in result.stderr
    assert "Traceback" not in result.stderr


def test_doctor_report_checks_env_trace_and_model_without_printing_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = _write_trace_fixture(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setattr("halo_cli.care_ai.Sandbox.get", lambda: object())
    monkeypatch.setattr("halo_cli.care_ai._check_model_access", lambda model: "ok")

    report = doctor_report(trace_path, model="gpt-test", check_model=True)

    assert report["model_provider"] == "environment"
    assert report["openai_api_key"] == "set"
    assert report["sandbox"] == "available"
    assert report["model"] == "gpt-test"
    assert report["model_access"] == "ok"
    assert report["trace"]["total_traces"] == 1
    assert report["trace"]["audit_counts"]["tool_errors"] == 0
    assert "sk-secret" not in json.dumps(report)


def test_doctor_report_can_route_model_check_through_gocode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trace_path = _write_trace_fixture(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("GOCODE_CODEX_API_KEY", "sk-gocode-secret")
    monkeypatch.setattr("halo_cli.care_ai.Sandbox.get", lambda: object())

    observed: dict[str, object] = {}

    def check_model(_model: str) -> str:
        observed["openai_api_key"] = os.environ.get("OPENAI_API_KEY")
        observed["openai_base_url"] = os.environ.get("OPENAI_BASE_URL")
        return "ok"

    monkeypatch.setattr("halo_cli.care_ai._check_model_access", check_model)

    report = doctor_report(trace_path, model="gpt-test", check_model=True, use_gocode=True)

    assert report["model_provider"] == "gocode"
    assert report["openai_api_key"] == "set"
    assert report["gocode_codex_api_key"] == "set"
    assert report["openai_base_url"] == "gocode"
    assert report["model_access"] == "ok"
    assert observed == {
        "openai_api_key": "sk-gocode-secret",
        "openai_base_url": _GOCODE_OPENAI_BASE_URL,
    }
    assert "sk-gocode-secret" not in json.dumps(report)
    assert os.environ.get("OPENAI_API_KEY") is None
    assert os.environ.get("OPENAI_BASE_URL") is None


def test_doctor_cli_json_reports_missing_key_without_traceback(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("halo_cli.care_ai.Sandbox.get", lambda: None)

    result = CliRunner().invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["openai_api_key"] == "missing"
    assert report["sandbox"] == "unavailable"


def test_diagnose_gocode_sets_provider_env_for_run_trace(tmp_path: Path, monkeypatch) -> None:
    trace_path = _write_trace_fixture(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("GOCODE_CODEX_API_KEY", "sk-gocode-secret")
    observed: dict[str, object] = {}

    def capture_run_trace(**_kwargs):
        observed["openai_api_key"] = os.environ.get("OPENAI_API_KEY")
        observed["openai_base_url"] = os.environ.get("OPENAI_BASE_URL")
        observed["trace_detail_tools_enabled"] = _kwargs["trace_detail_tools_enabled"]
        observed["run_code_enabled"] = _kwargs["run_code_enabled"]
        observed["timeout_seconds"] = _kwargs["timeout_seconds"]
        observed["output_path"] = _kwargs["output_path"]
        observed["events_path"] = _kwargs["events_path"]

    monkeypatch.setattr("halo_cli.care_ai.run_trace", capture_run_trace)

    result = CliRunner().invoke(app, ["diagnose", str(trace_path), "--gocode"])

    assert result.exit_code == 0
    assert observed == {
        "openai_api_key": "sk-gocode-secret",
        "openai_base_url": _GOCODE_OPENAI_BASE_URL,
        "trace_detail_tools_enabled": False,
        "run_code_enabled": False,
        "timeout_seconds": 180,
        "output_path": None,
        "events_path": None,
    }
    assert os.environ.get("OPENAI_API_KEY") is None
    assert os.environ.get("OPENAI_BASE_URL") is None


def test_diagnose_full_tool_mode_enables_detail_tools(tmp_path: Path, monkeypatch) -> None:
    trace_path = _write_trace_fixture(tmp_path)
    observed: dict[str, object] = {}

    def capture_run_trace(**kwargs):
        observed["trace_detail_tools_enabled"] = kwargs["trace_detail_tools_enabled"]
        observed["run_code_enabled"] = kwargs["run_code_enabled"]
        observed["timeout_seconds"] = kwargs["timeout_seconds"]
        observed["output_path"] = kwargs["output_path"]
        observed["events_path"] = kwargs["events_path"]

    monkeypatch.setattr("halo_cli.care_ai.run_trace", capture_run_trace)

    result = CliRunner().invoke(
        app,
        ["diagnose", str(trace_path), "--tool-mode", TraceToolMode.full.value],
    )

    assert result.exit_code == 0
    assert observed == {
        "trace_detail_tools_enabled": True,
        "run_code_enabled": True,
        "timeout_seconds": 180,
        "output_path": None,
        "events_path": None,
    }


def test_diagnose_passes_report_artifact_paths_to_run_trace(tmp_path: Path, monkeypatch) -> None:
    trace_path = _write_trace_fixture(tmp_path)
    output_path = tmp_path / "diagnosis.md"
    events_path = tmp_path / "events.jsonl"
    observed: dict[str, object] = {}

    def capture_run_trace(**kwargs):
        observed["output_path"] = kwargs["output_path"]
        observed["events_path"] = kwargs["events_path"]

    monkeypatch.setattr("halo_cli.care_ai.run_trace", capture_run_trace)

    result = CliRunner().invoke(
        app,
        [
            "diagnose",
            str(trace_path),
            "--output",
            str(output_path),
            "--events-jsonl",
            str(events_path),
        ],
    )

    assert result.exit_code == 0
    assert observed == {"output_path": output_path, "events_path": events_path}


def test_inspect_halo_jsonl_returns_metadata_summary(tmp_path: Path) -> None:
    summary = inspect_halo_jsonl(_write_trace_fixture(tmp_path))

    assert summary["total_traces"] == 1
    assert summary["total_spans"] == 4
    assert summary["session_count"] == 1
    assert summary["ucid_count"] == 1
    assert summary["total_input_tokens"] == 11
    assert summary["total_output_tokens"] == 3
    assert summary["session_trace_distribution"] == {
        "count": 1,
        "total": 1,
        "min": 1,
        "median": 1,
        "p95": 1,
        "p99": 1,
        "max": 1,
        "mean": 1.0,
    }
    assert summary["session_input_token_distribution"] == {
        "count": 1,
        "total": 11,
        "min": 11,
        "median": 11,
        "p95": 11,
        "p99": 11,
        "max": 11,
        "mean": 11.0,
    }
    assert summary["top_sessions_by_input_tokens"][0]["trace_count"] == 1
    assert summary["top_sessions_by_input_tokens"][0]["input_tokens"] == 11
    assert "session_sha256_12" in summary["top_sessions_by_input_tokens"][0]
    assert summary["llm_input_token_distribution"] == {
        "count": 1,
        "total": 11,
        "min": 11,
        "median": 11,
        "p95": 11,
        "p99": 11,
        "max": 11,
        "mean": 11.0,
    }
    assert summary["llm_output_token_distribution"]["median"] == 3
    assert summary["model_names"] == [{"name": "gpt-5.4", "count": 1}]
    assert summary["agent_names"] == [{"name": "orchestrator-test", "count": 2}]
    assert summary["tool_names"] == [{"name": "http_probe", "count": 1}]
    assert summary["planned_tool_names"] == [{"name": "http_probe", "count": 1}]


def test_inspect_halo_jsonl_reports_llm_token_distribution(tmp_path: Path) -> None:
    trace_path = _write_trace_fixture(tmp_path)
    second = _langfuse_bundle()
    second["trace"]["id"] = "trace-2"
    second["observations"][0]["traceId"] = "trace-2"
    second["observations"][1]["traceId"] = "trace-2"
    second["observations"][2]["traceId"] = "trace-2"
    second["observations"][1]["usageDetails"] = {
        "input": 40_001,
        "output": 7,
        "total": 40_008,
    }
    write_halo_jsonl(convert_langfuse_export_to_spans(second), trace_path, append=True)

    summary = inspect_halo_jsonl(trace_path)

    assert summary["total_input_tokens"] == 40_012
    assert summary["llm_input_token_distribution"] == {
        "count": 2,
        "total": 40_012,
        "min": 11,
        "median": 20_006.0,
        "p95": 40_001,
        "p99": 40_001,
        "max": 40_001,
        "mean": 20_006.0,
    }
    assert summary["session_trace_distribution"]["median"] == 2
    assert summary["session_input_token_distribution"]["total"] == 40_012
    assert summary["top_sessions_by_input_tokens"][0]["trace_count"] == 2
    assert summary["top_sessions_by_input_tokens"][0]["span_count"] == 8
    assert summary["llm_output_token_distribution"]["median"] == 5.0


def test_inspect_cli_json_does_not_print_raw_io(tmp_path: Path) -> None:
    trace_path = _write_trace_fixture(tmp_path)
    result = CliRunner().invoke(app, ["inspect", str(trace_path), "--json"])

    assert result.exit_code == 0
    summary = json.loads(result.output)
    assert summary["total_traces"] == 1
    assert summary["tool_names"] == [{"name": "http_probe", "count": 1}]
    assert summary["planned_tool_names"] == [{"name": "http_probe", "count": 1}]
    assert summary["llm_input_token_distribution"]["p95"] == 11
    assert summary["session_input_token_distribution"]["p95"] == 11
    assert "site down" not in result.output
    assert "checking" not in result.output


def test_sanitize_halo_jsonl_redacts_raw_payloads_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    input_path = _write_trace_fixture(tmp_path)
    output_path = tmp_path / "sanitized.jsonl"

    report = sanitize_halo_jsonl(input_path, output_path)

    text = output_path.read_text()
    summary = inspect_halo_jsonl(output_path)
    assert report["spans_written"] == 4
    assert report["redacted_attribute_count"] >= 3
    assert report["redacted_identifier_count"] >= 6
    assert "site down" not in text
    assert "checking" not in text
    assert "https://example.test" not in text
    assert "conv-123" not in text
    assert "ucid-123" not in text
    assert "trace-1" not in text
    assert "care_ai.redacted.input_value.sha256_12" in text
    assert summary["planned_tool_names"] == [{"name": "http_probe", "count": 1}]
    assert summary["tool_names"] == [{"name": "http_probe", "count": 1}]
    assert summary["session_count"] == 1
    assert summary["ucid_count"] == 1


def test_sanitize_cli_json_does_not_print_raw_io(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    output_path = tmp_path / "sanitized.jsonl"

    result = CliRunner().invoke(
        app,
        ["sanitize", str(input_path), str(output_path), "--json"],
    )

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["spans_written"] == 4
    assert report["kept_identifiers"] is False
    assert "site down" not in result.output
    assert "conv-123" not in result.output
    assert "checking" not in result.output
    assert output_path.exists()


def test_sanitize_can_keep_identifiers_when_explicitly_requested(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    output_path = tmp_path / "sanitized.jsonl"

    report = sanitize_halo_jsonl(input_path, output_path, keep_identifiers=True)

    text = output_path.read_text()
    assert report["redacted_identifier_count"] == 0
    assert report["kept_identifiers"] is True
    assert "conv-123" in text
    assert "ucid-123" in text


def test_trace_safety_report_flags_raw_and_sanitized_trace_sets(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    sanitize_halo_jsonl(input_path, sanitized_path)

    raw_report = trace_safety_report(input_path)
    sanitized_report = trace_safety_report(sanitized_path)

    assert raw_report["safe_for_metadata_only_diagnosis"] is False
    assert raw_report["raw_payload_attribute_count"] >= 3
    assert raw_report["possible_raw_identifier_attribute_count"] >= 3
    assert sanitized_report["safe_for_metadata_only_diagnosis"] is True
    assert sanitized_report["raw_payload_attribute_count"] == 0
    assert sanitized_report["possible_raw_identifier_attribute_count"] == 0
    assert sanitized_report["redacted_payload_marker_count"] >= 3
    assert sanitized_report["redacted_identifier_attribute_count"] >= 3
    assert "site down" not in json.dumps(raw_report)
    assert "conv-123" not in json.dumps(raw_report)


def test_build_evidence_pack_includes_safe_metadata_without_raw_io(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    output_path = tmp_path / "sanitized.jsonl"
    sanitize_halo_jsonl(input_path, output_path)

    pack = build_evidence_pack(
        output_path,
        focus=DiagnosticFocus.routing,
        executor="orchestrator-test",
        extra_question="Which trace pattern should we fix first?",
    )
    text = json.dumps(pack, sort_keys=True)

    assert pack["purpose"] == "care-ai-halo-evidence-pack"
    assert pack["scope"]["focus"] == "routing"
    assert pack["inspect"]["total_traces"] == 1
    assert pack["audit"]["counts"]["tool_errors"] == 0
    assert pack["safety"]["safe_for_metadata_only_diagnosis"] is True
    assert pack["doctor"]["trace"]["total_traces"] == 1
    assert "Expected answer shape" in pack["diagnostic_prompt"]
    assert "Which trace pattern should we fix first?" in pack["diagnostic_prompt"]
    assert "site down" not in text
    assert "checking" not in text
    assert "https://example.test" not in text
    assert "conv-123" not in text
    assert "ucid-123" not in text


def test_build_prompt_evidence_summary_is_metadata_safe(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)

    summary = build_prompt_evidence_summary(input_path, top_n=1)

    assert "Deterministic local evidence summary" in summary
    assert '"total_traces": 1' in summary
    assert '"raw_payload_attribute_count":' in summary
    assert "Use this summary as the starting point" in summary
    assert "site down" not in summary
    assert "checking" not in summary
    assert "https://example.test" not in summary
    assert "conv-123" not in summary
    assert "ucid-123" not in summary


def test_build_diagnosis_prompt_with_evidence_reuses_safe_summary(tmp_path: Path) -> None:
    trace_path = _write_trace_fixture(tmp_path)

    prompt = build_diagnosis_prompt_with_evidence(
        trace_path,
        focus=DiagnosticFocus.cost,
        executor="orchestrator-test",
        extra_question="Which span class drives spend?",
        evidence_top=1,
    )

    assert "Deterministic local evidence summary" in prompt
    assert "Expected answer shape" in prompt
    assert "Which span class drives spend?" in prompt
    assert "site down" not in prompt
    assert "https://example.test" not in prompt


def test_build_pipeline_paths_slugifies_executor_and_focus(tmp_path: Path) -> None:
    trace_path = tmp_path / "care traces.sanitized.jsonl"

    paths = build_pipeline_paths(
        trace_path,
        tmp_path / "run dir",
        focus=DiagnosticFocus.tool_errors,
        executor="orchestrator test/email",
    )

    assert paths["evidence_pack"].name == "orchestrator-test-email-tool-errors-evidence-pack.json"
    assert paths["diagnosis_report"].name == "orchestrator-test-email-tool-errors.md"
    assert paths["diagnosis_events"].name == "orchestrator-test-email-tool-errors.events.jsonl"
    assert paths["experiment_plan"].name == (
        "orchestrator-test-email-tool-errors-experiment-plan.md"
    )
    assert paths["candidate_handoff"].name == (
        "orchestrator-test-email-tool-errors-candidate-handoff.md"
    )
    assert paths["loop_status"].name == "orchestrator-test-email-tool-errors-loop-status.md"
    assert paths["manifest"].name == "orchestrator-test-email-tool-errors-manifest.json"


def test_local_pipeline_skip_diagnose_writes_artifacts_without_raw_io(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = _write_trace_fixture(tmp_path)
    trace_path = tmp_path / "sanitized.jsonl"
    output_dir = tmp_path / "pipeline"
    sanitize_halo_jsonl(input_path, trace_path)
    monkeypatch.setattr("halo_cli.care_ai.Sandbox.get", lambda: object())

    result = CliRunner().invoke(
        app,
        [
            "local-pipeline",
            str(trace_path),
            str(output_dir),
            "--focus",
            "cost",
            "--executor",
            "orchestrator-test",
        ],
    )

    assert result.exit_code == 0
    paths = build_pipeline_paths(
        trace_path,
        output_dir,
        focus=DiagnosticFocus.cost,
        executor="orchestrator-test",
    )
    assert paths["evidence_pack"].exists()
    assert paths["experiment_plan"].exists()
    assert paths["candidate_handoff"].exists()
    assert paths["loop_status"].exists()
    assert paths["manifest"].exists()
    assert not paths["diagnosis_report"].exists()
    manifest = json.loads(paths["manifest"].read_text())
    artifact_text = paths["evidence_pack"].read_text() + paths["experiment_plan"].read_text()
    assert manifest["diagnosis"]["enabled"] is False
    assert manifest["diagnosis"]["status"] == "skipped"
    assert manifest["summary"]["total_traces"] == 1
    assert manifest["artifacts"]["loop_status"] == str(paths["loop_status"])
    assert "State: `baseline_ready`" in paths["loop_status"].read_text()
    assert "site down" not in artifact_text
    assert "https://example.test" not in artifact_text
    assert "conv-123" not in artifact_text


def test_local_pipeline_diagnose_writes_report_events_and_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = _write_trace_fixture(tmp_path)
    trace_path = tmp_path / "sanitized.jsonl"
    output_dir = tmp_path / "pipeline"
    sanitize_halo_jsonl(input_path, trace_path)
    monkeypatch.setattr("halo_cli.care_ai.Sandbox.get", lambda: object())
    observed: dict[str, object] = {}

    def capture_run_trace(**kwargs):
        observed["trace_detail_tools_enabled"] = kwargs["trace_detail_tools_enabled"]
        observed["run_code_enabled"] = kwargs["run_code_enabled"]
        observed["timeout_seconds"] = kwargs["timeout_seconds"]
        observed["output_path"] = kwargs["output_path"]
        observed["events_path"] = kwargs["events_path"]
        kwargs["output_path"].write_text(
            "### Recommended next experiment\n\nRun a reduced-context candidate.\n"
        )
        kwargs["events_path"].write_text('{"event":"done"}\n')

    monkeypatch.setattr("halo_cli.care_ai.run_trace", capture_run_trace)

    result = CliRunner().invoke(
        app,
        [
            "local-pipeline",
            str(trace_path),
            str(output_dir),
            "--focus",
            "cost",
            "--executor",
            "orchestrator-test",
            "--diagnose",
            "--no-gocode",
        ],
    )

    assert result.exit_code == 0
    paths = build_pipeline_paths(
        trace_path,
        output_dir,
        focus=DiagnosticFocus.cost,
        executor="orchestrator-test",
    )
    assert observed == {
        "trace_detail_tools_enabled": False,
        "run_code_enabled": False,
        "timeout_seconds": 60,
        "output_path": paths["diagnosis_report"],
        "events_path": paths["diagnosis_events"],
    }
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["diagnosis"]["enabled"] is True
    assert manifest["diagnosis"]["status"] == "completed"
    assert paths["diagnosis_report"].exists()
    assert paths["diagnosis_events"].exists()
    assert "Run a reduced-context candidate" in paths["experiment_plan"].read_text()
    assert "## Proposed Prompt Addendum" in paths["candidate_handoff"].read_text()
    assert "State: `baseline_ready`" in paths["loop_status"].read_text()


def test_candidate_local_pipeline_writes_local_artifact_chain_without_raw_manifest(
    tmp_path: Path,
) -> None:
    input_path = _write_trace_fixture(tmp_path)
    trace_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    diagnosis_path = tmp_path / "diagnosis.md"
    prompt_path = _write_prompt_fixture(tmp_path)
    output_dir = tmp_path / "candidate-local"
    sanitize_halo_jsonl(input_path, trace_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                trace_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )
    diagnosis_path.write_text(
        "### Recommended next experiment\n\n"
        "Run a reduced-context candidate against the same scenarios.\n"
    )

    manifest = build_candidate_local_pipeline(
        prompt_path,
        evidence_path,
        output_dir,
        environment="test",
        diagnosis_report_path=diagnosis_path,
        trace_output_dir=tmp_path / "candidate-runtime",
    )

    paths = build_candidate_pipeline_paths(
        evidence_path,
        output_dir,
        focus=DiagnosticFocus.cost,
        executor="orchestrator-test",
    )
    assert manifest["purpose"] == "care-ai-halo-candidate-local-pipeline"
    assert manifest["loop_status"]["state"] == "ready_for_approved_candidate_push"
    assert manifest["loop_status"]["external_approval_required"] is True
    assert manifest["safety"]["mutated_langfuse"] is False
    for key in [
        "prompt_snapshot",
        "candidate_prompt",
        "candidate_metadata",
        "candidate_review",
        "candidate_review_json",
        "runtime_plan",
        "runtime_plan_json",
        "preflight",
        "preflight_json",
        "loop_status",
        "manifest",
    ]:
        assert paths[key].exists()
    candidate_prompt_text = paths["candidate_prompt"].read_text()
    assert "SYSTEM: secret routing prompt" in candidate_prompt_text
    assert "Observed HALO evidence to address:" in candidate_prompt_text
    assert "Run a reduced-context candidate" in candidate_prompt_text
    metadata_text = paths["candidate_metadata"].read_text()
    review_text = paths["candidate_review_json"].read_text()
    manifest_text = paths["manifest"].read_text()
    assert "SYSTEM: secret routing prompt" not in metadata_text
    assert "SYSTEM: secret routing prompt" not in review_text
    assert "SYSTEM: secret routing prompt" not in manifest_text
    assert "State: `ready_for_approved_candidate_push`" in paths["loop_status"].read_text()
    assert "`runtime_check` | `missing`" in paths["loop_status"].read_text()


def test_candidate_local_pipeline_cli_writes_manifest(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    trace_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    output_dir = tmp_path / "candidate-local"
    sanitize_halo_jsonl(input_path, trace_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                trace_path,
                focus=DiagnosticFocus.routing,
                executor="orchestrator-test",
            )
        )
    )

    result = CliRunner().invoke(
        app,
        [
            "candidate-local-pipeline",
            str(prompt_path),
            str(evidence_path),
            str(output_dir),
            "--environment",
            "test",
        ],
    )

    assert result.exit_code == 0
    assert "state=ready_for_approved_candidate_push" in result.output
    paths = build_candidate_pipeline_paths(
        evidence_path,
        output_dir,
        focus=DiagnosticFocus.routing,
        executor="orchestrator-test",
    )
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["focus"] == "routing"
    assert manifest["executor"] == "orchestrator-test"
    assert manifest["loop_status"]["state"] == "ready_for_approved_candidate_push"


def test_evidence_pack_fingerprints_status_messages(tmp_path: Path) -> None:
    trace_path = _write_trace_fixture(tmp_path)
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    for record in records:
        if record["attributes"].get("tool.name") == "http_probe":
            record["status"] = {
                "code": "STATUS_CODE_ERROR",
                "message": "raw backend diagnostic failure",
            }
    trace_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    pack = build_evidence_pack(trace_path)
    text = json.dumps(pack, sort_keys=True)
    error = pack["audit"]["tool_errors"][0]

    assert pack["audit"]["counts"]["tool_errors"] == 1
    assert "status_message" not in error
    assert error["status_message_bytes"] == len("raw backend diagnostic failure")
    assert "status_message_sha256_12" in error
    assert pack["safety"]["raw_status_message_count"] == 1
    assert "raw backend diagnostic failure" not in text


def test_evidence_pack_can_compare_candidate_trace_set(tmp_path: Path) -> None:
    baseline_payload = _langfuse_bundle()
    baseline_payload["observations"][1]["usageDetails"] = {
        "input": 40_001,
        "output": 3,
        "total": 40_004,
    }

    candidate_payload = _langfuse_bundle()
    candidate_payload["observations"][1]["usageDetails"] = {
        "input": 10_000,
        "output": 3,
        "total": 10_003,
    }

    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    write_halo_jsonl(convert_langfuse_export_to_spans(baseline_payload), baseline_path)
    write_halo_jsonl(convert_langfuse_export_to_spans(candidate_payload), candidate_path)

    pack = build_evidence_pack(baseline_path, candidate_path=candidate_path)

    assert pack["comparison"]["token_totals"]["input"]["delta"] == -30_001
    assert pack["comparison"]["token_distributions"]["input"]["delta"]["p95"] == -30_001
    assert pack["comparison"]["token_distributions"]["input"]["direction_by_p95"] == "improved"
    assert (
        pack["comparison"]["session_token_distributions"]["input"]["direction_by_p95"] == "improved"
    )
    assert pack["candidate"]["inspect"]["total_traces"] == 1
    assert pack["candidate"]["audit"]["counts"]["high_token_llm_spans"] == 0


def test_audit_flags_agent_reported_tool_count_mismatch(tmp_path: Path) -> None:
    payload = _langfuse_bundle()
    payload["observations"][0]["output"] = {
        "responseCount": 1,
        "toolCallCount": 2,
        "finalOutputLength": 18,
    }
    payload["observations"][2]["metadata"].pop("traceKind")
    payload["observations"][2]["metadata"].pop("toolName")
    trace_path = tmp_path / "mismatch.jsonl"
    write_halo_jsonl(convert_langfuse_export_to_spans(payload), trace_path)

    report = audit_halo_jsonl(trace_path)
    inspect_report = inspect_halo_jsonl(trace_path)

    assert inspect_report["total_agent_reported_tool_calls"] == 2
    assert report["counts"]["reported_tool_count_mismatch"] == 1
    mismatch = report["reported_tool_count_mismatch"][0]
    assert mismatch["reported_tool_call_count"] == 2
    assert mismatch["executed_tool_count"] == 1
    assert mismatch["executed_tool_names"] == ["http_probe"]
    assert report["counts"]["planned_missing_tool_calls"] == 0


def test_evidence_pack_cli_writes_json_without_raw_io(tmp_path: Path, monkeypatch) -> None:
    input_path = _write_trace_fixture(tmp_path)
    output_path = tmp_path / "sanitized.jsonl"
    pack_path = tmp_path / "pack" / "evidence.json"
    sanitize_halo_jsonl(input_path, output_path)
    monkeypatch.setattr("halo_cli.care_ai.Sandbox.get", lambda: object())

    result = CliRunner().invoke(
        app,
        [
            "evidence-pack",
            str(output_path),
            str(pack_path),
            "--focus",
            "cost",
            "--executor",
            "orchestrator-test",
        ],
    )

    assert result.exit_code == 0
    assert pack_path.exists()
    pack_text = pack_path.read_text()
    pack = json.loads(pack_text)
    assert pack["scope"]["focus"] == "cost"
    assert pack["care_ai_surface"]["prompt_name"] == ("c1/airo-care/orchestrator-test/orchestrator")
    assert pack["safety"]["safe_for_metadata_only_diagnosis"] is True
    assert "safe_metadata=True" in result.output
    assert "site down" not in pack_text
    assert "https://example.test" not in pack_text


def test_build_experiment_plan_from_cost_evidence_pack(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    output_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    diagnosis_path = tmp_path / "diagnosis.md"
    sanitize_halo_jsonl(input_path, output_path)
    pack = build_evidence_pack(
        output_path,
        focus=DiagnosticFocus.cost,
        executor="orchestrator-test",
    )
    pack["audit"]["counts"]["high_token_llm_spans"] = 3
    pack["inspect"]["total_input_tokens"] = 12345
    evidence_path.write_text(json.dumps(pack))
    diagnosis_path.write_text(
        "### Recommended next experiment\n\n"
        "Run a reduced-context candidate against the same scenarios.\n\n"
        "### No-change call\n\nDo not swap models yet.\n"
    )

    plan = build_experiment_plan(evidence_path, diagnosis_report_path=diagnosis_path)
    markdown = render_experiment_plan_markdown(plan)

    assert plan["purpose"] == "care-ai-halo-experiment-plan"
    assert plan["title"] == "Reduce high input-token spend in orchestrator-test"
    assert plan["care_ai_surface"]["prompt_name"] == ("c1/airo-care/orchestrator-test/orchestrator")
    assert "3 high-token LLM spans" in plan["hypothesis"]
    assert "caps or summarizes" in plan["candidate_change"]
    assert "c1/airo-care/orchestrator-test/orchestrator" in plan["candidate_change"]
    assert "median and p95" in "\n".join(plan["success_metrics"])
    assert "compare-audits" in "\n".join(plan["verification_commands"])
    assert "Run a reduced-context candidate" in plan["diagnosis_recommended_experiment_excerpt"]
    assert "# Reduce high input-token spend in orchestrator-test" in markdown
    assert "## Care AI Surface" in markdown
    assert "c1/airo-care/orchestrator-test/orchestrator" in markdown
    assert "orchestrator-test/index.ts" in markdown
    assert "Input tokens: 12345" in markdown
    assert "Input token distribution:" in markdown
    assert "site down" not in markdown
    assert "https://example.test" not in markdown


def test_experiment_plan_cli_writes_markdown_and_json(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    markdown_path = tmp_path / "plan.md"
    json_path = tmp_path / "plan.json"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.routing,
                executor="orchestrator-test",
            )
        )
    )

    markdown_result = CliRunner().invoke(
        app,
        ["experiment-plan", str(evidence_path), str(markdown_path)],
    )
    json_result = CliRunner().invoke(
        app,
        [
            "experiment-plan",
            str(evidence_path),
            str(json_path),
            "--format",
            ExperimentPlanFormat.json.value,
        ],
    )

    assert markdown_result.exit_code == 0
    assert json_result.exit_code == 0
    assert markdown_path.read_text().startswith("# Validate routing behavior")
    payload = json.loads(json_path.read_text())
    assert payload["scope"]["focus"] == "routing"
    assert payload["scope"]["executor"] == "orchestrator-test"


def test_build_candidate_handoff_from_cost_evidence_pack(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    diagnosis_path = tmp_path / "diagnosis.md"
    sanitize_halo_jsonl(input_path, sanitized_path)
    pack = build_evidence_pack(
        sanitized_path,
        focus=DiagnosticFocus.cost,
        executor="orchestrator-test",
    )
    pack["audit"]["counts"]["high_token_llm_spans"] = 3
    pack["inspect"]["llm_input_token_distribution"] = {
        "count": 4,
        "max": 63816,
        "mean": 56175.32,
        "median": 62644,
        "min": 21985,
        "p95": 63816,
        "p99": 63816,
        "total": 224701,
    }
    pack["inspect"]["session_input_token_distribution"] = {
        "count": 2,
        "max": 444234,
        "mean": 407062.5,
        "median": 407062.5,
        "min": 369891,
        "p95": 444234,
        "p99": 444234,
        "total": 814125,
    }
    pack["inspect"]["top_sessions_by_input_tokens"] = [
        {
            "session_sha256_12": "a637a5ece030",
            "trace_count": 5,
            "span_count": 29,
            "input_tokens": 444234,
            "output_tokens": 601,
            "reported_tool_calls": 6,
            "executed_tool_calls": 6,
            "planned_tool_calls": 6,
            "error_trace_count": 0,
        }
    ]
    evidence_path.write_text(json.dumps(pack))
    diagnosis_path.write_text(
        "\n".join(
            [
                "## 2) Proposed prompt / configuration improvement",
                "",
                "### Improvement ideas",
                "",
                "The metadata suggests an opportunity to reduce prompt growth by:",
                "- tightening what prior conversation or tool output is re-injected before each LLM call",
                "- limiting repeated routing passes when the context is already large",
                "- ignore trace `d882c642a1091b86428955192683ba90` and https://example.test here",
                "",
                "## 5) Verification plan",
                "",
                "Compare token p95.",
            ]
        )
    )

    handoff = build_candidate_handoff(evidence_path, diagnosis_report_path=diagnosis_path)
    markdown = render_candidate_handoff_markdown(handoff)

    assert handoff["purpose"] == "care-ai-halo-candidate-handoff"
    assert handoff["care_ai_surface"]["prompt_name"] == (
        "c1/airo-care/orchestrator-test/orchestrator"
    )
    assert handoff["proposed_candidate"]["change_type"] == "prompt_addendum"
    addendum = handoff["proposed_candidate"]["addendum"]
    assert "context budget and routing discipline" in addendum
    assert "`3` high-input-token LLM spans across `1` traces" in addendum
    assert "Input-token median `62644`, p95 `63816`, max `63816`" in addendum
    assert "Session input-token median `407062.5`, p95 `444234`, max `444234`" in addendum
    assert "Highest-cost session used `444234` input tokens across `5` traces" in addendum
    assert "tightening what prior conversation or tool output" in addendum
    assert "redacted-id" in addendum
    assert "[redacted-url]" in addendum
    assert "d882c642a1091b86428955192683ba90" not in addendum
    assert "https://example.test" not in addendum
    assert "session-executor wiring" not in addendum
    assert "lf api prompts get c1/airo-care/orchestrator-test/orchestrator" in "\n".join(
        handoff["langfuse_commands"]
    )
    assert "--label latest" in "\n".join(handoff["langfuse_commands"])
    assert "uv run halo-careai prompt-snapshot" in "\n".join(handoff["langfuse_commands"])
    assert "uv run halo-careai candidate-prompt-file" in "\n".join(handoff["langfuse_commands"])
    assert "--curl" in "\n".join(handoff["langfuse_commands"])
    assert ".create.curl" in "\n".join(handoff["langfuse_commands"])
    assert "> .halo-careai/prompts/" in "\n".join(handoff["langfuse_commands"])
    assert "uv run halo-careai candidate-review" in "\n".join(handoff["langfuse_commands"])
    assert "uv run halo-careai candidate-runtime-plan" in "\n".join(handoff["langfuse_commands"])
    assert "uv run halo-careai candidate-preflight" in "\n".join(handoff["langfuse_commands"])
    assert "uv run halo-careai candidate-runtime-check" in "\n".join(handoff["langfuse_commands"])
    assert "uv run halo-careai candidate-evaluate" in "\n".join(handoff["langfuse_commands"])
    assert "uv run halo-careai loop-status" in "\n".join(handoff["langfuse_commands"])
    assert "runtime-plan.json" in "\n".join(handoff["langfuse_commands"])
    assert "## Proposed Prompt Addendum" in markdown
    assert "site down" not in markdown
    assert "https://example.test" not in markdown


def test_candidate_handoff_recovers_surface_for_legacy_evidence_pack(
    tmp_path: Path,
) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    sanitize_halo_jsonl(input_path, sanitized_path)
    pack = build_evidence_pack(
        sanitized_path,
        focus=DiagnosticFocus.cost,
        executor="orchestrator-test",
    )
    pack.pop("care_ai_surface")
    evidence_path.write_text(json.dumps(pack))

    handoff = build_candidate_handoff(evidence_path)

    assert handoff["care_ai_surface"]["prompt_name"] == (
        "c1/airo-care/orchestrator-test/orchestrator"
    )
    assert "<prompt-name>" not in "\n".join(handoff["langfuse_commands"])
    assert "c1/airo-care/orchestrator-test/orchestrator" in "\n".join(handoff["langfuse_commands"])


def test_build_prompt_snapshot_fingerprints_prompt_without_body(tmp_path: Path) -> None:
    prompt_path = _write_prompt_fixture(tmp_path)

    snapshot = build_prompt_snapshot(prompt_path)
    snapshot_text = json.dumps(snapshot)

    assert snapshot["purpose"] == "care-ai-halo-prompt-snapshot"
    assert snapshot["prompt"]["name"] == "c1/airo-care/orchestrator-test/orchestrator"
    assert snapshot["prompt"]["version"] == 42
    assert snapshot["prompt"]["labels"] == ["latest"]
    assert snapshot["body"]["sha256_12"]
    assert snapshot["body"]["bytes"] == len("SYSTEM: secret routing prompt\nDo the thing.")
    assert snapshot["body"]["line_count"] == 2
    assert snapshot["config"]["keys"] == ["model"]
    assert snapshot["safety"]["raw_prompt_retained"] is False
    assert "secret routing prompt" not in snapshot_text
    assert "gpt-5.4" not in snapshot_text


def test_prompt_snapshot_cli_writes_metadata_only_json(tmp_path: Path) -> None:
    prompt_path = _write_prompt_fixture(tmp_path)
    snapshot_path = tmp_path / "prompt.snapshot.json"

    result = CliRunner().invoke(
        app,
        ["prompt-snapshot", str(prompt_path), str(snapshot_path)],
    )

    assert result.exit_code == 0
    payload = json.loads(snapshot_path.read_text())
    assert payload["body"]["sha256_12"] in result.output
    assert payload["safety"]["raw_prompt_retained"] is False
    assert "secret routing prompt" not in snapshot_path.read_text()
    assert "raw_prompt_retained=False" in result.output


def test_candidate_handoff_can_include_prompt_snapshot(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    snapshot_path = tmp_path / "prompt.snapshot.json"
    sanitize_halo_jsonl(input_path, sanitized_path)
    snapshot_path.write_text(json.dumps(build_prompt_snapshot(prompt_path)))
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
                prompt_snapshot_path=snapshot_path,
            )
        )
    )

    handoff = build_candidate_handoff(evidence_path)
    markdown = render_candidate_handoff_markdown(handoff)

    assert handoff["prompt_snapshot"]["prompt"]["version"] == 42
    assert "## Prompt Snapshot" in markdown
    assert "secret routing prompt" not in markdown


def test_build_candidate_prompt_artifact_appends_addendum_without_metadata_body(
    tmp_path: Path,
) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    candidate_path = tmp_path / "candidate.txt"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )

    candidate_text, metadata = build_candidate_prompt_artifact(
        prompt_path,
        evidence_path,
        output_path=candidate_path,
    )
    metadata_text = json.dumps(metadata)

    assert "SYSTEM: secret routing prompt" in candidate_text
    assert "[HALO candidate addendum begins]" in candidate_text
    assert "context budget and routing discipline" in candidate_text
    assert metadata["purpose"] == "care-ai-halo-candidate-prompt-file"
    assert metadata["prompt"]["name"] == "c1/airo-care/orchestrator-test/orchestrator"
    assert metadata["prompt"]["source_version"] == 42
    assert metadata["safety"]["mutated_langfuse"] is False
    assert metadata["safety"]["raw_prompt_retained_in_metadata"] is False
    assert "secret routing prompt" not in metadata_text
    assert "context budget and routing discipline" not in metadata_text


def test_candidate_prompt_file_cli_writes_candidate_and_metadata(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    candidate_path = tmp_path / "candidate.txt"
    metadata_path = tmp_path / "candidate.metadata.json"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.routing,
                executor="orchestrator-test",
            )
        )
    )

    result = CliRunner().invoke(
        app,
        [
            "candidate-prompt-file",
            str(prompt_path),
            str(evidence_path),
            str(candidate_path),
            "--metadata-output",
            str(metadata_path),
        ],
    )

    assert result.exit_code == 0
    assert "mutated_langfuse=False" in result.output
    assert "evidence-gated routing" in candidate_path.read_text()
    metadata_text = metadata_path.read_text()
    assert "secret routing prompt" not in metadata_text
    assert "evidence-gated routing" not in metadata_text


def test_build_candidate_review_verifies_candidate_without_full_prompt_body(
    tmp_path: Path,
) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    candidate_path = tmp_path / "candidate.txt"
    metadata_path = tmp_path / "candidate.metadata.json"
    create_curl_path = tmp_path / "candidate.create.curl"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )
    candidate_text, metadata = build_candidate_prompt_artifact(
        prompt_path,
        evidence_path,
        output_path=candidate_path,
    )
    candidate_path.write_text(candidate_text)
    metadata_path.write_text(json.dumps(metadata))
    create_curl_path.write_text("curl -sS -X POST https://example.test c1/airo-care/")

    review = build_candidate_review(
        prompt_path,
        candidate_path,
        metadata_path,
        create_curl_path=create_curl_path,
    )
    markdown = render_candidate_review_markdown(review)
    review_text = json.dumps(review)

    assert review["purpose"] == "care-ai-halo-candidate-review"
    assert review["ready_for_human_review"] is True
    assert review["body_deltas"]["bytes"] > 0
    assert "context budget and routing discipline" in markdown
    assert "secret routing prompt" not in markdown
    assert "secret routing prompt" not in review_text
    assert all(check["passed"] for check in review["checks"])


def test_candidate_review_cli_writes_markdown_and_json(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    candidate_path = tmp_path / "candidate.txt"
    metadata_path = tmp_path / "candidate.metadata.json"
    markdown_path = tmp_path / "candidate-review.md"
    json_path = tmp_path / "candidate-review.json"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.routing,
                executor="orchestrator-test",
            )
        )
    )
    candidate_text, metadata = build_candidate_prompt_artifact(
        prompt_path,
        evidence_path,
        output_path=candidate_path,
    )
    candidate_path.write_text(candidate_text)
    metadata_path.write_text(json.dumps(metadata))

    markdown_result = CliRunner().invoke(
        app,
        [
            "candidate-review",
            str(prompt_path),
            str(candidate_path),
            str(metadata_path),
            str(markdown_path),
        ],
    )
    json_result = CliRunner().invoke(
        app,
        [
            "candidate-review",
            str(prompt_path),
            str(candidate_path),
            str(metadata_path),
            str(json_path),
            "--format",
            ExperimentPlanFormat.json.value,
        ],
    )

    assert markdown_result.exit_code == 0
    assert json_result.exit_code == 0
    assert "ready_for_human_review=True" in markdown_result.output
    assert "evidence-gated routing" in markdown_path.read_text()
    assert "secret routing prompt" not in markdown_path.read_text()
    payload = json.loads(json_path.read_text())
    assert payload["ready_for_human_review"] is True


def test_build_candidate_runtime_plan_is_approval_gated(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    candidate_path = tmp_path / "candidate.txt"
    metadata_path = tmp_path / "candidate.metadata.json"
    review_path = tmp_path / "candidate-review.json"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )
    candidate_text, metadata = build_candidate_prompt_artifact(
        prompt_path,
        evidence_path,
        output_path=candidate_path,
    )
    candidate_path.write_text(candidate_text)
    metadata_path.write_text(json.dumps(metadata))
    review = build_candidate_review(prompt_path, candidate_path, metadata_path)
    review_path.write_text(json.dumps(review))

    plan = build_candidate_runtime_plan(
        review_path,
        environment="test",
        focus=DiagnosticFocus.cost,
        trace_output_dir=tmp_path / "candidate-runtime",
    )
    markdown = render_candidate_runtime_plan_markdown(plan)

    assert plan["purpose"] == "care-ai-halo-candidate-runtime-plan"
    assert plan["approval_required"] is True
    assert plan["scope"]["executor"] == "orchestrator-test"
    assert "`halo-candidate` is a review label only" in plan["runtime_label_warning"]
    assert "--labels latest,halo-candidate" in "\n".join(plan["approved_mutation_commands"])
    assert "compare-audits" in "\n".join(plan["comparison_commands"])
    assert "candidate-evaluate" in "\n".join(plan["comparison_commands"])
    assert "rollback" in "\n".join(plan["rollback_commands"])
    assert "secret routing prompt" not in markdown


def test_candidate_runtime_plan_cli_writes_markdown_and_json(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    candidate_path = tmp_path / "candidate.txt"
    metadata_path = tmp_path / "candidate.metadata.json"
    review_path = tmp_path / "candidate-review.json"
    markdown_path = tmp_path / "runtime-plan.md"
    json_path = tmp_path / "runtime-plan.json"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.routing,
                executor="orchestrator-test",
            )
        )
    )
    candidate_text, metadata = build_candidate_prompt_artifact(
        prompt_path,
        evidence_path,
        output_path=candidate_path,
    )
    candidate_path.write_text(candidate_text)
    metadata_path.write_text(json.dumps(metadata))
    review_path.write_text(
        json.dumps(build_candidate_review(prompt_path, candidate_path, metadata_path))
    )

    markdown_result = CliRunner().invoke(
        app,
        [
            "candidate-runtime-plan",
            str(review_path),
            str(markdown_path),
            "--environment",
            "test",
            "--focus",
            DiagnosticFocus.routing.value,
            "--trace-output-dir",
            str(tmp_path / "candidate-runtime"),
        ],
    )
    json_result = CliRunner().invoke(
        app,
        [
            "candidate-runtime-plan",
            str(review_path),
            str(json_path),
            "--format",
            ExperimentPlanFormat.json.value,
            "--trace-output-dir",
            str(tmp_path / "candidate-runtime"),
        ],
    )

    assert markdown_result.exit_code == 0
    assert json_result.exit_code == 0
    assert "approval_required=True" in markdown_result.output
    assert "## Approved Mutation Commands" in markdown_path.read_text()
    assert "secret routing prompt" not in markdown_path.read_text()
    payload = json.loads(json_path.read_text())
    assert payload["approval_required"] is True


def test_build_candidate_preflight_passes_for_current_review_and_runtime_plan(
    tmp_path: Path,
) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    candidate_path = tmp_path / "candidate.txt"
    metadata_path = tmp_path / "candidate.metadata.json"
    review_path = tmp_path / "candidate-review.json"
    runtime_plan_path = tmp_path / "runtime-plan.json"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )
    candidate_text, metadata = build_candidate_prompt_artifact(
        prompt_path,
        evidence_path,
        output_path=candidate_path,
    )
    candidate_path.write_text(candidate_text)
    metadata_path.write_text(json.dumps(metadata))
    review_path.write_text(
        json.dumps(build_candidate_review(prompt_path, candidate_path, metadata_path))
    )
    runtime_plan_path.write_text(
        json.dumps(
            build_candidate_runtime_plan(
                review_path,
                environment="test",
                focus=DiagnosticFocus.cost,
                trace_output_dir=tmp_path / "candidate-runtime",
            )
        )
    )

    preflight = build_candidate_preflight(prompt_path, review_path, runtime_plan_path)
    markdown = render_candidate_preflight_markdown(preflight)
    preflight_text = json.dumps(preflight)

    assert preflight["purpose"] == "care-ai-halo-candidate-preflight"
    assert preflight["preflight_passed"] is True
    assert preflight["next_action"] == "ready_for_approval"
    assert all(check["passed"] for check in preflight["checks"])
    assert "--labels latest,halo-candidate" not in markdown
    assert "secret routing prompt" not in markdown
    assert "secret routing prompt" not in preflight_text


def test_build_candidate_preflight_fails_when_current_prompt_is_stale(
    tmp_path: Path,
) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    stale_prompt_path = tmp_path / "stale-prompt.json"
    candidate_path = tmp_path / "candidate.txt"
    metadata_path = tmp_path / "candidate.metadata.json"
    review_path = tmp_path / "candidate-review.json"
    runtime_plan_path = tmp_path / "runtime-plan.json"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )
    candidate_text, metadata = build_candidate_prompt_artifact(
        prompt_path,
        evidence_path,
        output_path=candidate_path,
    )
    candidate_path.write_text(candidate_text)
    metadata_path.write_text(json.dumps(metadata))
    review_path.write_text(
        json.dumps(build_candidate_review(prompt_path, candidate_path, metadata_path))
    )
    runtime_plan_path.write_text(
        json.dumps(
            build_candidate_runtime_plan(
                review_path,
                environment="test",
                focus=DiagnosticFocus.cost,
                trace_output_dir=tmp_path / "candidate-runtime",
            )
        )
    )
    stale_payload = json.loads(prompt_path.read_text())
    stale_payload["body"]["version"] = 43
    stale_payload["body"]["prompt"] = "SYSTEM: different current prompt\nDo the thing."
    stale_prompt_path.write_text(json.dumps(stale_payload))

    preflight = build_candidate_preflight(stale_prompt_path, review_path, runtime_plan_path)

    assert preflight["preflight_passed"] is False
    assert preflight["next_action"] == "refresh_candidate_artifacts"
    failed_checks = {check["name"] for check in preflight["checks"] if check["passed"] is not True}
    assert "source_version_matches_review" in failed_checks
    assert "current_prompt_hash_matches_review" in failed_checks


def test_candidate_preflight_cli_writes_markdown_and_json(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    candidate_path = tmp_path / "candidate.txt"
    metadata_path = tmp_path / "candidate.metadata.json"
    review_path = tmp_path / "candidate-review.json"
    runtime_plan_path = tmp_path / "runtime-plan.json"
    markdown_path = tmp_path / "preflight.md"
    json_path = tmp_path / "preflight.json"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.routing,
                executor="orchestrator-test",
            )
        )
    )
    candidate_text, metadata = build_candidate_prompt_artifact(
        prompt_path,
        evidence_path,
        output_path=candidate_path,
    )
    candidate_path.write_text(candidate_text)
    metadata_path.write_text(json.dumps(metadata))
    review_path.write_text(
        json.dumps(build_candidate_review(prompt_path, candidate_path, metadata_path))
    )
    runtime_plan_path.write_text(
        json.dumps(
            build_candidate_runtime_plan(
                review_path,
                environment="test",
                focus=DiagnosticFocus.routing,
                trace_output_dir=tmp_path / "candidate-runtime",
            )
        )
    )

    markdown_result = CliRunner().invoke(
        app,
        [
            "candidate-preflight",
            str(prompt_path),
            str(review_path),
            str(runtime_plan_path),
            str(markdown_path),
        ],
    )
    json_result = CliRunner().invoke(
        app,
        [
            "candidate-preflight",
            str(prompt_path),
            str(review_path),
            str(runtime_plan_path),
            str(json_path),
            "--format",
            ExperimentPlanFormat.json.value,
        ],
    )

    assert markdown_result.exit_code == 0
    assert json_result.exit_code == 0
    assert "preflight_passed=True" in markdown_result.output
    assert "## Checks" in markdown_path.read_text()
    assert "secret routing prompt" not in markdown_path.read_text()
    payload = json.loads(json_path.read_text())
    assert payload["preflight_passed"] is True


def test_build_candidate_runtime_check_verifies_created_prompt_without_body(
    tmp_path: Path,
) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    candidate_path = tmp_path / "candidate.txt"
    metadata_path = tmp_path / "candidate.metadata.json"
    review_path = tmp_path / "candidate-review.json"
    runtime_plan_path = tmp_path / "runtime-plan.json"
    preflight_path = tmp_path / "preflight.json"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )
    candidate_text, metadata = build_candidate_prompt_artifact(
        prompt_path,
        evidence_path,
        output_path=candidate_path,
    )
    candidate_path.write_text(candidate_text)
    metadata_path.write_text(json.dumps(metadata))
    review_path.write_text(
        json.dumps(build_candidate_review(prompt_path, candidate_path, metadata_path))
    )
    runtime_plan_path.write_text(
        json.dumps(
            build_candidate_runtime_plan(
                review_path,
                environment="test",
                focus=DiagnosticFocus.cost,
                trace_output_dir=tmp_path / "candidate-runtime",
            )
        )
    )
    preflight_path.write_text(
        json.dumps(build_candidate_preflight(prompt_path, review_path, runtime_plan_path))
    )
    created_path = _write_created_prompt_fixture(tmp_path, candidate_body=candidate_text)

    runtime_check = build_candidate_runtime_check(
        created_path,
        review_path,
        runtime_plan_path,
        preflight_path=preflight_path,
    )
    markdown = render_candidate_runtime_check_markdown(runtime_check)
    runtime_check_text = json.dumps(runtime_check)

    assert runtime_check["purpose"] == "care-ai-halo-candidate-runtime-check"
    assert runtime_check["runtime_check_passed"] is True
    assert runtime_check["next_action"] == "rehydrate_prompt_cache_and_collect_candidate_traces"
    assert all(check["passed"] for check in runtime_check["checks"])
    assert "secret routing prompt" not in markdown
    assert "secret routing prompt" not in runtime_check_text
    assert "context budget and routing discipline" not in runtime_check_text


def test_build_candidate_runtime_check_accepts_langfuse_terminal_newline_normalization(
    tmp_path: Path,
) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    candidate_path = tmp_path / "candidate.txt"
    metadata_path = tmp_path / "candidate.metadata.json"
    review_path = tmp_path / "candidate-review.json"
    runtime_plan_path = tmp_path / "runtime-plan.json"
    preflight_path = tmp_path / "preflight.json"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )
    candidate_text, metadata = build_candidate_prompt_artifact(
        prompt_path,
        evidence_path,
        output_path=candidate_path,
    )
    assert candidate_text.endswith("\n")
    candidate_path.write_text(candidate_text)
    metadata_path.write_text(json.dumps(metadata))
    review_path.write_text(
        json.dumps(build_candidate_review(prompt_path, candidate_path, metadata_path))
    )
    runtime_plan_path.write_text(
        json.dumps(
            build_candidate_runtime_plan(
                review_path,
                environment="test",
                focus=DiagnosticFocus.cost,
                trace_output_dir=tmp_path / "candidate-runtime",
            )
        )
    )
    preflight_path.write_text(
        json.dumps(build_candidate_preflight(prompt_path, review_path, runtime_plan_path))
    )
    created_path = _write_created_prompt_fixture(
        tmp_path,
        candidate_body=candidate_text.rstrip("\n"),
    )

    runtime_check = build_candidate_runtime_check(
        created_path,
        review_path,
        runtime_plan_path,
        preflight_path=preflight_path,
    )

    assert runtime_check["runtime_check_passed"] is True
    assert all(check["passed"] for check in runtime_check["checks"])


def test_build_candidate_runtime_check_fails_without_latest_label(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    candidate_path = tmp_path / "candidate.txt"
    metadata_path = tmp_path / "candidate.metadata.json"
    review_path = tmp_path / "candidate-review.json"
    runtime_plan_path = tmp_path / "runtime-plan.json"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )
    candidate_text, metadata = build_candidate_prompt_artifact(
        prompt_path,
        evidence_path,
        output_path=candidate_path,
    )
    candidate_path.write_text(candidate_text)
    metadata_path.write_text(json.dumps(metadata))
    review_path.write_text(
        json.dumps(build_candidate_review(prompt_path, candidate_path, metadata_path))
    )
    runtime_plan_path.write_text(
        json.dumps(
            build_candidate_runtime_plan(
                review_path,
                environment="test",
                focus=DiagnosticFocus.cost,
                trace_output_dir=tmp_path / "candidate-runtime",
            )
        )
    )
    created_path = _write_created_prompt_fixture(
        tmp_path,
        candidate_body=candidate_text,
        labels=["halo-candidate"],
    )

    runtime_check = build_candidate_runtime_check(created_path, review_path, runtime_plan_path)

    assert runtime_check["runtime_check_passed"] is False
    failed_checks = {
        check["name"] for check in runtime_check["checks"] if check["passed"] is not True
    }
    assert "created_labels_include_runtime_latest" in failed_checks


def test_candidate_runtime_check_cli_writes_markdown_and_json(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    candidate_path = tmp_path / "candidate.txt"
    metadata_path = tmp_path / "candidate.metadata.json"
    review_path = tmp_path / "candidate-review.json"
    runtime_plan_path = tmp_path / "runtime-plan.json"
    markdown_path = tmp_path / "runtime-check.md"
    json_path = tmp_path / "runtime-check.json"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )
    candidate_text, metadata = build_candidate_prompt_artifact(
        prompt_path,
        evidence_path,
        output_path=candidate_path,
    )
    candidate_path.write_text(candidate_text)
    metadata_path.write_text(json.dumps(metadata))
    review_path.write_text(
        json.dumps(build_candidate_review(prompt_path, candidate_path, metadata_path))
    )
    runtime_plan_path.write_text(
        json.dumps(
            build_candidate_runtime_plan(
                review_path,
                environment="test",
                focus=DiagnosticFocus.cost,
                trace_output_dir=tmp_path / "candidate-runtime",
            )
        )
    )
    created_path = _write_created_prompt_fixture(tmp_path, candidate_body=candidate_text)

    markdown_result = CliRunner().invoke(
        app,
        [
            "candidate-runtime-check",
            str(created_path),
            str(review_path),
            str(runtime_plan_path),
            str(markdown_path),
        ],
    )
    json_result = CliRunner().invoke(
        app,
        [
            "candidate-runtime-check",
            str(created_path),
            str(review_path),
            str(runtime_plan_path),
            str(json_path),
            "--format",
            ExperimentPlanFormat.json.value,
        ],
    )

    assert markdown_result.exit_code == 0
    assert json_result.exit_code == 0
    assert "runtime_check_passed=True" in markdown_result.output
    assert "## Checks" in markdown_path.read_text()
    assert "secret routing prompt" not in markdown_path.read_text()
    payload = json.loads(json_path.read_text())
    assert payload["runtime_check_passed"] is True


def test_build_candidate_loop_status_reports_ready_for_approved_candidate_push(
    tmp_path: Path,
) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    candidate_path = tmp_path / "candidate.txt"
    metadata_path = tmp_path / "candidate.metadata.json"
    review_path = tmp_path / "candidate-review.json"
    runtime_plan_path = tmp_path / "runtime-plan.json"
    preflight_path = tmp_path / "preflight.json"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )
    candidate_text, metadata = build_candidate_prompt_artifact(
        prompt_path,
        evidence_path,
        output_path=candidate_path,
    )
    candidate_path.write_text(candidate_text)
    metadata_path.write_text(json.dumps(metadata))
    review_path.write_text(
        json.dumps(build_candidate_review(prompt_path, candidate_path, metadata_path))
    )
    runtime_plan_path.write_text(
        json.dumps(
            build_candidate_runtime_plan(
                review_path,
                environment="test",
                focus=DiagnosticFocus.cost,
                trace_output_dir=tmp_path / "candidate-runtime",
            )
        )
    )
    preflight_path.write_text(
        json.dumps(build_candidate_preflight(prompt_path, review_path, runtime_plan_path))
    )

    status = build_candidate_loop_status(
        evidence_pack_path=evidence_path,
        candidate_review_path=review_path,
        runtime_plan_path=runtime_plan_path,
        preflight_path=preflight_path,
    )
    markdown = render_candidate_loop_status_markdown(status)

    assert status["purpose"] == "care-ai-halo-loop-status"
    assert status["state"] == "ready_for_approved_candidate_push"
    assert status["external_approval_required"] is True
    assert status["local_artifact_chain_complete"] is False
    assert status["loop_complete"] is False
    assert [stage["status"] for stage in status["stages"]] == [
        "passed",
        "passed",
        "passed",
        "passed",
        "missing",
        "missing",
    ]
    assert "secret routing prompt" not in json.dumps(status)
    assert "secret routing prompt" not in markdown


def test_candidate_loop_status_not_complete_when_evaluation_exists_without_runtime_check(
    tmp_path: Path,
) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    candidate_path = tmp_path / "candidate.txt"
    metadata_path = tmp_path / "candidate.metadata.json"
    review_path = tmp_path / "candidate-review.json"
    runtime_plan_path = tmp_path / "runtime-plan.json"
    preflight_path = tmp_path / "preflight.json"
    output_dir = tmp_path / "candidate-evaluation"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )
    candidate_text, metadata = build_candidate_prompt_artifact(
        prompt_path,
        evidence_path,
        output_path=candidate_path,
    )
    candidate_path.write_text(candidate_text)
    metadata_path.write_text(json.dumps(metadata))
    review_path.write_text(
        json.dumps(build_candidate_review(prompt_path, candidate_path, metadata_path))
    )
    runtime_plan_path.write_text(
        json.dumps(
            build_candidate_runtime_plan(
                review_path,
                environment="test",
                focus=DiagnosticFocus.cost,
                trace_output_dir=tmp_path / "candidate-runtime",
            )
        )
    )
    preflight_path.write_text(
        json.dumps(build_candidate_preflight(prompt_path, review_path, runtime_plan_path))
    )
    manifest = build_candidate_evaluation(evidence_path, sanitized_path, output_dir)

    status = build_candidate_loop_status(
        evidence_pack_path=evidence_path,
        candidate_review_path=review_path,
        runtime_plan_path=runtime_plan_path,
        preflight_path=preflight_path,
        evaluation_manifest_path=Path(manifest["artifacts"]["manifest"]),
    )

    assert status["state"] == "ready_for_approved_candidate_push"
    assert status["external_approval_required"] is True
    assert status["local_artifact_chain_complete"] is False
    assert [stage["status"] for stage in status["stages"]] == [
        "passed",
        "passed",
        "passed",
        "passed",
        "missing",
        "passed",
    ]


def test_build_candidate_loop_status_reports_candidate_evaluated(
    tmp_path: Path,
) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    candidate_prompt_path = tmp_path / "candidate.txt"
    metadata_path = tmp_path / "candidate.metadata.json"
    review_path = tmp_path / "candidate-review.json"
    runtime_plan_path = tmp_path / "runtime-plan.json"
    preflight_path = tmp_path / "preflight.json"
    runtime_check_path = tmp_path / "runtime-check.json"
    output_dir = tmp_path / "candidate-evaluation"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )
    candidate_text, metadata = build_candidate_prompt_artifact(
        prompt_path,
        evidence_path,
        output_path=candidate_prompt_path,
    )
    candidate_prompt_path.write_text(candidate_text)
    metadata_path.write_text(json.dumps(metadata))
    review_path.write_text(
        json.dumps(build_candidate_review(prompt_path, candidate_prompt_path, metadata_path))
    )
    runtime_plan_path.write_text(
        json.dumps(
            build_candidate_runtime_plan(
                review_path,
                environment="test",
                focus=DiagnosticFocus.cost,
                trace_output_dir=tmp_path / "candidate-runtime",
            )
        )
    )
    preflight_path.write_text(
        json.dumps(build_candidate_preflight(prompt_path, review_path, runtime_plan_path))
    )
    created_path = _write_created_prompt_fixture(tmp_path, candidate_body=candidate_text)
    runtime_check_path.write_text(
        json.dumps(
            build_candidate_runtime_check(
                created_path,
                review_path,
                runtime_plan_path,
                preflight_path=preflight_path,
            )
        )
    )
    manifest = build_candidate_evaluation(evidence_path, sanitized_path, output_dir)

    status = build_candidate_loop_status(
        evidence_pack_path=evidence_path,
        candidate_review_path=review_path,
        runtime_plan_path=runtime_plan_path,
        preflight_path=preflight_path,
        runtime_check_path=runtime_check_path,
        evaluation_manifest_path=Path(manifest["artifacts"]["manifest"]),
    )

    assert status["state"] == "candidate_evaluated"
    assert status["decision"]["decision"] == "iterate_candidate"
    assert status["external_approval_required"] is False
    assert status["local_artifact_chain_complete"] is True
    assert status["loop_complete"] is False
    assert status["scope"]["focus"] == "cost"
    assert status["scope"]["executor"] == "orchestrator-test"


def test_loop_status_cli_writes_json(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    output_path = tmp_path / "loop-status.json"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.routing,
                executor="orchestrator-test",
            )
        )
    )

    result = CliRunner().invoke(
        app,
        [
            "loop-status",
            "--evidence-pack",
            str(evidence_path),
            "--output",
            str(output_path),
            "--format",
            ExperimentPlanFormat.json.value,
        ],
    )

    assert result.exit_code == 0
    assert "state=baseline_ready" in result.output
    payload = json.loads(output_path.read_text())
    assert payload["state"] == "baseline_ready"
    assert payload["stages"][0]["status"] == "passed"
    assert payload["stages"][1]["status"] == "missing"
    assert "site down" not in output_path.read_text()


def test_candidate_prompt_file_rejects_prompt_surface_mismatch(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    prompt_path = _write_prompt_fixture(tmp_path)
    prompt_payload = json.loads(prompt_path.read_text())
    prompt_payload["body"]["name"] = "c1/airo-care/orchestrator/orchestrator"
    prompt_path.write_text(json.dumps(prompt_payload))
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )

    result = CliRunner().invoke(
        app,
        [
            "candidate-prompt-file",
            str(prompt_path),
            str(evidence_path),
            str(tmp_path / "candidate.txt"),
        ],
    )

    assert result.exit_code == 1
    assert "Prompt export does not match evidence-pack surface" in result.stderr


def test_candidate_handoff_cli_writes_markdown_and_json(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    markdown_path = tmp_path / "candidate.md"
    json_path = tmp_path / "candidate.json"
    sanitize_halo_jsonl(input_path, sanitized_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.routing,
                executor="orchestrator-test",
            )
        )
    )

    markdown_result = CliRunner().invoke(
        app,
        ["candidate-handoff", str(evidence_path), str(markdown_path)],
    )
    json_result = CliRunner().invoke(
        app,
        [
            "candidate-handoff",
            str(evidence_path),
            str(json_path),
            "--format",
            ExperimentPlanFormat.json.value,
        ],
    )

    assert markdown_result.exit_code == 0
    assert json_result.exit_code == 0
    assert markdown_path.read_text().startswith("# HALO Candidate Handoff")
    payload = json.loads(json_path.read_text())
    assert payload["scope"]["focus"] == "routing"
    assert payload["proposed_candidate"]["prompt_name"] == (
        "c1/airo-care/orchestrator-test/orchestrator"
    )


def test_candidate_handoff_cli_accepts_prompt_snapshot_override(tmp_path: Path) -> None:
    input_path = _write_trace_fixture(tmp_path)
    sanitized_path = tmp_path / "sanitized.jsonl"
    evidence_path = tmp_path / "evidence.json"
    markdown_path = tmp_path / "candidate.md"
    prompt_path = _write_prompt_fixture(tmp_path)
    snapshot_path = tmp_path / "prompt.snapshot.json"
    sanitize_halo_jsonl(input_path, sanitized_path)
    snapshot_path.write_text(json.dumps(build_prompt_snapshot(prompt_path)))
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                sanitized_path,
                focus=DiagnosticFocus.routing,
                executor="orchestrator-test",
            )
        )
    )

    result = CliRunner().invoke(
        app,
        [
            "candidate-handoff",
            str(evidence_path),
            str(markdown_path),
            "--prompt-snapshot",
            str(snapshot_path),
        ],
    )

    assert result.exit_code == 0
    markdown = markdown_path.read_text()
    assert "## Prompt Snapshot" in markdown
    assert "secret routing prompt" not in markdown


def test_audit_halo_jsonl_flags_missing_planned_tool_and_transfer_heuristic(
    tmp_path: Path,
) -> None:
    payload = _langfuse_bundle()
    payload["observations"][1]["output"] = [
        {"type": "function_call", "name": "domain_and_dns_tool", "call_id": "call-2"}
    ]
    payload["observations"][1]["usageDetails"] = {
        "input": 40_001,
        "output": 3,
        "total": 40_004,
    }
    payload["observations"][2]["name"] = "tool_call: transfer_to_human"
    payload["observations"][2]["metadata"]["toolName"] = "transfer_to_human"

    trace_path = tmp_path / "care-ai-traces.jsonl"
    write_halo_jsonl(convert_langfuse_export_to_spans(payload), trace_path)

    report = audit_halo_jsonl(trace_path)

    assert report["counts"]["planned_missing_tool_calls"] == 1
    assert report["counts"]["transfer_without_diagnostic_tool"] == 1
    assert report["counts"]["high_token_llm_spans"] == 1
    assert report["planned_missing_tool_calls"][0]["planned_tool_name"] == ("domain_and_dns_tool")
    assert report["transfer_without_diagnostic_tool"][0]["executed_tool_names"] == [
        "transfer_to_human"
    ]
    assert report["high_token_llm_spans"][0]["input_tokens"] == 40_001


def test_audit_halo_jsonl_flags_repeated_tool_call_fingerprint(tmp_path: Path) -> None:
    payload = _langfuse_bundle()
    duplicate = dict(payload["observations"][2])
    duplicate["id"] = "tool-2"
    duplicate["metadata"] = dict(duplicate["metadata"])
    duplicate["metadata"]["toolCallId"] = "call-2"
    payload["observations"].append(duplicate)

    trace_path = tmp_path / "care-ai-traces.jsonl"
    write_halo_jsonl(convert_langfuse_export_to_spans(payload), trace_path)

    report = audit_halo_jsonl(trace_path)

    assert report["counts"]["repeated_tool_calls"] == 1
    repeated = report["repeated_tool_calls"][0]
    assert repeated["tool_name"] == "http_probe"
    assert repeated["count"] == 2
    assert repeated["input_bytes"] > 0
    assert "input.value" not in json.dumps(repeated)


def test_audit_counts_are_not_truncated_by_top_n(tmp_path: Path) -> None:
    trace_path = _write_trace_fixture(tmp_path)
    write_halo_jsonl(
        convert_langfuse_export_to_spans(_langfuse_bundle()),
        trace_path,
        append=True,
    )

    report = audit_halo_jsonl(trace_path, high_input_tokens=1, top_n=1)

    assert report["counts"]["high_token_llm_spans"] == 2
    assert len(report["high_token_llm_spans"]) == 1


def test_audit_cli_json_does_not_print_raw_io(tmp_path: Path) -> None:
    trace_path = _write_trace_fixture(tmp_path)
    result = CliRunner().invoke(app, ["audit", str(trace_path), "--json"])

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["total_traces"] == 1
    assert "site down" not in result.output
    assert "https://example.test" not in result.output


def test_compare_audits_reports_signal_and_token_delta(tmp_path: Path) -> None:
    baseline_payload = _langfuse_bundle()
    baseline_payload["observations"][1]["output"] = [
        {"type": "function_call", "name": "domain_and_dns_tool", "call_id": "call-2"}
    ]
    baseline_payload["observations"][1]["usageDetails"] = {
        "input": 40_001,
        "output": 3,
        "total": 40_004,
    }
    baseline_payload["observations"][2]["name"] = "tool_call: transfer_to_human"
    baseline_payload["observations"][2]["metadata"]["toolName"] = "transfer_to_human"

    candidate_payload = _langfuse_bundle()
    candidate_payload["observations"][1]["usageDetails"] = {
        "input": 10_000,
        "output": 3,
        "total": 10_003,
    }

    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    write_halo_jsonl(convert_langfuse_export_to_spans(baseline_payload), baseline_path)
    write_halo_jsonl(convert_langfuse_export_to_spans(candidate_payload), candidate_path)

    report = compare_audits(baseline_path, candidate_path)

    assert report["signal_deltas"]["planned_missing_tool_calls"]["direction"] == "improved"
    assert report["signal_deltas"]["transfer_without_diagnostic_tool"]["delta"] == -1
    assert report["signal_deltas"]["high_token_llm_spans"]["direction"] == "improved"
    assert report["token_totals"]["input"]["delta"] == -30_001
    assert report["token_distributions"]["input"]["delta"]["p95"] == -30_001
    assert report["token_distributions"]["input"]["direction_by_p95"] == "improved"
    assert report["session_token_distributions"]["input"]["delta"]["p95"] == -30_001
    assert report["session_token_distributions"]["input"]["direction_by_p95"] == "improved"


def test_compare_audits_cli_json_does_not_print_raw_io(tmp_path: Path) -> None:
    baseline_path = _write_trace_fixture(tmp_path)
    candidate_path = tmp_path / "candidate.jsonl"
    write_halo_jsonl(convert_langfuse_export_to_spans(_langfuse_bundle()), candidate_path)

    result = CliRunner().invoke(
        app,
        ["compare-audits", str(baseline_path), str(candidate_path), "--json"],
    )

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["trace_counts"]["baseline"] == 1
    assert report["token_distributions"]["input"]["baseline"]["median"] == 11
    assert report["session_token_distributions"]["input"]["baseline"]["median"] == 11
    assert "site down" not in result.output
    assert "https://example.test" not in result.output


def test_build_candidate_evaluation_writes_compare_evidence_decision_bundle(
    tmp_path: Path,
) -> None:
    baseline_payload = _langfuse_bundle()
    baseline_payload["observations"][1]["usageDetails"] = {
        "input": 40_001,
        "output": 3,
        "total": 40_004,
    }
    candidate_payload = _langfuse_bundle()
    candidate_payload["observations"][1]["usageDetails"] = {
        "input": 10_000,
        "output": 3,
        "total": 10_003,
    }
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    evidence_path = tmp_path / "baseline-evidence.json"
    runtime_plan_path = tmp_path / "runtime-plan.json"
    output_dir = tmp_path / "candidate-evaluation"
    write_halo_jsonl(convert_langfuse_export_to_spans(baseline_payload), baseline_path)
    write_halo_jsonl(convert_langfuse_export_to_spans(candidate_payload), candidate_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                baseline_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )
    runtime_plan_path.write_text(
        json.dumps(
            {
                "rollback_commands": ["lf api prompts create --labels latest"],
                "comparison_commands": ["uv run halo-careai candidate-evaluate evidence candidate"],
            }
        )
    )

    manifest = build_candidate_evaluation(
        evidence_path,
        candidate_path,
        output_dir,
        runtime_plan_path=runtime_plan_path,
        candidate_traffic_note=(
            "direct visitor API traffic; runtime cost proof, not authenticated shopper UI parity"
        ),
    )

    assert manifest["purpose"] == "care-ai-halo-candidate-evaluation"
    assert manifest["scope"]["focus"] == "cost"
    assert manifest["scope"]["executor"] == "orchestrator-test"
    assert manifest["decision"]["decision"] == "promote_candidate"
    assert manifest["decision"]["runtime_boundary"]["production_label_allowed"] is False
    assert (
        manifest["decision"]["candidate_trace_profile"]["traffic_note"]
        == "direct visitor API traffic; runtime cost proof, not authenticated shopper UI parity"
    )
    assert manifest["decision"]["candidate_trace_profile"]["token_bearing_llm_span_count"] == 1
    assert (
        manifest["comparison_summary"]["session_token_distributions"]["input"]["direction_by_p95"]
        == "improved"
    )
    assert Path(manifest["artifacts"]["comparison"]).exists()
    assert Path(manifest["artifacts"]["candidate_evidence_pack"]).exists()
    assert Path(manifest["artifacts"]["decision_json"]).exists()
    assert Path(manifest["artifacts"]["decision_markdown"]).exists()
    assert Path(manifest["artifacts"]["manifest"]).exists()
    manifest_text = json.dumps(manifest)
    assert "site down" not in manifest_text
    assert "https://example.test" not in manifest_text


def test_build_candidate_evaluation_records_runtime_prompt_coverage(
    tmp_path: Path,
) -> None:
    baseline_payload = _langfuse_bundle()
    baseline_payload["observations"][1]["usageDetails"] = {
        "input": 40_001,
        "output": 3,
        "total": 40_004,
    }
    candidate_payload = _langfuse_bundle()
    candidate_payload["observations"][1]["usageDetails"] = {
        "input": 10_000,
        "output": 3,
        "total": 10_003,
    }
    candidate_payload["observations"][1]["metadata"].update(
        {
            "promptName": "c1/airo-care/orchestrator-test/orchestrator",
            "promptVersion": 43,
        }
    )
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    evidence_path = tmp_path / "baseline-evidence.json"
    runtime_check_path = _write_runtime_check_fixture(tmp_path, version=43)
    output_dir = tmp_path / "candidate-evaluation"
    write_halo_jsonl(convert_langfuse_export_to_spans(baseline_payload), baseline_path)
    write_halo_jsonl(convert_langfuse_export_to_spans(candidate_payload), candidate_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                baseline_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )

    manifest = build_candidate_evaluation(
        evidence_path,
        candidate_path,
        output_dir,
        runtime_check_path=runtime_check_path,
    )

    assert manifest["runtime_prompt_coverage"]["passed"] is True
    assert manifest["runtime_prompt_coverage"]["expected_prompt_version"] == 43
    assert manifest["runtime_prompt_coverage"]["target_prompt_span_count"] == 1
    assert manifest["source_artifacts"]["runtime_check"] == str(runtime_check_path)


def test_build_candidate_evaluation_accepts_timestamp_proof_when_prompt_version_missing(
    tmp_path: Path,
) -> None:
    baseline_payload = _langfuse_bundle()
    candidate_payload = _langfuse_bundle()
    candidate_payload["trace"]["timestamp"] = "2026-05-22T04:55:00.000Z"
    candidate_payload["trace"]["updatedAt"] = "2026-05-22T04:55:04.000Z"
    for observation in candidate_payload["observations"]:
        observation["startTime"] = "2026-05-22T04:55:01.000Z"
        observation["endTime"] = "2026-05-22T04:55:02.000Z"
    candidate_payload["observations"][1]["metadata"].update(
        {
            "promptName": "c1/airo-care/orchestrator-test/orchestrator",
        }
    )
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    evidence_path = tmp_path / "baseline-evidence.json"
    runtime_check_path = _write_runtime_check_fixture(tmp_path, version=43)
    output_dir = tmp_path / "candidate-evaluation"
    write_halo_jsonl(convert_langfuse_export_to_spans(baseline_payload), baseline_path)
    write_halo_jsonl(convert_langfuse_export_to_spans(candidate_payload), candidate_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                baseline_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )

    manifest = build_candidate_evaluation(
        evidence_path,
        candidate_path,
        output_dir,
        runtime_check_path=runtime_check_path,
    )

    assert manifest["runtime_prompt_coverage"]["passed"] is True
    assert (
        manifest["runtime_prompt_coverage"]["proof_method"] == "trace_timestamp_after_runtime_push"
    )
    assert manifest["runtime_prompt_coverage"]["missing_prompt_version_span_count"] == 1


def test_build_candidate_evaluation_rejects_stale_runtime_prompt_version(
    tmp_path: Path,
) -> None:
    baseline_payload = _langfuse_bundle()
    candidate_payload = _langfuse_bundle()
    candidate_payload["observations"][1]["metadata"].update(
        {
            "promptName": "c1/airo-care/orchestrator-test/orchestrator",
            "promptVersion": 42,
        }
    )
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    evidence_path = tmp_path / "baseline-evidence.json"
    runtime_check_path = _write_runtime_check_fixture(tmp_path, version=43)
    write_halo_jsonl(convert_langfuse_export_to_spans(baseline_payload), baseline_path)
    write_halo_jsonl(convert_langfuse_export_to_spans(candidate_payload), candidate_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                baseline_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )

    with pytest.raises(ValueError, match="do not prove the pushed runtime prompt version"):
        build_candidate_evaluation(
            evidence_path,
            candidate_path,
            tmp_path / "candidate-evaluation",
            runtime_check_path=runtime_check_path,
        )


def test_candidate_evaluate_cli_writes_post_candidate_artifacts(tmp_path: Path) -> None:
    baseline_payload = _langfuse_bundle()
    baseline_payload["observations"][1]["usageDetails"] = {
        "input": 40_001,
        "output": 3,
        "total": 40_004,
    }
    candidate_payload = _langfuse_bundle()
    candidate_payload["observations"][1]["usageDetails"] = {
        "input": 10_000,
        "output": 3,
        "total": 10_003,
    }
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    evidence_path = tmp_path / "baseline-evidence.json"
    output_dir = tmp_path / "candidate-evaluation"
    write_halo_jsonl(convert_langfuse_export_to_spans(baseline_payload), baseline_path)
    write_halo_jsonl(convert_langfuse_export_to_spans(candidate_payload), candidate_path)
    evidence_path.write_text(
        json.dumps(
            build_evidence_pack(
                baseline_path,
                focus=DiagnosticFocus.cost,
                executor="orchestrator-test",
            )
        )
    )

    result = CliRunner().invoke(
        app,
        [
            "candidate-evaluate",
            str(evidence_path),
            str(candidate_path),
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert "decision=promote_candidate" in result.output
    manifest_path = output_dir / "orchestrator-test-cost-candidate-evaluation-manifest.json"
    decision_path = output_dir / "orchestrator-test-cost-candidate-decision.md"
    assert manifest_path.exists()
    assert decision_path.exists()
    assert "promote_candidate" in decision_path.read_text()
    assert "site down" not in manifest_path.read_text()


def test_build_candidate_decision_promotes_cost_candidate(tmp_path: Path) -> None:
    baseline_payload = _langfuse_bundle()
    baseline_payload["observations"][1]["usageDetails"] = {
        "input": 40_001,
        "output": 3,
        "total": 40_004,
    }
    candidate_payload = _langfuse_bundle()
    candidate_payload["observations"][1]["usageDetails"] = {
        "input": 10_000,
        "output": 3,
        "total": 10_003,
    }
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    comparison_path = tmp_path / "comparison.json"
    runtime_plan_path = tmp_path / "runtime-plan.json"
    write_halo_jsonl(convert_langfuse_export_to_spans(baseline_payload), baseline_path)
    write_halo_jsonl(convert_langfuse_export_to_spans(candidate_payload), candidate_path)
    comparison_path.write_text(json.dumps(compare_audits(baseline_path, candidate_path)))
    runtime_plan_path.write_text(
        json.dumps(
            {
                "rollback_commands": ["lf api prompts create --labels latest"],
                "comparison_commands": ["uv run halo-careai compare-audits baseline candidate"],
            }
        )
    )

    decision = build_candidate_decision(
        comparison_path,
        runtime_plan_path=runtime_plan_path,
        focus=DiagnosticFocus.cost,
        candidate_traffic_note=(
            "direct visitor API traffic; runtime cost proof, not authenticated shopper UI parity"
        ),
    )
    markdown = render_candidate_decision_markdown(decision)

    assert decision["decision"] == "promote_candidate"
    assert decision["approval_required"] is True
    assert decision["runtime_boundary"]["production_label_allowed"] is False
    assert decision["trace_coverage"]["passed"] is True
    assert decision["target_assessment"]["passed"] is True
    assert decision["target_assessment"]["observed"]["session_input_direction_by_p95"] == (
        "improved"
    )
    assert decision["candidate_trace_profile"]["token_bearing_llm_span_count"] == 1
    assert (
        decision["candidate_trace_profile"]["traffic_note"]
        == "direct visitor API traffic; runtime cost proof, not authenticated shopper UI parity"
    )
    assert decision["regressions"] == []
    assert "promote_candidate" in markdown
    assert "Production label allowed: `False`" in markdown
    assert "Session input p95 direction: `improved`" in markdown
    assert "direct visitor API traffic" in markdown


def test_build_candidate_decision_collects_more_traces_for_zero_llm_candidate(
    tmp_path: Path,
) -> None:
    baseline_payload = _langfuse_bundle()
    baseline_payload["observations"][1]["usageDetails"] = {
        "input": 40_001,
        "output": 3,
        "total": 40_004,
    }
    candidate_payload = _langfuse_bundle()
    candidate_payload["observations"] = []
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    comparison_path = tmp_path / "comparison.json"
    write_halo_jsonl(convert_langfuse_export_to_spans(baseline_payload), baseline_path)
    write_halo_jsonl(convert_langfuse_export_to_spans(candidate_payload), candidate_path)
    comparison_path.write_text(json.dumps(compare_audits(baseline_path, candidate_path)))

    decision = build_candidate_decision(comparison_path, focus=DiagnosticFocus.cost)
    markdown = render_candidate_decision_markdown(decision)

    assert decision["decision"] == "collect_more_traces"
    assert decision["approval_required"] is False
    assert decision["evidence_gate"]["passed"] is False
    assert decision["target_assessment"]["passed"] is True
    assert "Evidence gate passed: `False`" in markdown
    assert "Token-bearing LLM spans: `0`" in markdown


def test_build_candidate_decision_rolls_back_guardrail_regression(tmp_path: Path) -> None:
    baseline_payload = _langfuse_bundle()
    baseline_payload["observations"][1]["usageDetails"] = {
        "input": 40_001,
        "output": 3,
        "total": 40_004,
    }
    candidate_payload = _langfuse_bundle()
    candidate_payload["observations"][1]["usageDetails"] = {
        "input": 10_000,
        "output": 3,
        "total": 10_003,
    }
    candidate_payload["observations"][1]["output"] = [
        {"type": "function_call", "name": "domain_and_dns_tool", "call_id": "call-2"}
    ]
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    comparison_path = tmp_path / "comparison.json"
    write_halo_jsonl(convert_langfuse_export_to_spans(baseline_payload), baseline_path)
    write_halo_jsonl(convert_langfuse_export_to_spans(candidate_payload), candidate_path)
    comparison_path.write_text(json.dumps(compare_audits(baseline_path, candidate_path)))

    decision = build_candidate_decision(comparison_path, focus=DiagnosticFocus.cost)

    assert decision["decision"] == "rollback_candidate"
    assert decision["approval_required"] is True
    assert decision["regressions"][0]["signal"] == "planned_missing_tool_calls"


def test_candidate_decision_cli_writes_markdown_and_json(tmp_path: Path) -> None:
    baseline_payload = _langfuse_bundle()
    baseline_payload["observations"][1]["usageDetails"] = {
        "input": 40_001,
        "output": 3,
        "total": 40_004,
    }
    candidate_payload = _langfuse_bundle()
    candidate_payload["observations"][1]["usageDetails"] = {
        "input": 10_000,
        "output": 3,
        "total": 10_003,
    }
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    comparison_path = tmp_path / "comparison.json"
    markdown_path = tmp_path / "decision.md"
    json_path = tmp_path / "decision.json"
    write_halo_jsonl(convert_langfuse_export_to_spans(baseline_payload), baseline_path)
    write_halo_jsonl(convert_langfuse_export_to_spans(candidate_payload), candidate_path)
    comparison_path.write_text(json.dumps(compare_audits(baseline_path, candidate_path)))

    markdown_result = CliRunner().invoke(
        app,
        ["candidate-decision", str(comparison_path), str(markdown_path)],
    )
    json_result = CliRunner().invoke(
        app,
        [
            "candidate-decision",
            str(comparison_path),
            str(json_path),
            "--format",
            ExperimentPlanFormat.json.value,
        ],
    )

    assert markdown_result.exit_code == 0
    assert json_result.exit_code == 0
    assert "decision=promote_candidate" in markdown_result.output
    assert "HALO Candidate Decision" in markdown_path.read_text()
    payload = json.loads(json_path.read_text())
    assert payload["decision"] == "promote_candidate"
    assert "site down" not in markdown_path.read_text()


def test_convert_lf_pair_writes_halo_jsonl(tmp_path: Path) -> None:
    trace_path, observations_path = _write_raw_lf_pair(tmp_path)
    output_path = tmp_path / "out.jsonl"

    result = CliRunner().invoke(
        app,
        ["convert-lf-pair", str(trace_path), str(observations_path), str(output_path)],
    )

    assert result.exit_code == 0
    assert "Wrote 4 spans" in result.output
    summary = inspect_halo_jsonl(output_path)
    assert summary["total_traces"] == 1
    assert summary["tool_names"] == [{"name": "http_probe", "count": 1}]


def test_convert_lf_pair_falls_back_to_trace_embedded_observations(tmp_path: Path) -> None:
    bundle = _langfuse_bundle()
    trace_path = tmp_path / "trace.json"
    observations_path = tmp_path / "empty-observations.json"
    output_path = tmp_path / "out.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "ok": True,
                "status": 200,
                "body": {
                    **bundle["trace"],
                    "observations": bundle["observations"],
                },
            }
        )
    )
    observations_path.write_text(json.dumps({"ok": True, "status": 200, "body": {"data": []}}))

    result = CliRunner().invoke(
        app,
        ["convert-lf-pair", str(trace_path), str(observations_path), str(output_path)],
    )

    assert result.exit_code == 0
    assert "Wrote 4 spans" in result.output
    summary = inspect_halo_jsonl(output_path)
    assert summary["llm_input_token_distribution"]["count"] == 1
    assert summary["tool_names"] == [{"name": "http_probe", "count": 1}]


def test_convert_lf_pair_can_append_for_session_loops(tmp_path: Path) -> None:
    trace_path, observations_path = _write_raw_lf_pair(tmp_path)
    output_path = tmp_path / "out.jsonl"

    first = CliRunner().invoke(
        app,
        ["convert-lf-pair", str(trace_path), str(observations_path), str(output_path)],
    )
    second = CliRunner().invoke(
        app,
        [
            "convert-lf-pair",
            str(trace_path),
            str(observations_path),
            str(output_path),
            "--append",
        ],
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert output_path.read_text().count("\n") == 8


def test_build_lf_batch_recipe_uses_lf_and_local_halo_commands(tmp_path: Path) -> None:
    recipe = build_lf_batch_recipe(
        tmp_path / "batch dir",
        environment="test",
        executor="orchestrator-test",
        limit=25,
        sample=12,
        focus=DiagnosticFocus.cost,
        model="gpt-5.4-mini",
        use_gocode=True,
    )

    assert "lf api traces list" in recipe
    assert '--tags "executor:$EXECUTOR"' in recipe
    assert '--fields "core,metrics"' in recipe
    assert "--fields core,basic,time,io,metadata,model,usage,prompt,metrics" in recipe
    assert "--expand-metadata traceKind,toolName,toolCallId" in recipe
    assert 'jq -r --arg executor "$EXECUTOR"' in recipe
    assert "uv run halo-careai convert-lf-pair" in recipe
    assert "uv run halo-careai sanitize" in recipe
    assert "uv run halo-careai evidence-pack" in recipe
    assert "uv run halo-careai diagnose" in recipe
    assert "--gocode" in recipe
    assert "FOCUS=cost" in recipe
    assert "OUT_DIR=" in recipe
    assert "python3 -c" not in recipe


def test_lf_batch_recipe_cli_prints_without_calling_langfuse(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "lf-batch-recipe",
            str(tmp_path / "export"),
            "--environment",
            "test",
            "--executor",
            "orchestrator-test",
            "--sample",
            "3",
            "--no-gocode",
        ],
    )

    assert result.exit_code == 0
    assert "lf api traces list" in result.output
    assert "SAMPLE=3" in result.output
    assert "--gocode" not in result.output
    assert "uv run halo-careai diagnose" in result.output
