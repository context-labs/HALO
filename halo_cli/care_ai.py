"""Care AI helper CLI for preparing Langfuse traces for HALO."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr
from datetime import datetime
from enum import Enum
from hashlib import sha256
from io import StringIO
from math import ceil
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from engine.sandbox.sandbox import Sandbox
from engine.traces.care_ai_langfuse import (
    convert_langfuse_export_to_spans,
    load_langfuse_export,
    write_halo_jsonl,
)
from engine.traces.models.canonical_span import SpanRecord
from halo_cli.engine_runner import REASONING_EFFORT_CHOICES, console, run_trace

app = typer.Typer(no_args_is_help=True)


class DiagnosticFocus(str, Enum):
    overview = "overview"
    tool_errors = "tool-errors"
    routing = "routing"
    loops = "loops"
    cost = "cost"
    mcp = "mcp"


class TraceToolMode(str, Enum):
    summary = "summary"
    full = "full"


class ExperimentPlanFormat(str, Enum):
    markdown = "markdown"
    json = "json"


_PROD_ORCHESTRATOR_TOOLS = [
    "transfer_to_human",
    "domain_and_dns_tool",
    "domain_lifecycle_tool",
    "http_probe",
]
_VARIANT_SPECIALIST_TOOLS = [
    "general_support_tool",
    "hosting_tool",
    "billing_tool",
    "email_diagnostic_tool",
]
_HELP_CENTER_VARIANT_TOOLS = [
    "refund_agent",
    "commerce_agent",
    "ssl_agent",
    "websites_agent",
]
_VARIANT_EXECUTORS = {
    "orchestrator-test",
    "orchestrator-test-email-mcp-bot",
    "orchestrator-test-tcf",
    "orchestrator-sandbox",
}
_HELP_CENTER_VARIANT_EXECUTORS = {
    "orchestrator-test-tcf",
    "orchestrator-sandbox",
}
_CARE_AI_EXECUTOR_SURFACES: dict[str, dict[str, str]] = {
    "airo-care-orchestrator": {
        "prompt_name": "c1/airo-care/orchestrator/orchestrator",
        "session_executor_path": (
            "care-ai-agents/packages/service/src/agents/session-executors/"
            "airo-care-orchestrator/index.ts"
        ),
        "route": "/v1/airo-care-orchestrator",
        "notes": (
            "Production orchestrator. Do not map test-variant findings here unless "
            "the trace executor is airo-care-orchestrator."
        ),
    },
    "orchestrator-test": {
        "prompt_name": "c1/airo-care/orchestrator-test/orchestrator",
        "session_executor_path": (
            "care-ai-agents/packages/service/src/agents/session-executors/"
            "orchestrator-test/index.ts"
        ),
        "route": "/v1/orchestrator-test",
        "notes": (
            "Temporary test variant with time and customer runtime context enabled "
            "and additional specialist tools."
        ),
    },
    "orchestrator-test-email-mcp-bot": {
        "prompt_name": "c1/airo-care/orchestrator-test-email-mcp-bot/orchestrator",
        "session_executor_path": (
            "care-ai-agents/packages/service/src/agents/session-executors/"
            "orchestrator-test-email-mcp-bot/index.ts"
        ),
        "route": "/v1/orchestrator-test-email-mcp-bot",
        "notes": "Temporary test variant for email MCP bot coverage.",
    },
    "orchestrator-test-tcf": {
        "prompt_name": "c1/airo-care/orchestrator-test-tcf/orchestrator",
        "session_executor_path": (
            "care-ai-agents/packages/service/src/agents/session-executors/"
            "orchestrator-test-tcf/index.ts"
        ),
        "route": "/v1/orchestrator-test-tcf",
        "notes": "Temporary TCF test variant with help-center agents-as-tools.",
    },
    "orchestrator-sandbox": {
        "prompt_name": "c1/airo-care/orchestrator-sandbox/orchestrator",
        "session_executor_path": (
            "care-ai-agents/packages/service/src/agents/session-executors/"
            "orchestrator-sandbox/index.ts"
        ),
        "route": "/v1/orchestrator-sandbox",
        "notes": "Sandbox orchestrator variant with help-center agents-as-tools.",
    },
}
_SENSITIVE_ATTRIBUTE_KEYS = (
    "input.value",
    "output.value",
    "llm.input_messages",
    "llm.output_messages",
)
_SENSITIVE_IDENTIFIER_ATTRIBUTE_KEYS = (
    "langfuse.trace_id",
    "langfuse.session_id",
    "langfuse.observation_id",
    "care_ai.conversation_id",
    "care_ai.ucid",
    "care_ai.request_id",
    "care_ai.customer_id",
    "care_ai.visitor_id",
    "care_ai.tool_call_id",
    "llm.response.id",
    "inference.user_id",
    "user.id",
)
_GOCODE_OPENAI_BASE_URL = "https://caas-gocode-prod.caas-prod.prod.onkatana.net/v1"
_LF_OBSERVATION_FIELDS = "core,basic,time,io,metadata,model,usage,prompt,metrics"
_LF_EXPAND_METADATA = (
    "traceKind,toolName,toolCallId,parentAgent,targetAgent,targetAgentDisplayName,"
    "conversationId,ucid,sessionExecutorName,conversationModel,requestType,requestId,"
    "promptName,promptVersion,status,agentName,agentDisplayName,userType,path"
)
_DEFAULT_PIPELINE_DIR = Path(".halo-careai/runs")


@app.callback()
def main() -> None:
    """Prepare Care AI trace exports for HALO."""


@app.command("convert-langfuse")
def convert_langfuse(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="JSON or JSONL export from Langfuse.",
    ),
    output_path: Path = typer.Argument(
        ...,
        dir_okay=False,
        writable=True,
        help="Output HALO-compatible JSONL trace path.",
    ),
    project_id: str = typer.Option(
        "care-ai",
        "--project-id",
        help="Value written to inference.project_id for HALO filtering.",
    ),
    service_name: str = typer.Option(
        "care-ai-agents",
        "--service-name",
        help='Value written to resource.attributes."service.name".',
    ),
) -> None:
    """Convert Care AI Langfuse trace exports to HALO's OTel-shaped JSONL."""
    payload = load_langfuse_export(input_path)
    spans = convert_langfuse_export_to_spans(
        payload,
        project_id=project_id,
        service_name=service_name,
    )
    count = write_halo_jsonl(spans, output_path)
    typer.echo(f"Wrote {count} spans to {output_path}")


@app.command("convert-lf-pair")
def convert_lf_pair(
    trace_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Raw `lf api traces get ... --json` response.",
    ),
    observations_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Raw `lf api observations list ... --json` response.",
    ),
    output_path: Path = typer.Argument(
        ...,
        dir_okay=False,
        writable=True,
        help="Output HALO-compatible JSONL trace path.",
    ),
    project_id: str = typer.Option(
        "care-ai",
        "--project-id",
        help="Value written to inference.project_id for HALO filtering.",
    ),
    service_name: str = typer.Option(
        "care-ai-agents",
        "--service-name",
        help='Value written to resource.attributes."service.name".',
    ),
    append: bool = typer.Option(
        False,
        "--append/--overwrite",
        help="Append spans to OUTPUT_PATH instead of overwriting it.",
    ),
) -> None:
    """Convert raw local `lf` trace + observation JSON files to HALO JSONL."""
    trace_payload = load_langfuse_export(trace_path)
    observations_payload = load_langfuse_export(observations_path)
    payload = {
        "trace": trace_payload,
        "observations": _observations_or_embedded_trace_observations(
            trace_payload,
            observations_payload,
        ),
    }
    spans = convert_langfuse_export_to_spans(
        payload,
        project_id=project_id,
        service_name=service_name,
    )
    count = write_halo_jsonl(spans, output_path, append=append)
    mode = "Appended" if append else "Wrote"
    typer.echo(f"{mode} {count} spans to {output_path}")


def _observations_or_embedded_trace_observations(
    trace_payload: Any, observations_payload: Any
) -> Any:
    if _lf_observation_count(observations_payload) > 0:
        return observations_payload
    embedded = _embedded_trace_observations(trace_payload)
    return embedded if embedded else observations_payload


def _embedded_trace_observations(trace_payload: Any) -> list[dict[str, Any]]:
    trace = _unwrap_lf_body(trace_payload)
    observations = trace.get("observations")
    if not isinstance(observations, list):
        return []
    return [item for item in observations if isinstance(item, dict)]


def _lf_observation_count(payload: Any) -> int:
    value = _unwrap_lf_body(payload) if isinstance(payload, dict) else payload
    if isinstance(value, dict):
        if isinstance(value.get("data"), list):
            return len(value["data"])
        if isinstance(value.get("items"), list):
            return len(value["items"])
    if isinstance(value, list):
        return len(value)
    return 0


def build_lf_batch_recipe(
    output_dir: Path,
    *,
    environment: str,
    executor: str,
    limit: int = 25,
    sample: int = 12,
    prefix: str | None = None,
    project_id: str = "care-ai",
    service_name: str = "care-ai-agents",
    focus: DiagnosticFocus = DiagnosticFocus.cost,
    model: str = "gpt-5.4-mini",
    use_gocode: bool = True,
) -> str:
    """Return a copy-paste `lf` recipe without calling Langfuse from Python."""
    safe_prefix = prefix or executor
    diagnose_gocode = " --gocode" if use_gocode else ""
    lines = [
        "# Uses the approved lf CLI directly. This command block does not require a Python Langfuse client.",
        "# convert-lf-pair falls back to trace-embedded observations when observations list is empty.",
        f"OUT_DIR={_shell_quote(str(output_dir))}",
        f"EXECUTOR={_shell_quote(executor)}",
        f"ENVIRONMENT={_shell_quote(environment)}",
        f"LIMIT={limit}",
        f"SAMPLE={sample}",
        f"PREFIX={_shell_quote(safe_prefix)}",
        f"FOCUS={_shell_quote(focus.value)}",
        'mkdir -p "$OUT_DIR/raw" "$OUT_DIR/reports"',
        "",
        "lf api traces list \\",
        '  --environment "$ENVIRONMENT" \\',
        '  --tags "executor:$EXECUTOR" \\',
        '  --fields "core,metrics" \\',
        "  --order-by timestamp.desc \\",
        '  --limit "$LIMIT" \\',
        "  --json \\",
        '  > "$OUT_DIR/raw/traces.json"',
        "",
        ': > "$OUT_DIR/${PREFIX}-traces.jsonl"',
        (
            'jq -r --arg executor "$EXECUTOR" '
            "'(.body.data // .data // .items // .)[] | select(.name == $executor) | .id' "
            '"$OUT_DIR/raw/traces.json" |'
        ),
        'head -n "$SAMPLE" |',
        "while read -r TRACE_ID; do",
        '  lf api traces get "$TRACE_ID" --json \\',
        '    > "$OUT_DIR/raw/${TRACE_ID}.trace.json"',
        "  lf api observations list \\",
        '    --trace-id "$TRACE_ID" \\',
        f"    --fields {_shell_quote(_LF_OBSERVATION_FIELDS)} \\",
        f"    --expand-metadata {_shell_quote(_LF_EXPAND_METADATA)} \\",
        "    --limit 1000 \\",
        "    --json \\",
        '    > "$OUT_DIR/raw/${TRACE_ID}.observations.json"',
        "  uv run halo-careai convert-lf-pair \\",
        '    "$OUT_DIR/raw/${TRACE_ID}.trace.json" \\',
        '    "$OUT_DIR/raw/${TRACE_ID}.observations.json" \\',
        '    "$OUT_DIR/${PREFIX}-traces.jsonl" \\',
        f"    --project-id {_shell_quote(project_id)} \\",
        f"    --service-name {_shell_quote(service_name)} \\",
        "    --append",
        "done",
        "",
        "uv run halo-careai sanitize \\",
        '  "$OUT_DIR/${PREFIX}-traces.jsonl" \\',
        '  "$OUT_DIR/${PREFIX}-traces.sanitized.jsonl"',
        "",
        "uv run halo-careai evidence-pack \\",
        '  "$OUT_DIR/${PREFIX}-traces.sanitized.jsonl" \\',
        '  "$OUT_DIR/${PREFIX}-evidence-pack.json" \\',
        '  --focus "$FOCUS" \\',
        '  --executor "$EXECUTOR"',
        "",
        "uv run halo-careai diagnose \\",
        '  "$OUT_DIR/${PREFIX}-traces.sanitized.jsonl" \\',
        '  --focus "$FOCUS" \\',
        '  --executor "$EXECUTOR" \\',
        "  --tool-mode summary \\",
        "  --reasoning-effort low \\",
        "  --max-depth 0 \\",
        "  --max-turns 4 \\",
        "  --timeout-seconds 60 \\",
        f"  --model {_shell_quote(model)}{diagnose_gocode} \\",
        '  --output "$OUT_DIR/reports/${PREFIX}-${ENVIRONMENT}-${FOCUS}.md" \\',
        '  --events-jsonl "$OUT_DIR/reports/${PREFIX}-${ENVIRONMENT}-${FOCUS}.events.jsonl"',
    ]
    return "\n".join(lines) + "\n"


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


@app.command("lf-batch-recipe")
def lf_batch_recipe(
    output_dir: Path = typer.Argument(
        ...,
        dir_okay=True,
        file_okay=False,
        help="Directory to use in the printed lf recipe.",
    ),
    environment: str = typer.Option(
        "test",
        "--environment",
        help="Langfuse environment passed to `lf api traces list`.",
    ),
    executor: str = typer.Option(
        ...,
        "--executor",
        help="Care AI executor name, also used for the executor tag and trace-name filter.",
    ),
    limit: int = typer.Option(
        25,
        "--limit",
        min=1,
        help="Trace list limit passed to `lf api traces list`.",
    ),
    sample: int = typer.Option(
        12,
        "--sample",
        min=1,
        help="Number of matching trace ids to fetch from the trace list.",
    ),
    prefix: str | None = typer.Option(
        None,
        "--prefix",
        help="Output filename prefix. Defaults to the executor name.",
    ),
    focus: DiagnosticFocus = typer.Option(
        DiagnosticFocus.cost,
        "--focus",
        "-f",
        case_sensitive=False,
        help="Care AI diagnostic focus for evidence-pack and diagnose commands.",
    ),
    model: str = typer.Option("gpt-5.4-mini", "--model", "-m"),
    use_gocode: bool = typer.Option(
        True,
        "--gocode/--no-gocode",
        help="Include --gocode in the printed diagnose command.",
    ),
) -> None:
    """Print an approved lf+jq pull recipe without calling Langfuse."""
    typer.echo(
        build_lf_batch_recipe(
            output_dir,
            environment=environment,
            executor=executor,
            limit=limit,
            sample=sample,
            prefix=prefix,
            focus=focus,
            model=model,
            use_gocode=use_gocode,
        )
    )


@app.command("experiment-plan")
def experiment_plan_command(
    evidence_pack_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Evidence pack JSON produced by `halo-careai evidence-pack`.",
    ),
    output_path: Path = typer.Argument(
        ...,
        dir_okay=False,
        writable=True,
        help="Output experiment plan path.",
    ),
    diagnosis_report_path: Path | None = typer.Option(
        None,
        "--diagnosis-report",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Optional final HALO diagnosis Markdown report.",
    ),
    output_format: ExperimentPlanFormat = typer.Option(
        ExperimentPlanFormat.markdown,
        "--format",
        case_sensitive=False,
        help="Output format.",
    ),
) -> None:
    """Build a deterministic harness experiment plan from HALO artifacts."""
    plan = build_experiment_plan(
        evidence_pack_path,
        diagnosis_report_path=diagnosis_report_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == ExperimentPlanFormat.json:
        output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    else:
        output_path.write_text(render_experiment_plan_markdown(plan))
    typer.echo(f"Wrote experiment plan to {output_path}")


@app.command("candidate-handoff")
def candidate_handoff_command(
    evidence_pack_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Evidence pack JSON produced by `halo-careai evidence-pack`.",
    ),
    output_path: Path = typer.Argument(
        ...,
        dir_okay=False,
        writable=True,
        help="Output candidate handoff path.",
    ),
    diagnosis_report_path: Path | None = typer.Option(
        None,
        "--diagnosis-report",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Optional final HALO diagnosis Markdown report.",
    ),
    prompt_snapshot_path: Path | None = typer.Option(
        None,
        "--prompt-snapshot",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Optional metadata-only prompt snapshot produced by `prompt-snapshot`.",
    ),
    output_format: ExperimentPlanFormat = typer.Option(
        ExperimentPlanFormat.markdown,
        "--format",
        case_sensitive=False,
        help="Output format.",
    ),
) -> None:
    """Build a deterministic prompt/config candidate handoff without mutating Langfuse."""
    handoff = build_candidate_handoff(
        evidence_pack_path,
        diagnosis_report_path=diagnosis_report_path,
        prompt_snapshot_path=prompt_snapshot_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == ExperimentPlanFormat.json:
        output_path.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n")
    else:
        output_path.write_text(render_candidate_handoff_markdown(handoff))
    typer.echo(f"Wrote candidate handoff to {output_path}")


@app.command("prompt-snapshot")
def prompt_snapshot_command(
    prompt_json_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Raw `lf api prompts get ... --json` response.",
    ),
    output_path: Path = typer.Argument(
        ...,
        dir_okay=False,
        writable=True,
        help="Output metadata-only prompt snapshot JSON.",
    ),
) -> None:
    """Fingerprint a Langfuse prompt export without retaining prompt body text."""
    snapshot = build_prompt_snapshot(prompt_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    prompt = snapshot["prompt"]
    body = snapshot["body"]
    typer.echo(
        " | ".join(
            [
                f"wrote={output_path}",
                f"prompt={prompt.get('name', 'unknown')}",
                f"version={prompt.get('version', 'unknown')}",
                f"body_sha256_12={body['sha256_12']}",
                "raw_prompt_retained=False",
            ]
        )
    )


@app.command("candidate-prompt-file")
def candidate_prompt_file_command(
    prompt_json_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Raw `lf api prompts get ... --json` response for the current prompt.",
    ),
    evidence_pack_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Evidence pack JSON produced by `halo-careai evidence-pack`.",
    ),
    output_path: Path = typer.Argument(
        ...,
        dir_okay=False,
        writable=True,
        help="Output candidate prompt body text file.",
    ),
    diagnosis_report_path: Path | None = typer.Option(
        None,
        "--diagnosis-report",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Optional final HALO diagnosis Markdown report.",
    ),
    metadata_output_path: Path | None = typer.Option(
        None,
        "--metadata-output",
        dir_okay=False,
        writable=True,
        help="Optional metadata-only JSON output for the generated candidate prompt file.",
    ),
) -> None:
    """Write a reviewable candidate prompt file without mutating Langfuse."""
    try:
        candidate_text, metadata = build_candidate_prompt_artifact(
            prompt_json_path,
            evidence_pack_path,
            output_path=output_path,
            diagnosis_report_path=diagnosis_report_path,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(candidate_text)
    if metadata_output_path is not None:
        metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    typer.echo(
        " | ".join(
            [
                f"wrote={output_path}",
                f"prompt={metadata['prompt']['name']}",
                f"candidate_sha256_12={metadata['candidate_body']['sha256_12']}",
                "mutated_langfuse=False",
            ]
        )
    )


@app.command("candidate-review")
def candidate_review_command(
    prompt_json_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Raw `lf api prompts get ... --json` response for the current prompt.",
    ),
    candidate_prompt_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Candidate prompt body file produced by `candidate-prompt-file`.",
    ),
    metadata_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Candidate metadata JSON produced by `candidate-prompt-file`.",
    ),
    output_path: Path = typer.Argument(
        ...,
        dir_okay=False,
        writable=True,
        help="Output review packet path.",
    ),
    create_curl_path: Path | None = typer.Option(
        None,
        "--create-curl",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Optional dry-run `lf api prompts create --curl` output file.",
    ),
    output_format: ExperimentPlanFormat = typer.Option(
        ExperimentPlanFormat.markdown,
        "--format",
        case_sensitive=False,
        help="Output format.",
    ),
) -> None:
    """Review a candidate prompt file without exposing the full prompt body."""
    try:
        review = build_candidate_review(
            prompt_json_path,
            candidate_prompt_path,
            metadata_path,
            create_curl_path=create_curl_path,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == ExperimentPlanFormat.json:
        output_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    else:
        output_path.write_text(render_candidate_review_markdown(review))
    typer.echo(
        " | ".join(
            [
                f"wrote={output_path}",
                f"prompt={review['prompt']['name']}",
                f"ready_for_human_review={review['ready_for_human_review']}",
            ]
        )
    )


@app.command("candidate-runtime-plan")
def candidate_runtime_plan_command(
    candidate_review_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Candidate review JSON or Markdown companion JSON produced by `candidate-review`.",
    ),
    output_path: Path = typer.Argument(
        ...,
        dir_okay=False,
        writable=True,
        help="Output candidate runtime plan path.",
    ),
    environment: str = typer.Option(
        "test",
        "--environment",
        help="Target Care AI/Langfuse environment for candidate trace collection.",
    ),
    executor: str | None = typer.Option(
        None,
        "--executor",
        help="Executor to collect candidate traces from. Defaults to review scope.",
    ),
    focus: DiagnosticFocus = typer.Option(
        DiagnosticFocus.cost,
        "--focus",
        "-f",
        case_sensitive=False,
        help="HALO focus for candidate trace collection and comparison.",
    ),
    trace_output_dir: Path = typer.Option(
        Path(".halo-careai/export/candidate-runtime"),
        "--trace-output-dir",
        dir_okay=True,
        file_okay=False,
        help="Directory for candidate trace export artifacts.",
    ),
    output_format: ExperimentPlanFormat = typer.Option(
        ExperimentPlanFormat.markdown,
        "--format",
        case_sensitive=False,
        help="Output format.",
    ),
) -> None:
    """Write an approval-gated runtime plan for collecting candidate traces."""
    try:
        plan = build_candidate_runtime_plan(
            candidate_review_path,
            environment=environment,
            executor=executor,
            focus=focus,
            trace_output_dir=trace_output_dir,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == ExperimentPlanFormat.json:
        output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    else:
        output_path.write_text(render_candidate_runtime_plan_markdown(plan))
    typer.echo(
        " | ".join(
            [
                f"wrote={output_path}",
                f"prompt={plan['prompt']['name']}",
                f"approval_required={plan['approval_required']}",
            ]
        )
    )


@app.command("candidate-preflight")
def candidate_preflight_command(
    prompt_json_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Fresh raw `lf api prompts get ... --label latest --json` response.",
    ),
    candidate_review_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Candidate review JSON produced by `candidate-review --format json`.",
    ),
    runtime_plan_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Runtime plan JSON produced by `candidate-runtime-plan --format json`.",
    ),
    output_path: Path = typer.Argument(
        ...,
        dir_okay=False,
        writable=True,
        help="Output candidate preflight packet path.",
    ),
    output_format: ExperimentPlanFormat = typer.Option(
        ExperimentPlanFormat.markdown,
        "--format",
        case_sensitive=False,
        help="Output format.",
    ),
) -> None:
    """Validate that a reviewed candidate still matches the current runtime prompt."""
    preflight = build_candidate_preflight(
        prompt_json_path,
        candidate_review_path,
        runtime_plan_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == ExperimentPlanFormat.json:
        output_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    else:
        output_path.write_text(render_candidate_preflight_markdown(preflight))
    typer.echo(
        " | ".join(
            [
                f"wrote={output_path}",
                f"prompt={preflight['prompt']['name']}",
                f"preflight_passed={preflight['preflight_passed']}",
            ]
        )
    )
    if preflight["preflight_passed"] is not True:
        raise typer.Exit(1)


@app.command("candidate-runtime-check")
def candidate_runtime_check_command(
    created_prompt_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Raw `lf api prompts create ... --json` response from the approved runtime push.",
    ),
    candidate_review_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Candidate review JSON produced by `candidate-review --format json`.",
    ),
    runtime_plan_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Runtime plan JSON produced by `candidate-runtime-plan --format json`.",
    ),
    output_path: Path = typer.Argument(
        ...,
        dir_okay=False,
        writable=True,
        help="Output candidate runtime check packet path.",
    ),
    preflight_path: Path | None = typer.Option(
        None,
        "--preflight",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Optional candidate preflight JSON.",
    ),
    output_format: ExperimentPlanFormat = typer.Option(
        ExperimentPlanFormat.markdown,
        "--format",
        case_sensitive=False,
        help="Output format.",
    ),
) -> None:
    """Verify an approved prompt push before collecting candidate traces."""
    runtime_check = build_candidate_runtime_check(
        created_prompt_path,
        candidate_review_path,
        runtime_plan_path,
        preflight_path=preflight_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == ExperimentPlanFormat.json:
        output_path.write_text(json.dumps(runtime_check, indent=2, sort_keys=True) + "\n")
    else:
        output_path.write_text(render_candidate_runtime_check_markdown(runtime_check))
    typer.echo(
        " | ".join(
            [
                f"wrote={output_path}",
                f"prompt={runtime_check['prompt']['name']}",
                f"runtime_check_passed={runtime_check['runtime_check_passed']}",
            ]
        )
    )
    if runtime_check["runtime_check_passed"] is not True:
        raise typer.Exit(1)


@app.command("candidate-local-pipeline")
def candidate_local_pipeline_command(
    current_prompt_json_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Current Langfuse prompt export JSON, usually fetched with `--label latest`.",
    ),
    evidence_pack_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Baseline evidence pack JSON produced by `evidence-pack` or `local-pipeline`.",
    ),
    output_dir: Path = typer.Argument(
        ...,
        file_okay=False,
        writable=True,
        help="Directory for local candidate artifacts.",
    ),
    environment: str = typer.Option(
        "test",
        "--environment",
        help="Target Care AI/Langfuse environment for the runtime plan.",
    ),
    executor: str | None = typer.Option(
        None,
        "--executor",
        help="Executor to collect candidate traces from. Defaults to evidence scope.",
    ),
    focus: DiagnosticFocus | None = typer.Option(
        None,
        "--focus",
        case_sensitive=False,
        help="HALO focus. Defaults to evidence scope.",
    ),
    diagnosis_report_path: Path | None = typer.Option(
        None,
        "--diagnosis-report",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Optional HALO diagnosis report to shape the candidate handoff.",
    ),
    trace_output_dir: Path = typer.Option(
        Path(".halo-careai/export/candidate-runtime"),
        "--trace-output-dir",
        file_okay=False,
        help="Directory where future candidate trace exports should be written.",
    ),
) -> None:
    """Write local candidate artifacts from a prompt export and evidence pack."""
    try:
        manifest = build_candidate_local_pipeline(
            current_prompt_json_path,
            evidence_pack_path,
            output_dir,
            environment=environment,
            executor=executor,
            focus=focus,
            diagnosis_report_path=diagnosis_report_path,
            trace_output_dir=trace_output_dir,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    typer.echo(
        " | ".join(
            [
                f"candidate_prompt={manifest['artifacts']['candidate_prompt']}",
                f"review={manifest['artifacts']['candidate_review_json']}",
                f"runtime_plan={manifest['artifacts']['runtime_plan_json']}",
                f"loop_status={manifest['artifacts']['loop_status']}",
                f"state={manifest['loop_status']['state']}",
            ]
        )
    )


@app.command("candidate-decision")
def candidate_decision_command(
    comparison_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Comparison JSON produced by `halo-careai compare-audits --json`.",
    ),
    output_path: Path = typer.Argument(
        ...,
        dir_okay=False,
        writable=True,
        help="Output candidate decision packet path.",
    ),
    runtime_plan_path: Path | None = typer.Option(
        None,
        "--runtime-plan",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Optional runtime plan JSON produced by `candidate-runtime-plan --format json`.",
    ),
    focus: DiagnosticFocus = typer.Option(
        DiagnosticFocus.cost,
        "--focus",
        "-f",
        case_sensitive=False,
        help="HALO focus used for decision thresholds.",
    ),
    min_candidate_trace_ratio: float = typer.Option(
        0.8,
        "--min-candidate-trace-ratio",
        min=0.0,
        max=1.0,
        help="Minimum candidate/baseline trace ratio before making a promote decision.",
    ),
    candidate_traffic_note: str | None = typer.Option(
        None,
        "--candidate-traffic-note",
        help=(
            "Optional metadata-only note describing how candidate traffic was collected, "
            "for example direct visitor API traffic versus authenticated shopper UI traffic."
        ),
    ),
    output_format: ExperimentPlanFormat = typer.Option(
        ExperimentPlanFormat.markdown,
        "--format",
        case_sensitive=False,
        help="Output format.",
    ),
) -> None:
    """Decide promote, iterate, rollback, or collect more traces from comparison output."""
    decision = build_candidate_decision(
        comparison_path,
        runtime_plan_path=runtime_plan_path,
        focus=focus,
        min_candidate_trace_ratio=min_candidate_trace_ratio,
        candidate_traffic_note=candidate_traffic_note,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == ExperimentPlanFormat.json:
        output_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    else:
        output_path.write_text(render_candidate_decision_markdown(decision))
    typer.echo(
        " | ".join(
            [
                f"wrote={output_path}",
                f"decision={decision['decision']}",
                f"approval_required={decision['approval_required']}",
            ]
        )
    )


@app.command("candidate-evaluate")
def candidate_evaluate_command(
    evidence_pack_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Baseline evidence pack JSON produced by `halo-careai evidence-pack`.",
    ),
    candidate_trace_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Candidate HALO-compatible JSONL trace file after the prompt/config change.",
    ),
    output_dir: Path = typer.Argument(
        ...,
        dir_okay=True,
        file_okay=False,
        writable=True,
        help="Directory for comparison, evidence, decision, and manifest artifacts.",
    ),
    runtime_plan_path: Path | None = typer.Option(
        None,
        "--runtime-plan",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Optional runtime plan JSON produced by `candidate-runtime-plan --format json`.",
    ),
    runtime_check_path: Path | None = typer.Option(
        None,
        "--runtime-check",
        exists=True,
        readable=True,
        dir_okay=False,
        help=(
            "Optional passing runtime-check JSON. When provided, candidate traces must include "
            "the pushed prompt name and version."
        ),
    ),
    prompt_snapshot_path: Path | None = typer.Option(
        None,
        "--prompt-snapshot",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Optional metadata-only prompt snapshot to include in the candidate evidence pack.",
    ),
    focus: DiagnosticFocus | None = typer.Option(
        None,
        "--focus",
        "-f",
        case_sensitive=False,
        help="Override the evidence-pack focus. Defaults to the baseline evidence scope.",
    ),
    executor: str | None = typer.Option(
        None,
        "--executor",
        help="Override the evidence-pack executor. Defaults to the baseline evidence scope.",
    ),
    min_candidate_trace_ratio: float = typer.Option(
        0.8,
        "--min-candidate-trace-ratio",
        min=0.0,
        max=1.0,
        help="Minimum candidate/baseline trace ratio before making a promote decision.",
    ),
    top: int = typer.Option(25, "--top", min=1, max=100, help="Rows per finding type."),
    high_input_tokens: int = typer.Option(
        32_000,
        "--high-input-tokens",
        min=1,
        help="Flag LLM spans with at least this many input tokens.",
    ),
    high_output_tokens: int = typer.Option(
        2_000,
        "--high-output-tokens",
        min=1,
        help="Flag LLM spans with at least this many output tokens.",
    ),
    model: str = typer.Option("gpt-5.4-mini", "--model", "-m"),
    check_model: bool = typer.Option(
        False,
        "--check-model",
        help="Make a live model API request in the included evidence-pack doctor report.",
    ),
    use_gocode: bool = typer.Option(
        False,
        "--gocode",
        help="Use Tommy's GoCode provider for the live model check.",
    ),
    candidate_traffic_note: str | None = typer.Option(
        None,
        "--candidate-traffic-note",
        help=(
            "Optional metadata-only note describing how candidate traffic was collected, "
            "for example direct visitor API traffic versus authenticated shopper UI traffic."
        ),
    ),
) -> None:
    """Evaluate real candidate traces and write the post-candidate decision bundle."""
    try:
        manifest = build_candidate_evaluation(
            evidence_pack_path,
            candidate_trace_path,
            output_dir,
            runtime_plan_path=runtime_plan_path,
            runtime_check_path=runtime_check_path,
            prompt_snapshot_path=prompt_snapshot_path,
            focus=focus,
            executor=executor,
            min_candidate_trace_ratio=min_candidate_trace_ratio,
            top_n=top,
            high_input_tokens=high_input_tokens,
            high_output_tokens=high_output_tokens,
            model=model,
            check_model=check_model,
            use_gocode=use_gocode,
            candidate_traffic_note=candidate_traffic_note,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    artifacts = manifest["artifacts"]
    typer.echo(
        " | ".join(
            [
                f"manifest={artifacts['manifest']}",
                f"decision={manifest['decision']['decision']}",
                f"approval_required={manifest['decision']['approval_required']}",
            ]
        )
    )


@app.command("loop-status")
def loop_status_command(
    output_path: Path | None = typer.Option(
        None,
        "--output",
        dir_okay=False,
        writable=True,
        help="Optional output path. If omitted, status is printed to stdout.",
    ),
    evidence_pack_path: Path | None = typer.Option(
        None,
        "--evidence-pack",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Baseline evidence pack JSON.",
    ),
    candidate_review_path: Path | None = typer.Option(
        None,
        "--candidate-review",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Candidate review JSON.",
    ),
    runtime_plan_path: Path | None = typer.Option(
        None,
        "--runtime-plan",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Candidate runtime plan JSON.",
    ),
    preflight_path: Path | None = typer.Option(
        None,
        "--preflight",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Candidate preflight JSON.",
    ),
    runtime_check_path: Path | None = typer.Option(
        None,
        "--runtime-check",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Candidate runtime check JSON.",
    ),
    evaluation_manifest_path: Path | None = typer.Option(
        None,
        "--evaluation",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Candidate evaluation manifest JSON.",
    ),
    output_format: ExperimentPlanFormat = typer.Option(
        ExperimentPlanFormat.markdown,
        "--format",
        case_sensitive=False,
        help="Output format.",
    ),
) -> None:
    """Audit the local Care AI HALO loop state without mutating external systems."""
    try:
        status = build_candidate_loop_status(
            evidence_pack_path=evidence_pack_path,
            candidate_review_path=candidate_review_path,
            runtime_plan_path=runtime_plan_path,
            preflight_path=preflight_path,
            runtime_check_path=runtime_check_path,
            evaluation_manifest_path=evaluation_manifest_path,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    rendered = (
        json.dumps(status, indent=2, sort_keys=True) + "\n"
        if output_format == ExperimentPlanFormat.json
        else render_candidate_loop_status_markdown(status)
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered)
        typer.echo(
            " | ".join(
                [
                    f"wrote={output_path}",
                    f"state={status['state']}",
                    f"external_approval_required={status['external_approval_required']}",
                ]
            )
        )
        return
    typer.echo(rendered, nl=False)


@app.command("local-pipeline")
def local_pipeline_command(
    trace_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="HALO-compatible JSONL trace file, preferably sanitized.",
    ),
    output_dir: Path = typer.Argument(
        _DEFAULT_PIPELINE_DIR,
        dir_okay=True,
        file_okay=False,
        help="Directory for evidence, diagnosis, plan, and manifest artifacts.",
    ),
    focus: DiagnosticFocus = typer.Option(
        DiagnosticFocus.cost,
        "--focus",
        "-f",
        case_sensitive=False,
        help="Care AI diagnostic focus.",
    ),
    executor: str | None = typer.Option(
        None,
        "--executor",
        help="Prioritize one care_ai.session_executor_name.",
    ),
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        help="Restrict prompt instructions to one langfuse.session_id / care_ai.conversation_id.",
    ),
    question: str | None = typer.Option(
        None,
        "--question",
        "-q",
        help="Additional trace-data question to append to the selected focus prompt.",
    ),
    top: int = typer.Option(25, "--top", min=1, max=100, help="Rows per finding type."),
    diagnose_enabled: bool = typer.Option(
        False,
        "--diagnose/--skip-diagnose",
        help="Run live HALO diagnosis after writing the evidence pack.",
    ),
    model: str = typer.Option("gpt-5.4-mini", "--model", "-m"),
    use_gocode: bool = typer.Option(
        True,
        "--gocode/--no-gocode",
        help="Route live diagnosis through Tommy's GoCode provider.",
    ),
    tool_mode: TraceToolMode = typer.Option(
        TraceToolMode.summary,
        "--tool-mode",
        case_sensitive=False,
        help="Trace tool set for live diagnosis.",
    ),
    reasoning_effort: str | None = typer.Option(
        "low",
        "--reasoning-effort",
        help=(
            "Reasoning effort forwarded to HALO diagnosis. One of: "
            f"{', '.join(REASONING_EFFORT_CHOICES)}."
        ),
    ),
    max_depth: int = typer.Option(0, "--max-depth", min=0),
    max_turns: int = typer.Option(4, "--max-turns", min=1),
    max_parallel: int = typer.Option(1, "--max-parallel", min=1),
    timeout_seconds: int | None = typer.Option(
        60,
        "--timeout-seconds",
        min=1,
        help="Abort live diagnosis after this many seconds.",
    ),
    prompt_snapshot_path: Path | None = typer.Option(
        None,
        "--prompt-snapshot",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Optional metadata-only prompt snapshot to include in generated artifacts.",
    ),
) -> None:
    """Run the local Care AI HALO artifact pipeline from an existing trace file."""
    paths = build_pipeline_paths(trace_path, output_dir, focus=focus, executor=executor)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths["diagnosis_report"].parent.mkdir(parents=True, exist_ok=True)

    pack = build_evidence_pack(
        trace_path,
        focus=focus,
        executor=executor,
        session_id=session_id,
        extra_question=question,
        top_n=top,
        model=model,
        use_gocode=use_gocode,
        prompt_snapshot_path=prompt_snapshot_path,
    )
    paths["evidence_pack"].write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n")

    diagnosis_status = "skipped"
    diagnosis_report_path: Path | None = None
    if diagnose_enabled:
        prompt = build_diagnosis_prompt_with_evidence(
            trace_path,
            focus=focus,
            executor=executor,
            session_id=session_id,
            extra_question=question,
            include_evidence=True,
            evidence_top=min(top, 10),
        )
        try:
            with _model_provider_env(use_gocode=use_gocode):
                run_trace(
                    trace_path=trace_path,
                    prompt=prompt,
                    model=model,
                    max_depth=max_depth,
                    max_turns=max_turns,
                    max_parallel=max_parallel,
                    refusal_retries=0,
                    reasoning_effort=reasoning_effort,
                    telemetry=False,
                    trace_detail_tools_enabled=tool_mode == TraceToolMode.full,
                    run_code_enabled=tool_mode == TraceToolMode.full,
                    timeout_seconds=timeout_seconds,
                    output_path=paths["diagnosis_report"],
                    events_path=paths["diagnosis_events"],
                )
            diagnosis_status = "completed"
            diagnosis_report_path = paths["diagnosis_report"]
        except TimeoutError as exc:
            diagnosis_status = "timed_out"
            _write_pipeline_manifest(
                paths["manifest"],
                _pipeline_manifest(
                    trace_path=trace_path,
                    paths=paths,
                    focus=focus,
                    executor=executor,
                    diagnosis_status=diagnosis_status,
                    diagnose_enabled=diagnose_enabled,
                    model=model,
                    use_gocode=use_gocode,
                    tool_mode=tool_mode,
                    max_depth=max_depth,
                    max_turns=max_turns,
                    max_parallel=max_parallel,
                    timeout_seconds=timeout_seconds,
                    evidence_pack=pack,
                ),
            )
            typer.echo(
                "HALO diagnosis timed out; evidence pack was written and manifest records the timeout.",
                err=True,
            )
            raise typer.Exit(1) from exc
        except Exception as exc:
            if _is_authentication_error(exc):
                diagnosis_status = "authentication_failed"
                _write_pipeline_manifest(
                    paths["manifest"],
                    _pipeline_manifest(
                        trace_path=trace_path,
                        paths=paths,
                        focus=focus,
                        executor=executor,
                        diagnosis_status=diagnosis_status,
                        diagnose_enabled=diagnose_enabled,
                        model=model,
                        use_gocode=use_gocode,
                        tool_mode=tool_mode,
                        max_depth=max_depth,
                        max_turns=max_turns,
                        max_parallel=max_parallel,
                        timeout_seconds=timeout_seconds,
                        evidence_pack=pack,
                    ),
                )
                typer.echo(
                    "HALO diagnosis could not authenticate. Evidence pack was written; run doctor with --gocode.",
                    err=True,
                )
                raise typer.Exit(1) from exc
            raise

    plan = build_experiment_plan(
        paths["evidence_pack"],
        diagnosis_report_path=diagnosis_report_path,
    )
    paths["experiment_plan"].write_text(render_experiment_plan_markdown(plan))
    handoff = build_candidate_handoff(
        paths["evidence_pack"],
        diagnosis_report_path=diagnosis_report_path,
    )
    paths["candidate_handoff"].write_text(render_candidate_handoff_markdown(handoff))
    loop_status = build_candidate_loop_status(evidence_pack_path=paths["evidence_pack"])
    paths["loop_status"].write_text(render_candidate_loop_status_markdown(loop_status))

    manifest = _pipeline_manifest(
        trace_path=trace_path,
        paths=paths,
        focus=focus,
        executor=executor,
        diagnosis_status=diagnosis_status,
        diagnose_enabled=diagnose_enabled,
        model=model,
        use_gocode=use_gocode,
        tool_mode=tool_mode,
        max_depth=max_depth,
        max_turns=max_turns,
        max_parallel=max_parallel,
        timeout_seconds=timeout_seconds,
        evidence_pack=pack,
    )
    _write_pipeline_manifest(paths["manifest"], manifest)
    console.print(
        " | ".join(
            [
                f"evidence={paths['evidence_pack']}",
                f"diagnosis={diagnosis_status}",
                f"plan={paths['experiment_plan']}",
                f"handoff={paths['candidate_handoff']}",
                f"loop_status={paths['loop_status']}",
                f"manifest={paths['manifest']}",
            ]
        )
    )


def _pipeline_manifest(
    *,
    trace_path: Path,
    paths: dict[str, Path],
    focus: DiagnosticFocus,
    executor: str | None,
    diagnosis_status: str,
    diagnose_enabled: bool,
    model: str,
    use_gocode: bool,
    tool_mode: TraceToolMode,
    max_depth: int,
    max_turns: int,
    max_parallel: int,
    timeout_seconds: int | None,
    evidence_pack: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "purpose": "care-ai-halo-local-pipeline-manifest",
        "trace_path": str(trace_path),
        "focus": focus.value,
        "executor": executor,
        "diagnosis": {
            "enabled": diagnose_enabled,
            "status": diagnosis_status,
            "model": model,
            "model_provider": "gocode" if use_gocode else "environment",
            "tool_mode": tool_mode.value,
            "max_depth": max_depth,
            "max_turns": max_turns,
            "max_parallel": max_parallel,
            "timeout_seconds": timeout_seconds,
        },
        "artifacts": {key: str(path) for key, path in paths.items() if key != "output_dir"},
    }
    if evidence_pack is not None:
        inspect_report = evidence_pack.get("inspect", {})
        audit_report = evidence_pack.get("audit", {})
        safety_report = evidence_pack.get("safety", {})
        manifest["summary"] = {
            "total_traces": inspect_report.get("total_traces", 0),
            "total_spans": inspect_report.get("total_spans", 0),
            "total_input_tokens": inspect_report.get("total_input_tokens", 0),
            "total_output_tokens": inspect_report.get("total_output_tokens", 0),
            "session_trace_distribution": inspect_report.get("session_trace_distribution", {}),
            "session_span_distribution": inspect_report.get("session_span_distribution", {}),
            "session_input_token_distribution": inspect_report.get(
                "session_input_token_distribution", {}
            ),
            "session_output_token_distribution": inspect_report.get(
                "session_output_token_distribution", {}
            ),
            "llm_input_token_distribution": inspect_report.get("llm_input_token_distribution", {}),
            "llm_output_token_distribution": inspect_report.get(
                "llm_output_token_distribution", {}
            ),
            "audit_counts": audit_report.get("counts", {}),
            "safe_for_metadata_only_diagnosis": safety_report.get(
                "safe_for_metadata_only_diagnosis"
            ),
        }
    return manifest


def _write_pipeline_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def inspect_halo_jsonl(trace_path: Path, *, top_n: int = 10) -> dict[str, Any]:
    """Return local, metadata-level summary stats for a HALO JSONL trace file."""
    trace_ids: set[str] = set()
    error_trace_ids: set[str] = set()
    session_ids: set[str] = set()
    ucids: set[str] = set()
    project_ids: Counter[str] = Counter()
    service_names: Counter[str] = Counter()
    observation_kinds: Counter[str] = Counter()
    model_names: Counter[str] = Counter()
    agent_names: Counter[str] = Counter()
    executor_names: Counter[str] = Counter()
    tool_names: Counter[str] = Counter()
    planned_tool_names: Counter[str] = Counter()
    tool_error_names: Counter[str] = Counter()
    status_codes: Counter[str] = Counter()
    span_names: Counter[str] = Counter()
    total_input_tokens = 0
    total_output_tokens = 0
    total_agent_reported_tool_calls = 0
    llm_input_tokens: list[int] = []
    llm_output_tokens: list[int] = []
    total_raw_jsonl_bytes = 0
    total_spans = 0
    sessions: dict[str, dict[str, Any]] = {}

    with trace_path.open("rb") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            total_raw_jsonl_bytes += len(stripped)
            total_spans += 1
            span = SpanRecord.model_validate_json(stripped)
            attrs = span.attributes
            resource_attrs = span.resource.attributes

            trace_ids.add(span.trace_id)
            status_codes.update([span.status.code])
            span_names.update([span.name])
            if span.status.code == "STATUS_CODE_ERROR":
                error_trace_ids.add(span.trace_id)
            session_id = _session_key(attrs)
            session = sessions.setdefault(
                session_id,
                {
                    "trace_ids": set(),
                    "span_count": 0,
                    "error_trace_ids": set(),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reported_tool_calls": 0,
                    "executed_tool_calls": 0,
                    "planned_tool_calls": 0,
                },
            )
            session["trace_ids"].add(span.trace_id)
            session["span_count"] += 1
            if span.status.code == "STATUS_CODE_ERROR":
                session["error_trace_ids"].add(span.trace_id)

            _counter_update(project_ids, attrs.get("inference.project_id"))
            _counter_update(service_names, resource_attrs.get("service.name"))
            observation_kind = attrs.get("inference.observation_kind")
            _counter_update(observation_kinds, observation_kind)
            _counter_update_unique(
                model_names,
                [attrs.get("inference.llm.model_name"), attrs.get("llm.model_name")],
            )
            _counter_update_unique(
                agent_names,
                [attrs.get("inference.agent_name"), attrs.get("agent.name")],
            )
            _counter_update(executor_names, attrs.get("care_ai.session_executor_name"))
            _set_update(session_ids, attrs.get("langfuse.session_id"))
            _set_update(session_ids, attrs.get("care_ai.conversation_id"))
            _set_update(ucids, attrs.get("care_ai.ucid"))

            input_tokens = attrs.get("inference.llm.input_tokens")
            output_tokens = attrs.get("inference.llm.output_tokens")
            if isinstance(input_tokens, int):
                total_input_tokens += input_tokens
                llm_input_tokens.append(input_tokens)
                session["input_tokens"] += input_tokens
            if isinstance(output_tokens, int):
                total_output_tokens += output_tokens
                llm_output_tokens.append(output_tokens)
                session["output_tokens"] += output_tokens
            reported_tool_calls = attrs.get("care_ai.agent_reported_tool_call_count")
            if isinstance(reported_tool_calls, int):
                total_agent_reported_tool_calls += reported_tool_calls
                session["reported_tool_calls"] += reported_tool_calls

            tool_name = attrs.get("tool.name")
            if isinstance(tool_name, str) and tool_name:
                tool_names.update([tool_name])
                session["executed_tool_calls"] += 1
                if span.status.code == "STATUS_CODE_ERROR":
                    tool_error_names.update([tool_name])
            planned_tools = attrs.get("care_ai.llm_planned_tool_names")
            if isinstance(planned_tools, list):
                valid_planned_tools = [
                    name for name in planned_tools if isinstance(name, str) and name
                ]
                planned_tool_names.update(valid_planned_tools)
                session["planned_tool_calls"] += len(valid_planned_tools)

    session_summary = _session_summary(sessions, top_n=top_n)

    return {
        "trace_path": str(trace_path),
        "total_traces": len(trace_ids),
        "total_spans": total_spans,
        "error_trace_count": len(error_trace_ids),
        "session_count": len(session_ids),
        "ucid_count": len(ucids),
        "session_trace_distribution": session_summary["trace_distribution"],
        "session_span_distribution": session_summary["span_distribution"],
        "session_input_token_distribution": session_summary["input_token_distribution"],
        "session_output_token_distribution": session_summary["output_token_distribution"],
        "session_reported_tool_call_distribution": session_summary[
            "reported_tool_call_distribution"
        ],
        "session_executed_tool_call_distribution": session_summary[
            "executed_tool_call_distribution"
        ],
        "session_planned_tool_call_distribution": session_summary["planned_tool_call_distribution"],
        "top_sessions_by_input_tokens": session_summary["top_by_input_tokens"],
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_agent_reported_tool_calls": total_agent_reported_tool_calls,
        "llm_input_token_distribution": _token_distribution(llm_input_tokens),
        "llm_output_token_distribution": _token_distribution(llm_output_tokens),
        "raw_jsonl_bytes": total_raw_jsonl_bytes,
        "project_ids": _top_counts(project_ids, top_n),
        "service_names": _top_counts(service_names, top_n),
        "observation_kinds": _top_counts(observation_kinds, top_n),
        "model_names": _top_counts(model_names, top_n),
        "agent_names": _top_counts(agent_names, top_n),
        "executor_names": _top_counts(executor_names, top_n),
        "tool_names": _top_counts(tool_names, top_n),
        "planned_tool_names": _top_counts(planned_tool_names, top_n),
        "tool_error_names": _top_counts(tool_error_names, top_n),
        "status_codes": _top_counts(status_codes, top_n),
        "span_names": _top_counts(span_names, top_n),
    }


def audit_halo_jsonl(
    trace_path: Path,
    *,
    high_input_tokens: int = 32_000,
    high_output_tokens: int = 2_000,
    top_n: int = 25,
) -> dict[str, Any]:
    """Return deterministic Care AI trace-risk signals without running an LLM."""
    trace_spans: dict[str, list[SpanRecord]] = defaultdict(list)
    for span in _iter_halo_spans(trace_path):
        trace_spans[span.trace_id].append(span)

    planned_missing_tool_calls: list[dict[str, Any]] = []
    repeated_tool_calls: list[dict[str, Any]] = []
    transfer_without_diagnostic_tool: list[dict[str, Any]] = []
    tool_errors: list[dict[str, Any]] = []
    high_token_llm_spans: list[dict[str, Any]] = []
    reported_tool_count_mismatch: list[dict[str, Any]] = []

    for trace_id, spans in trace_spans.items():
        executed_tool_names: list[str] = []
        planned_tool_to_span_ids: dict[str, set[str]] = defaultdict(set)
        tool_input_groups: dict[tuple[str, str], list[SpanRecord]] = defaultdict(list)
        agent_reported_tool_count = 0
        agent_span_ids: list[str] = []
        executors = _trace_executors(spans)

        for span in spans:
            attrs = span.attributes
            observation_kind = attrs.get("inference.observation_kind")
            if observation_kind == "AGENT":
                reported_count = attrs.get("care_ai.agent_reported_tool_call_count")
                if isinstance(reported_count, int):
                    agent_reported_tool_count += reported_count
                    agent_span_ids.append(span.span_id)
            tool_name = attrs.get("tool.name")
            if isinstance(tool_name, str) and tool_name:
                executed_tool_names.append(tool_name)
                input_sha, _ = _fingerprint_value(attrs.get("input.value"))
                tool_input_groups[(tool_name, input_sha)].append(span)
                if span.status.code == "STATUS_CODE_ERROR":
                    tool_errors.append(
                        {
                            "trace_id": trace_id,
                            "span_id": span.span_id,
                            "tool_name": tool_name,
                            "status_message": span.status.message,
                            "executor_names": executors,
                        }
                    )

            if observation_kind == "LLM":
                planned_tools = _string_list(attrs.get("care_ai.llm_planned_tool_names"))
                for planned_tool in planned_tools:
                    planned_tool_to_span_ids[planned_tool].add(span.span_id)

                input_tokens = attrs.get("inference.llm.input_tokens")
                output_tokens = attrs.get("inference.llm.output_tokens")
                if (
                    isinstance(input_tokens, int)
                    and input_tokens >= high_input_tokens
                    or isinstance(output_tokens, int)
                    and output_tokens >= high_output_tokens
                ):
                    high_token_llm_spans.append(
                        {
                            "trace_id": trace_id,
                            "span_id": span.span_id,
                            "model_name": _first_string_value(
                                attrs,
                                ["inference.llm.model_name", "llm.model_name"],
                            ),
                            "input_tokens": input_tokens if isinstance(input_tokens, int) else 0,
                            "output_tokens": (
                                output_tokens if isinstance(output_tokens, int) else 0
                            ),
                            "executor_names": executors,
                        }
                    )

        executed_tool_set = set(executed_tool_names)
        if agent_reported_tool_count and len(executed_tool_names) != agent_reported_tool_count:
            reported_tool_count_mismatch.append(
                {
                    "trace_id": trace_id,
                    "agent_span_ids": agent_span_ids,
                    "reported_tool_call_count": agent_reported_tool_count,
                    "executed_tool_count": len(executed_tool_names),
                    "executed_tool_names": sorted(executed_tool_set),
                    "planned_tool_names": sorted(planned_tool_to_span_ids),
                    "executor_names": executors,
                }
            )
        for planned_tool, span_ids in planned_tool_to_span_ids.items():
            if planned_tool not in executed_tool_set:
                planned_missing_tool_calls.append(
                    {
                        "trace_id": trace_id,
                        "planned_tool_name": planned_tool,
                        "llm_span_ids": sorted(span_ids),
                        "executed_tool_names": sorted(executed_tool_set),
                        "executor_names": executors,
                    }
                )

        for (tool_name, input_sha), repeated_spans in tool_input_groups.items():
            if len(repeated_spans) < 2:
                continue
            first_input = repeated_spans[0].attributes.get("input.value")
            _, input_bytes = _fingerprint_value(first_input)
            repeated_tool_calls.append(
                {
                    "trace_id": trace_id,
                    "tool_name": tool_name,
                    "count": len(repeated_spans),
                    "span_ids": [span.span_id for span in repeated_spans],
                    "input_sha256_12": input_sha,
                    "input_bytes": input_bytes,
                    "executor_names": executors,
                }
            )

        transfer_span_ids = [
            span.span_id
            for span in spans
            if span.attributes.get("tool.name") == "transfer_to_human"
        ]
        non_transfer_tools = [
            tool_name for tool_name in executed_tool_names if tool_name != "transfer_to_human"
        ]
        if transfer_span_ids and not non_transfer_tools:
            transfer_without_diagnostic_tool.append(
                {
                    "trace_id": trace_id,
                    "transfer_span_ids": transfer_span_ids,
                    "executed_tool_names": sorted(executed_tool_set),
                    "planned_tool_names": sorted(planned_tool_to_span_ids),
                    "executor_names": executors,
                }
            )

    all_findings = {
        "planned_missing_tool_calls": planned_missing_tool_calls,
        "repeated_tool_calls": repeated_tool_calls,
        "transfer_without_diagnostic_tool": transfer_without_diagnostic_tool,
        "tool_errors": tool_errors,
        "high_token_llm_spans": high_token_llm_spans,
        "reported_tool_count_mismatch": reported_tool_count_mismatch,
    }
    findings = {key: value[:top_n] for key, value in all_findings.items()}
    return {
        "trace_path": str(trace_path),
        "total_traces": len(trace_spans),
        "thresholds": {
            "high_input_tokens": high_input_tokens,
            "high_output_tokens": high_output_tokens,
        },
        "counts": {key: len(value) for key, value in all_findings.items()},
        **findings,
    }


def compare_audits(
    baseline_path: Path,
    candidate_path: Path,
    *,
    high_input_tokens: int = 32_000,
    high_output_tokens: int = 2_000,
    top_n: int = 25,
) -> dict[str, Any]:
    """Compare deterministic audit signals for two Care AI trace sets."""
    baseline = audit_halo_jsonl(
        baseline_path,
        high_input_tokens=high_input_tokens,
        high_output_tokens=high_output_tokens,
        top_n=top_n,
    )
    candidate = audit_halo_jsonl(
        candidate_path,
        high_input_tokens=high_input_tokens,
        high_output_tokens=high_output_tokens,
        top_n=top_n,
    )
    baseline_inspect = inspect_halo_jsonl(baseline_path, top_n=top_n)
    candidate_inspect = inspect_halo_jsonl(candidate_path, top_n=top_n)

    signal_deltas = {
        signal: _compare_count(
            baseline["counts"][signal],
            candidate["counts"][signal],
            baseline["total_traces"],
            candidate["total_traces"],
        )
        for signal in baseline["counts"]
    }
    return {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "thresholds": baseline["thresholds"],
        "trace_counts": {
            "baseline": baseline["total_traces"],
            "candidate": candidate["total_traces"],
            "delta": candidate["total_traces"] - baseline["total_traces"],
        },
        "token_totals": {
            "input": _compare_count(
                baseline_inspect["total_input_tokens"],
                candidate_inspect["total_input_tokens"],
                baseline["total_traces"],
                candidate["total_traces"],
            ),
            "output": _compare_count(
                baseline_inspect["total_output_tokens"],
                candidate_inspect["total_output_tokens"],
                baseline["total_traces"],
                candidate["total_traces"],
            ),
        },
        "token_distributions": {
            "input": _compare_token_distribution(
                baseline_inspect["llm_input_token_distribution"],
                candidate_inspect["llm_input_token_distribution"],
            ),
            "output": _compare_token_distribution(
                baseline_inspect["llm_output_token_distribution"],
                candidate_inspect["llm_output_token_distribution"],
            ),
        },
        "session_token_distributions": {
            "input": _compare_token_distribution(
                baseline_inspect["session_input_token_distribution"],
                candidate_inspect["session_input_token_distribution"],
            ),
            "output": _compare_token_distribution(
                baseline_inspect["session_output_token_distribution"],
                candidate_inspect["session_output_token_distribution"],
            ),
        },
        "signal_deltas": signal_deltas,
    }


def trace_safety_report(trace_path: Path, *, top_n: int = 25) -> dict[str, Any]:
    """Report whether a trace file still contains raw payloads or identifiers."""
    raw_payload_keys: Counter[str] = Counter()
    redacted_payload_marker_keys: Counter[str] = Counter()
    possible_raw_identifier_keys: Counter[str] = Counter()
    redacted_identifier_keys: Counter[str] = Counter()
    raw_status_message_count = 0
    redacted_status_message_count = 0
    total_spans = 0

    for span in _iter_halo_spans(trace_path):
        total_spans += 1
        attrs = span.attributes
        for key in _SENSITIVE_ATTRIBUTE_KEYS:
            if key in attrs:
                raw_payload_keys.update([key])
            safe_key = key.replace(".", "_")
            marker_key = f"care_ai.redacted.{safe_key}.sha256_12"
            if marker_key in attrs:
                redacted_payload_marker_keys.update([marker_key])

        for key in _SENSITIVE_IDENTIFIER_ATTRIBUTE_KEYS:
            value = attrs.get(key)
            if not isinstance(value, str) or not value:
                continue
            if value.startswith("redacted:"):
                redacted_identifier_keys.update([key])
            else:
                possible_raw_identifier_keys.update([key])

        if span.status.message:
            raw_status_message_count += 1
        if "care_ai.redacted.status_message.sha256_12" in attrs:
            redacted_status_message_count += 1

    raw_payload_attribute_count = sum(raw_payload_keys.values())
    possible_raw_identifier_attribute_count = sum(possible_raw_identifier_keys.values())
    safe_for_metadata_only_diagnosis = (
        raw_payload_attribute_count == 0
        and raw_status_message_count == 0
        and possible_raw_identifier_attribute_count == 0
    )
    return {
        "trace_path": str(trace_path),
        "total_spans": total_spans,
        "safe_for_metadata_only_diagnosis": safe_for_metadata_only_diagnosis,
        "raw_payload_attribute_count": raw_payload_attribute_count,
        "raw_payload_attribute_keys": _top_counts(raw_payload_keys, top_n),
        "redacted_payload_marker_count": sum(redacted_payload_marker_keys.values()),
        "redacted_payload_marker_keys": _top_counts(redacted_payload_marker_keys, top_n),
        "raw_status_message_count": raw_status_message_count,
        "redacted_status_message_count": redacted_status_message_count,
        "possible_raw_identifier_attribute_count": possible_raw_identifier_attribute_count,
        "possible_raw_identifier_attribute_keys": _top_counts(possible_raw_identifier_keys, top_n),
        "redacted_identifier_attribute_count": sum(redacted_identifier_keys.values()),
        "redacted_identifier_attribute_keys": _top_counts(redacted_identifier_keys, top_n),
        "notes": [
            "This report never prints raw payloads, status messages, or identifier values.",
            "possible_raw_identifier_attribute_count checks known identifier attributes whose values are not prefixed with redacted:.",
            "Top-level trace_id and span_id are not classified as raw or redacted because arbitrary source ids can be opaque.",
        ],
    }


def build_evidence_pack(
    trace_path: Path,
    *,
    candidate_path: Path | None = None,
    prompt_snapshot_path: Path | None = None,
    focus: DiagnosticFocus = DiagnosticFocus.overview,
    executor: str | None = None,
    session_id: str | None = None,
    extra_question: str | None = None,
    top_n: int = 25,
    model: str = "gpt-5.4-mini",
    check_model: bool = False,
    use_gocode: bool = False,
    high_input_tokens: int = 32_000,
    high_output_tokens: int = 2_000,
) -> dict[str, Any]:
    """Build a metadata-safe evidence bundle for HALO or human review."""
    baseline_audit = audit_halo_jsonl(
        trace_path,
        high_input_tokens=high_input_tokens,
        high_output_tokens=high_output_tokens,
        top_n=top_n,
    )
    pack: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "care-ai-halo-evidence-pack",
        "trace_path": str(trace_path),
        "scope": {
            "focus": focus.value,
            "executor": executor,
            "session_id": session_id,
            "question": extra_question,
            "top_n": top_n,
        },
        "care_ai_surface": care_ai_surface_for_executor(executor),
        "inspect": inspect_halo_jsonl(trace_path, top_n=top_n),
        "audit": _metadata_safe_audit_report(baseline_audit),
        "safety": trace_safety_report(trace_path, top_n=top_n),
        "doctor": doctor_report(
            trace_path,
            model=model,
            check_model=check_model,
            use_gocode=use_gocode,
        ),
        "diagnostic_prompt": build_diagnostic_prompt(
            focus=focus,
            executor=executor,
            session_id=session_id,
            extra_question=extra_question,
        ),
    }
    if prompt_snapshot_path is not None:
        pack["prompt_snapshot"] = load_prompt_snapshot(prompt_snapshot_path)
    if candidate_path is not None:
        candidate_audit = audit_halo_jsonl(
            candidate_path,
            high_input_tokens=high_input_tokens,
            high_output_tokens=high_output_tokens,
            top_n=top_n,
        )
        pack["candidate"] = {
            "trace_path": str(candidate_path),
            "inspect": inspect_halo_jsonl(candidate_path, top_n=top_n),
            "audit": _metadata_safe_audit_report(candidate_audit),
            "safety": trace_safety_report(candidate_path, top_n=top_n),
        }
        pack["comparison"] = compare_audits(
            trace_path,
            candidate_path,
            high_input_tokens=high_input_tokens,
            high_output_tokens=high_output_tokens,
            top_n=top_n,
        )
    return pack


def build_experiment_plan(
    evidence_pack_path: Path,
    *,
    diagnosis_report_path: Path | None = None,
) -> dict[str, Any]:
    """Create a deterministic Care AI harness experiment plan from HALO artifacts."""
    evidence_pack = _load_json_file(evidence_pack_path)
    scope = evidence_pack.get("scope", {})
    inspect_report = evidence_pack.get("inspect", {})
    audit_report = evidence_pack.get("audit", {})
    audit_counts = audit_report.get("counts", {})
    prompt_snapshot = _dict_or_empty(evidence_pack.get("prompt_snapshot"))
    focus = _string_or_default(scope.get("focus"), "overview")
    executor = _string_or_default(scope.get("executor"), "unknown-executor")
    care_ai_surface = _dict_or_empty(evidence_pack.get("care_ai_surface"))
    if not care_ai_surface:
        care_ai_surface = care_ai_surface_for_executor(executor)
    diagnosis_excerpt = (
        _extract_recommended_experiment(diagnosis_report_path.read_text())
        if diagnosis_report_path is not None
        else ""
    )
    title = _experiment_title(focus=focus, executor=executor, audit_counts=audit_counts)
    hypothesis = _experiment_hypothesis(focus=focus, executor=executor, audit_counts=audit_counts)
    candidate_change = _candidate_change(
        focus=focus,
        executor=executor,
        care_ai_surface=care_ai_surface,
    )
    success_metrics = _success_metrics(focus=focus)
    guardrails = _guardrails(focus=focus)
    verification_commands = _verification_commands(
        executor=executor,
        focus=focus,
        baseline_path=_string_or_default(evidence_pack.get("trace_path"), "<baseline.jsonl>"),
    )
    return {
        "schema_version": 1,
        "purpose": "care-ai-halo-experiment-plan",
        "source_artifacts": {
            "evidence_pack": str(evidence_pack_path),
            "diagnosis_report": str(diagnosis_report_path) if diagnosis_report_path else None,
        },
        "title": title,
        "scope": {
            "focus": focus,
            "executor": executor,
            "session_id": scope.get("session_id"),
        },
        "care_ai_surface": care_ai_surface,
        "prompt_snapshot": prompt_snapshot,
        "evidence_summary": {
            "total_traces": inspect_report.get("total_traces", 0),
            "total_spans": inspect_report.get("total_spans", 0),
            "total_input_tokens": inspect_report.get("total_input_tokens", 0),
            "total_output_tokens": inspect_report.get("total_output_tokens", 0),
            "session_trace_distribution": inspect_report.get("session_trace_distribution", {}),
            "session_span_distribution": inspect_report.get("session_span_distribution", {}),
            "session_input_token_distribution": inspect_report.get(
                "session_input_token_distribution", {}
            ),
            "top_sessions_by_input_tokens": inspect_report.get("top_sessions_by_input_tokens", []),
            "llm_input_token_distribution": inspect_report.get("llm_input_token_distribution", {}),
            "llm_output_token_distribution": inspect_report.get(
                "llm_output_token_distribution", {}
            ),
            "model_names": inspect_report.get("model_names", []),
            "tool_names": inspect_report.get("tool_names", []),
            "audit_counts": audit_counts,
        },
        "hypothesis": hypothesis,
        "candidate_change": candidate_change,
        "success_metrics": success_metrics,
        "guardrails": guardrails,
        "verification_commands": verification_commands,
        "diagnosis_recommended_experiment_excerpt": diagnosis_excerpt,
        "do_not_ship_without": [
            "A candidate trace set collected with the same executor and comparable scenario mix.",
            "A before/after `compare-audits` result showing signal and token deltas.",
            "A qualitative review of representative trace samples for task completion and transfer behavior.",
        ],
    }


def render_experiment_plan_markdown(plan: dict[str, Any]) -> str:
    """Render a deterministic experiment plan as Markdown."""
    scope = plan["scope"]
    care_ai_surface = plan.get("care_ai_surface") or {}
    prompt_snapshot = plan.get("prompt_snapshot") or {}
    evidence = plan["evidence_summary"]
    source = plan["source_artifacts"]
    lines = [
        f"# {plan['title']}",
        "",
        "## Scope",
        "",
        f"- Focus: `{scope['focus']}`",
        f"- Executor: `{scope['executor']}`",
        f"- Session: `{scope['session_id'] or 'all'}`",
        f"- Evidence pack: `{source['evidence_pack']}`",
    ]
    if source.get("diagnosis_report"):
        lines.append(f"- Diagnosis report: `{source['diagnosis_report']}`")
    if care_ai_surface:
        lines.extend(
            [
                "",
                "## Care AI Surface",
                "",
                f"- Prompt: `{care_ai_surface.get('prompt_name', 'unknown')}`",
                (
                    "- Session executor: "
                    f"`{care_ai_surface.get('session_executor_path', 'unknown')}`"
                ),
                f"- Route: `{care_ai_surface.get('route', 'unknown')}`",
                f"- Notes: {care_ai_surface.get('notes', 'unknown')}",
            ]
        )
    if prompt_snapshot:
        lines.extend(_render_prompt_snapshot_markdown_section(prompt_snapshot))
    lines.extend(
        [
            "",
            "## Evidence Summary",
            "",
            f"- Traces: {evidence['total_traces']}",
            f"- Spans: {evidence['total_spans']}",
            f"- Input tokens: {evidence['total_input_tokens']}",
            f"- Output tokens: {evidence['total_output_tokens']}",
            f"- Session trace distribution: `{json.dumps(evidence.get('session_trace_distribution', {}), sort_keys=True)}`",
            f"- Session input token distribution: `{json.dumps(evidence.get('session_input_token_distribution', {}), sort_keys=True)}`",
            f"- Input token distribution: `{json.dumps(evidence['llm_input_token_distribution'], sort_keys=True)}`",
            f"- Output token distribution: `{json.dumps(evidence['llm_output_token_distribution'], sort_keys=True)}`",
            f"- Audit counts: `{json.dumps(evidence['audit_counts'], sort_keys=True)}`",
            "",
            "## Hypothesis",
            "",
            plan["hypothesis"],
            "",
            "## Candidate Change",
            "",
            plan["candidate_change"],
            "",
            "## Success Metrics",
            "",
            *[f"- {metric}" for metric in plan["success_metrics"]],
            "",
            "## Guardrails",
            "",
            *[f"- {guardrail}" for guardrail in plan["guardrails"]],
            "",
            "## Verification Commands",
            "",
            "```bash",
            *plan["verification_commands"],
            "```",
            "",
            "## Do Not Ship Without",
            "",
            *[f"- {item}" for item in plan["do_not_ship_without"]],
        ]
    )
    excerpt = plan.get("diagnosis_recommended_experiment_excerpt")
    if excerpt:
        lines.extend(["", "## Diagnosis Excerpt", "", excerpt])
    return "\n".join(lines).rstrip() + "\n"


def _render_prompt_snapshot_markdown_section(snapshot: dict[str, Any]) -> list[str]:
    prompt = _dict_or_empty(snapshot.get("prompt"))
    body = _dict_or_empty(snapshot.get("body"))
    labels = _string_list(prompt.get("labels"))
    tags = _string_list(prompt.get("tags"))
    return [
        "",
        "## Prompt Snapshot",
        "",
        f"- Source: `{snapshot.get('source_path', 'unknown')}`",
        f"- Prompt: `{prompt.get('name') or 'unknown'}`",
        f"- Version: `{prompt.get('version') or 'unknown'}`",
        f"- Labels: `{', '.join(str(label) for label in labels) or 'none'}`",
        f"- Tags: `{', '.join(str(tag) for tag in tags) or 'none'}`",
        f"- Body fingerprint: `{body.get('sha256_12') or 'unknown'}`",
        f"- Body bytes: `{body.get('bytes', 0)}`",
        f"- Body lines: `{body.get('line_count', 0)}`",
        "- Raw prompt retained: `false`",
    ]


def build_candidate_handoff(
    evidence_pack_path: Path,
    *,
    diagnosis_report_path: Path | None = None,
    prompt_snapshot_path: Path | None = None,
) -> dict[str, Any]:
    """Create a deterministic handoff for a Care AI prompt/config candidate."""
    plan = build_experiment_plan(
        evidence_pack_path,
        diagnosis_report_path=diagnosis_report_path,
    )
    scope = plan["scope"]
    surface = _dict_or_empty(plan.get("care_ai_surface"))
    prompt_name = _string_or_default(surface.get("prompt_name"), "<prompt-name>")
    focus = _string_or_default(scope.get("focus"), "overview")
    executor = _string_or_default(scope.get("executor"), "unknown-executor")
    diagnosis_excerpt = _string_or_default(
        plan.get("diagnosis_recommended_experiment_excerpt"),
        "",
    )
    addendum = _candidate_prompt_addendum(
        focus=focus,
        executor=executor,
        evidence_summary=plan.get("evidence_summary"),
        diagnosis_excerpt=diagnosis_excerpt,
    )
    prompt_snapshot = _dict_or_empty(plan.get("prompt_snapshot"))
    if prompt_snapshot_path is not None:
        prompt_snapshot = load_prompt_snapshot(prompt_snapshot_path)
    return {
        "schema_version": 1,
        "purpose": "care-ai-halo-candidate-handoff",
        "source_artifacts": plan["source_artifacts"],
        "scope": scope,
        "care_ai_surface": surface,
        "prompt_snapshot": prompt_snapshot,
        "baseline_summary": plan["evidence_summary"],
        "hypothesis": plan["hypothesis"],
        "candidate_change": plan["candidate_change"],
        "proposed_candidate": {
            "change_type": "prompt_addendum",
            "prompt_name": prompt_name,
            "commit_message": f"HALO candidate: {focus} improvement for {executor}",
            "addendum": addendum,
        },
        "langfuse_commands": _candidate_langfuse_commands(
            prompt_name=prompt_name,
            focus=focus,
            executor=executor,
            evidence_pack_path=str(evidence_pack_path),
        ),
        "verification_commands": plan["verification_commands"],
        "success_metrics": plan["success_metrics"],
        "guardrails": [
            *plan["guardrails"],
            "Do not run the generated Langfuse create command without review. It is emitted with `--curl` and redirected to a local file, so this handoff is non-mutating and avoids printing the prompt body to the terminal.",
            "Use a new candidate label first. Do not move `production` or `latest` based only on metadata.",
        ],
        "diagnosis_recommended_experiment_excerpt": diagnosis_excerpt,
    }


def render_candidate_handoff_markdown(handoff: dict[str, Any]) -> str:
    """Render a deterministic candidate handoff as Markdown."""
    scope = handoff["scope"]
    surface = handoff.get("care_ai_surface") or {}
    prompt_snapshot = handoff.get("prompt_snapshot") or {}
    baseline = handoff["baseline_summary"]
    candidate = handoff["proposed_candidate"]
    lines = [
        f"# HALO Candidate Handoff: {scope['executor']} {scope['focus']}",
        "",
        "## Scope",
        "",
        f"- Focus: `{scope['focus']}`",
        f"- Executor: `{scope['executor']}`",
        f"- Session: `{scope['session_id'] or 'all'}`",
    ]
    if surface:
        lines.extend(
            [
                "",
                "## Care AI Surface",
                "",
                f"- Prompt: `{surface.get('prompt_name', 'unknown')}`",
                f"- Session executor: `{surface.get('session_executor_path', 'unknown')}`",
                f"- Route: `{surface.get('route', 'unknown')}`",
                f"- Notes: {surface.get('notes', 'unknown')}",
            ]
        )
    if prompt_snapshot:
        lines.extend(_render_prompt_snapshot_markdown_section(prompt_snapshot))
    lines.extend(
        [
            "",
            "## Baseline Evidence",
            "",
            f"- Traces: {baseline['total_traces']}",
            f"- Spans: {baseline['total_spans']}",
            f"- Input tokens: {baseline['total_input_tokens']}",
            f"- Output tokens: {baseline['total_output_tokens']}",
            f"- Session trace distribution: `{json.dumps(baseline.get('session_trace_distribution', {}), sort_keys=True)}`",
            f"- Session input token distribution: `{json.dumps(baseline.get('session_input_token_distribution', {}), sort_keys=True)}`",
            f"- Input token distribution: `{json.dumps(baseline['llm_input_token_distribution'], sort_keys=True)}`",
            f"- Audit counts: `{json.dumps(baseline['audit_counts'], sort_keys=True)}`",
            "",
            "## Hypothesis",
            "",
            handoff["hypothesis"],
            "",
            "## Candidate Change",
            "",
            handoff["candidate_change"],
            "",
            "## Proposed Prompt Addendum",
            "",
            f"- Prompt name: `{candidate['prompt_name']}`",
            f"- Commit message: `{candidate['commit_message']}`",
            "",
            "```text",
            candidate["addendum"],
            "```",
            "",
            "## Langfuse Commands",
            "",
            "These commands are non-mutating as written. The create command uses `--curl` and writes the dry-run request to an ignored local file.",
            "",
            "```bash",
            *handoff["langfuse_commands"],
            "```",
            "",
            "## Verification Commands",
            "",
            "```bash",
            *handoff["verification_commands"],
            "```",
            "",
            "## Success Metrics",
            "",
            *[f"- {metric}" for metric in handoff["success_metrics"]],
            "",
            "## Guardrails",
            "",
            *[f"- {guardrail}" for guardrail in handoff["guardrails"]],
        ]
    )
    excerpt = handoff.get("diagnosis_recommended_experiment_excerpt")
    if excerpt:
        lines.extend(["", "## Diagnosis Excerpt", "", excerpt])
    return "\n".join(lines).rstrip() + "\n"


def build_diagnosis_prompt_with_evidence(
    trace_path: Path,
    *,
    focus: DiagnosticFocus,
    executor: str | None = None,
    session_id: str | None = None,
    extra_question: str | None = None,
    include_evidence: bool = True,
    evidence_top: int = 10,
) -> str:
    """Build the exact prompt sent by `diagnose` and local pipeline runs."""
    prompt = build_diagnostic_prompt(
        focus=focus,
        executor=executor,
        session_id=session_id,
        extra_question=extra_question,
    )
    if not include_evidence:
        return prompt
    return "\n\n".join(
        [
            build_prompt_evidence_summary(trace_path, top_n=evidence_top),
            prompt,
        ]
    )


def build_pipeline_paths(
    trace_path: Path,
    output_dir: Path,
    *,
    focus: DiagnosticFocus,
    executor: str | None = None,
) -> dict[str, Path]:
    """Return stable artifact paths for a local Care AI HALO pipeline run."""
    prefix = _slugify(executor or trace_path.stem)
    focus_slug = _slugify(focus.value)
    return {
        "output_dir": output_dir,
        "evidence_pack": output_dir / f"{prefix}-{focus_slug}-evidence-pack.json",
        "diagnosis_report": output_dir / "reports" / f"{prefix}-{focus_slug}.md",
        "diagnosis_events": output_dir / "reports" / f"{prefix}-{focus_slug}.events.jsonl",
        "experiment_plan": output_dir / "reports" / f"{prefix}-{focus_slug}-experiment-plan.md",
        "candidate_handoff": output_dir / "reports" / f"{prefix}-{focus_slug}-candidate-handoff.md",
        "loop_status": output_dir / "reports" / f"{prefix}-{focus_slug}-loop-status.md",
        "manifest": output_dir / f"{prefix}-{focus_slug}-manifest.json",
    }


def build_candidate_pipeline_paths(
    evidence_pack_path: Path,
    output_dir: Path,
    *,
    focus: DiagnosticFocus | None = None,
    executor: str | None = None,
) -> dict[str, Path]:
    """Return stable local artifact paths for a Care AI HALO prompt candidate."""
    evidence_payload = _load_json_file(evidence_pack_path)
    if evidence_payload.get("purpose") != "care-ai-halo-evidence-pack":
        raise ValueError("candidate-local-pipeline requires a Care AI HALO evidence pack")
    scope = _dict_or_empty(evidence_payload.get("scope"))
    resolved_focus = focus or _diagnostic_focus_from_value(scope.get("focus"))
    resolved_executor = executor or _string_or_default(scope.get("executor"), "care-ai")
    prefix = _slugify(resolved_executor)
    focus_slug = _slugify(resolved_focus.value)
    base = output_dir / f"{prefix}-{focus_slug}"
    return {
        "output_dir": output_dir,
        "prompt_snapshot": base.with_name(f"{base.name}-current.snapshot.json"),
        "candidate_prompt": base.with_name(f"{base.name}-candidate.txt"),
        "candidate_metadata": base.with_name(f"{base.name}-candidate.metadata.json"),
        "candidate_review": base.with_name(f"{base.name}-candidate-review.md"),
        "candidate_review_json": base.with_name(f"{base.name}-candidate-review.json"),
        "runtime_plan": base.with_name(f"{base.name}-runtime-plan.md"),
        "runtime_plan_json": base.with_name(f"{base.name}-runtime-plan.json"),
        "preflight": base.with_name(f"{base.name}-candidate-preflight.md"),
        "preflight_json": base.with_name(f"{base.name}-candidate-preflight.json"),
        "loop_status": base.with_name(f"{base.name}-loop-status.md"),
        "manifest": base.with_name(f"{base.name}-candidate-local-manifest.json"),
    }


def build_candidate_local_pipeline(
    current_prompt_json_path: Path,
    evidence_pack_path: Path,
    output_dir: Path,
    *,
    environment: str,
    executor: str | None = None,
    focus: DiagnosticFocus | None = None,
    diagnosis_report_path: Path | None = None,
    trace_output_dir: Path,
) -> dict[str, Any]:
    """Write the non-mutating local artifacts needed before approved runtime push."""
    paths = build_candidate_pipeline_paths(
        evidence_pack_path,
        output_dir,
        focus=focus,
        executor=executor,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_payload = _load_json_file(evidence_pack_path)
    evidence_scope = _dict_or_empty(evidence_payload.get("scope"))
    resolved_focus = focus or _diagnostic_focus_from_value(evidence_scope.get("focus"))
    resolved_executor = executor or _string_or_default(evidence_scope.get("executor"), "")

    prompt_snapshot = build_prompt_snapshot(current_prompt_json_path)
    paths["prompt_snapshot"].write_text(
        json.dumps(prompt_snapshot, indent=2, sort_keys=True) + "\n"
    )

    candidate_text, candidate_metadata = build_candidate_prompt_artifact(
        current_prompt_json_path,
        evidence_pack_path,
        output_path=paths["candidate_prompt"],
        diagnosis_report_path=diagnosis_report_path,
    )
    paths["candidate_prompt"].write_text(candidate_text)
    paths["candidate_metadata"].write_text(
        json.dumps(candidate_metadata, indent=2, sort_keys=True) + "\n"
    )

    candidate_review = build_candidate_review(
        current_prompt_json_path,
        paths["candidate_prompt"],
        paths["candidate_metadata"],
    )
    paths["candidate_review"].write_text(render_candidate_review_markdown(candidate_review))
    paths["candidate_review_json"].write_text(
        json.dumps(candidate_review, indent=2, sort_keys=True) + "\n"
    )

    runtime_plan = build_candidate_runtime_plan(
        paths["candidate_review_json"],
        environment=environment,
        executor=resolved_executor or None,
        focus=resolved_focus,
        trace_output_dir=trace_output_dir,
    )
    paths["runtime_plan"].write_text(render_candidate_runtime_plan_markdown(runtime_plan))
    paths["runtime_plan_json"].write_text(json.dumps(runtime_plan, indent=2, sort_keys=True) + "\n")

    preflight = build_candidate_preflight(
        current_prompt_json_path,
        paths["candidate_review_json"],
        paths["runtime_plan_json"],
    )
    paths["preflight"].write_text(render_candidate_preflight_markdown(preflight))
    paths["preflight_json"].write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")

    loop_status = build_candidate_loop_status(
        evidence_pack_path=evidence_pack_path,
        candidate_review_path=paths["candidate_review_json"],
        runtime_plan_path=paths["runtime_plan_json"],
        preflight_path=paths["preflight_json"],
    )
    paths["loop_status"].write_text(render_candidate_loop_status_markdown(loop_status))

    review_prompt = _dict_or_empty(candidate_review.get("prompt"))
    fingerprints = _dict_or_empty(candidate_review.get("fingerprints"))
    manifest = {
        "schema_version": 1,
        "purpose": "care-ai-halo-candidate-local-pipeline",
        "environment": environment,
        "focus": resolved_focus.value,
        "executor": resolved_executor,
        "artifacts": {key: str(path) for key, path in paths.items() if key != "output_dir"},
        "prompt": {
            "name": review_prompt.get("name"),
            "source_version": review_prompt.get("source_version"),
            "current_sha256_12": fingerprints.get("current_sha256_12"),
            "candidate_sha256_12": fingerprints.get("candidate_sha256_12"),
        },
        "loop_status": {
            "state": loop_status["state"],
            "next_action": loop_status["next_action"],
            "external_approval_required": loop_status["external_approval_required"],
            "local_artifact_chain_complete": loop_status["local_artifact_chain_complete"],
        },
        "safety": {
            "mutated_langfuse": False,
            "raw_prompt_retained_in_manifest": False,
            "raw_trace_payloads_retained": False,
            "notes": [
                "This pipeline writes a candidate prompt file locally for human review.",
                "The manifest stores only metadata and fingerprints.",
                "Runtime mutation still requires explicit approval and `candidate-runtime-check`.",
            ],
        },
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "care-ai"


def _load_json_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def build_prompt_snapshot(prompt_json_path: Path) -> dict[str, Any]:
    """Build a metadata-only snapshot from a local Langfuse prompt JSON export."""
    raw_payload = _load_json_file(prompt_json_path)
    payload = _unwrap_lf_body(raw_payload)
    prompt_value = _extract_prompt_value(payload)
    body_sha256, body_sha256_12, body_bytes, body_line_count, body_value_type = (
        _fingerprint_prompt_body(prompt_value)
    )
    config = payload.get("config")
    config_sha256_12, config_keys = _config_snapshot(config)
    return {
        "schema_version": 1,
        "purpose": "care-ai-halo-prompt-snapshot",
        "source_path": str(prompt_json_path),
        "prompt": {
            "name": _first_string_value(
                payload,
                ["name", "promptName", "prompt_name"],
            ),
            "type": _first_string_value(payload, ["type", "promptType", "prompt_type"]),
            "version": _first_present_value(payload, ["version", "promptVersion"]),
            "labels": _string_list(payload.get("labels")),
            "tags": _string_list(payload.get("tags")),
            "commit_message": _first_string_value(
                payload,
                ["commitMessage", "commit_message"],
            ),
            "created_at": _first_string_value(payload, ["createdAt", "created_at"]),
            "updated_at": _first_string_value(payload, ["updatedAt", "updated_at"]),
        },
        "body": {
            "value_type": body_value_type,
            "sha256": body_sha256,
            "sha256_12": body_sha256_12,
            "bytes": body_bytes,
            "line_count": body_line_count,
        },
        "config": {
            "sha256_12": config_sha256_12,
            "keys": config_keys,
        },
        "safety": {
            "raw_prompt_retained": False,
            "raw_config_retained": False,
            "notes": [
                "This snapshot intentionally stores prompt/config fingerprints and metadata only.",
                "Use the source_path file locally when a human needs to inspect the prompt body.",
            ],
        },
    }


def build_candidate_prompt_artifact(
    prompt_json_path: Path,
    evidence_pack_path: Path,
    *,
    output_path: Path | None = None,
    diagnosis_report_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build a local candidate prompt body and metadata from HALO evidence."""
    prompt_payload = _unwrap_lf_body(_load_json_file(prompt_json_path))
    current_prompt_name = _first_string_value(
        prompt_payload,
        ["name", "promptName", "prompt_name"],
    )
    current_body = _extract_text_prompt_body(prompt_payload)
    handoff = build_candidate_handoff(
        evidence_pack_path,
        diagnosis_report_path=diagnosis_report_path,
    )
    candidate = _dict_or_empty(handoff.get("proposed_candidate"))
    expected_prompt_name = _string_or_default(candidate.get("prompt_name"), "<prompt-name>")
    if (
        current_prompt_name
        and expected_prompt_name != "<prompt-name>"
        and current_prompt_name != expected_prompt_name
    ):
        raise ValueError(
            "Prompt export does not match evidence-pack surface: "
            f"export={current_prompt_name!r} evidence={expected_prompt_name!r}"
        )

    addendum = _string_or_default(candidate.get("addendum"), "")
    if not addendum:
        raise ValueError("Evidence pack did not produce a candidate prompt addendum")
    candidate_body = _merge_candidate_addendum(current_body, addendum)
    current_digest = _prompt_body_digest(current_body)
    candidate_digest = _prompt_body_digest(candidate_body)
    addendum_digest = _prompt_body_digest(addendum)
    metadata = {
        "schema_version": 1,
        "purpose": "care-ai-halo-candidate-prompt-file",
        "source_artifacts": {
            "prompt_export": str(prompt_json_path),
            "evidence_pack": str(evidence_pack_path),
            "diagnosis_report": str(diagnosis_report_path) if diagnosis_report_path else None,
            "output_path": str(output_path) if output_path else None,
        },
        "scope": handoff.get("scope", {}),
        "prompt": {
            "name": current_prompt_name or expected_prompt_name,
            "type": _first_string_value(prompt_payload, ["type", "promptType", "prompt_type"]),
            "source_version": _first_present_value(prompt_payload, ["version", "promptVersion"]),
            "source_labels": _string_list(prompt_payload.get("labels")),
            "source_tags": _string_list(prompt_payload.get("tags")),
            "commit_message": _string_or_default(
                candidate.get("commit_message"),
                "HALO candidate",
            ),
        },
        "current_body": current_digest,
        "candidate_body": candidate_digest,
        "addendum": {
            **addendum_digest,
            "already_present": _candidate_addendum_present(current_body, addendum),
        },
        "safety": {
            "mutated_langfuse": False,
            "raw_prompt_retained_in_metadata": False,
            "notes": [
                "The candidate prompt body is written only to output_path for human review.",
                "This metadata intentionally stores fingerprints and prompt provenance only.",
            ],
        },
    }
    return candidate_body, metadata


def build_candidate_review(
    prompt_json_path: Path,
    candidate_prompt_path: Path,
    metadata_path: Path,
    *,
    create_curl_path: Path | None = None,
) -> dict[str, Any]:
    """Build a metadata-safe review packet for a candidate prompt artifact."""
    prompt_payload = _unwrap_lf_body(_load_json_file(prompt_json_path))
    metadata = _load_json_file(metadata_path)
    current_body = _extract_text_prompt_body(prompt_payload)
    candidate_body = candidate_prompt_path.read_text()
    current_digest = _prompt_body_digest(current_body)
    candidate_digest = _prompt_body_digest(candidate_body)
    addendum_text, marker_counts = _extract_candidate_addendum(candidate_body)
    expected_candidate = _merge_candidate_addendum(current_body, addendum_text)
    prompt_name = _first_string_value(prompt_payload, ["name", "promptName", "prompt_name"])
    metadata_prompt = _dict_or_empty(metadata.get("prompt"))
    current_metadata = _dict_or_empty(metadata.get("current_body"))
    candidate_metadata = _dict_or_empty(metadata.get("candidate_body"))
    addendum_metadata = _dict_or_empty(metadata.get("addendum"))
    safety = _dict_or_empty(metadata.get("safety"))
    curl_report = _candidate_create_curl_report(create_curl_path)
    checks = [
        _review_check(
            "metadata_purpose",
            metadata.get("purpose") == "care-ai-halo-candidate-prompt-file",
            "Metadata was produced by `candidate-prompt-file`.",
        ),
        _review_check(
            "prompt_name_matches",
            prompt_name is not None and prompt_name == metadata_prompt.get("name"),
            "Prompt export name matches candidate metadata.",
        ),
        _review_check(
            "current_prompt_hash_matches",
            current_digest.get("sha256") == current_metadata.get("sha256"),
            "Current prompt export hash matches candidate metadata.",
        ),
        _review_check(
            "candidate_prompt_hash_matches",
            candidate_digest.get("sha256") == candidate_metadata.get("sha256"),
            "Candidate prompt file hash matches candidate metadata.",
        ),
        _review_check(
            "addendum_block_once",
            marker_counts["begin"] == 1 and marker_counts["end"] == 1 and bool(addendum_text),
            "Candidate has exactly one bounded HALO addendum block.",
        ),
        _review_check(
            "candidate_is_current_plus_addendum",
            bool(addendum_text) and candidate_body == expected_candidate,
            "Candidate prompt equals current prompt plus the bounded HALO addendum.",
        ),
        _review_check(
            "addendum_hash_matches",
            bool(addendum_text)
            and _prompt_body_digest(addendum_text).get("sha256") == addendum_metadata.get("sha256"),
            "Extracted addendum hash matches candidate metadata.",
        ),
        _review_check(
            "langfuse_not_mutated",
            safety.get("mutated_langfuse") is False,
            "Candidate artifact metadata records no Langfuse mutation.",
        ),
    ]
    if create_curl_path is not None:
        checks.append(
            _review_check(
                "create_dry_run_written",
                bool(curl_report.get("exists"))
                and bool(curl_report.get("bytes"))
                and curl_report.get("contains_prompt_name") is True,
                "Dry-run create request was written locally.",
            )
        )
    ready = all(check["passed"] for check in checks)
    return {
        "schema_version": 1,
        "purpose": "care-ai-halo-candidate-review",
        "source_artifacts": {
            "prompt_export": str(prompt_json_path),
            "candidate_prompt": str(candidate_prompt_path),
            "candidate_metadata": str(metadata_path),
            "create_curl": str(create_curl_path) if create_curl_path else None,
        },
        "scope": metadata.get("scope", {}),
        "prompt": {
            "name": prompt_name or metadata_prompt.get("name"),
            "source_version": metadata_prompt.get("source_version"),
            "source_labels": metadata_prompt.get("source_labels", []),
            "commit_message": metadata_prompt.get("commit_message"),
        },
        "body_deltas": {
            "bytes": int(candidate_digest["bytes"]) - int(current_digest["bytes"]),
            "line_count": int(candidate_digest["line_count"]) - int(current_digest["line_count"]),
        },
        "fingerprints": {
            "current_sha256_12": current_digest["sha256_12"],
            "candidate_sha256_12": candidate_digest["sha256_12"],
            "addendum_sha256_12": _prompt_body_digest(addendum_text)["sha256_12"]
            if addendum_text
            else None,
        },
        "candidate_addendum": {
            "text": addendum_text,
            "line_count": _prompt_body_digest(addendum_text)["line_count"] if addendum_text else 0,
            "marker_counts": marker_counts,
        },
        "create_dry_run": curl_report,
        "checks": checks,
        "ready_for_human_review": ready,
        "safety": {
            "raw_full_prompt_retained_in_review": False,
            "mutated_langfuse": False,
            "notes": [
                "This review includes the HALO addendum text only, not the full current or candidate prompt.",
                "Open the candidate prompt file locally for full-body review before creating a Langfuse version.",
            ],
        },
    }


def render_candidate_review_markdown(review: dict[str, Any]) -> str:
    """Render a candidate review packet as Markdown without the full prompt body."""
    prompt = _dict_or_empty(review.get("prompt"))
    fingerprints = _dict_or_empty(review.get("fingerprints"))
    deltas = _dict_or_empty(review.get("body_deltas"))
    addendum = _dict_or_empty(review.get("candidate_addendum"))
    dry_run = _dict_or_empty(review.get("create_dry_run"))
    lines = [
        f"# HALO Candidate Review: {prompt.get('name', 'unknown')}",
        "",
        "## Summary",
        "",
        f"- Ready for human review: `{review['ready_for_human_review']}`",
        f"- Source version: `{prompt.get('source_version', 'unknown')}`",
        f"- Labels: `{', '.join(_string_list(prompt.get('source_labels'))) or 'none'}`",
        f"- Current prompt sha: `{fingerprints.get('current_sha256_12') or 'unknown'}`",
        f"- Candidate prompt sha: `{fingerprints.get('candidate_sha256_12') or 'unknown'}`",
        f"- Addendum sha: `{fingerprints.get('addendum_sha256_12') or 'unknown'}`",
        f"- Byte delta: `{deltas.get('bytes', 0)}`",
        f"- Line delta: `{deltas.get('line_count', 0)}`",
    ]
    if dry_run:
        lines.extend(
            [
                f"- Dry-run create file: `{dry_run.get('path') or 'none'}`",
                f"- Dry-run bytes: `{dry_run.get('bytes', 0)}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| Check | Passed | Detail |",
            "|---|---:|---|",
        ]
    )
    for check in review["checks"]:
        lines.append(f"| `{check['name']}` | `{check['passed']}` | {check['detail']} |")
    lines.extend(
        [
            "",
            "## Candidate Addendum",
            "",
            "```text",
            str(addendum.get("text") or ""),
            "```",
            "",
            "## Safety",
            "",
            "- Full current and candidate prompt bodies are not included in this review.",
            "- Langfuse was not mutated by this review.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_candidate_runtime_plan(
    candidate_review_path: Path,
    *,
    environment: str,
    executor: str | None = None,
    focus: DiagnosticFocus = DiagnosticFocus.cost,
    trace_output_dir: Path,
) -> dict[str, Any]:
    """Build a non-mutating plan for turning a reviewed candidate into trace data."""
    review = _load_json_file(candidate_review_path)
    if review.get("purpose") != "care-ai-halo-candidate-review":
        raise ValueError("candidate-runtime-plan requires candidate-review JSON input")
    if review.get("ready_for_human_review") is not True:
        raise ValueError("Candidate review is not ready for human review")

    source_artifacts = _dict_or_empty(review.get("source_artifacts"))
    prompt = _dict_or_empty(review.get("prompt"))
    scope = _dict_or_empty(review.get("scope"))
    resolved_executor = executor or _string_or_default(scope.get("executor"), "unknown-executor")
    if resolved_executor == "unknown-executor":
        raise ValueError("Executor is required when candidate review lacks scope.executor")
    focus_value = focus.value
    prompt_name = _string_or_default(prompt.get("name"), "<prompt-name>")
    candidate_prompt_path = _string_or_default(
        source_artifacts.get("candidate_prompt"),
        ".halo-careai/prompts/candidate.txt",
    )
    prompt_export_path = _string_or_default(
        source_artifacts.get("prompt_export"),
        ".halo-careai/prompts/current.json",
    )
    evidence_pack_path = _candidate_review_evidence_pack_path(source_artifacts)
    baseline_trace_path = _candidate_review_baseline_path(evidence_pack_path)
    trace_dir = trace_output_dir / _slugify(resolved_executor)
    candidate_trace_path = trace_dir / f"{_slugify(resolved_executor)}-traces.sanitized.jsonl"
    rollback_prompt_path = (
        Path(".halo-careai/prompts")
        / f"{_slugify(resolved_executor)}-{focus_value}-pre-candidate-rollback.txt"
    )
    create_json_path = (
        Path(".halo-careai/prompts")
        / f"{_slugify(resolved_executor)}-{focus_value}-candidate.created.json"
    )
    runtime_check_json_path = (
        Path(".halo-careai/prompts")
        / f"{_slugify(resolved_executor)}-{focus_value}-runtime-check.json"
    )
    rollback_json_path = (
        Path(".halo-careai/prompts")
        / f"{_slugify(resolved_executor)}-{focus_value}-rollback.created.json"
    )
    compare_path = trace_dir / f"{_slugify(resolved_executor)}-{focus_value}-compare.json"
    pull_recipe_path = trace_dir / f"{_slugify(resolved_executor)}-{focus_value}-pull.sh"
    mutation_commands = [
        "# Approval required. This moves the runtime `latest` label for the target environment.",
        "mkdir -p .halo-careai/prompts " + shlex.quote(str(trace_dir)),
        (
            "jq -r '.body.prompt // .prompt' "
            f"{shlex.quote(prompt_export_path)} > {shlex.quote(str(rollback_prompt_path))}"
        ),
        (
            "lf api prompts create "
            f"--name {shlex.quote(prompt_name)} "
            "--type text "
            f"--prompt-file {shlex.quote(candidate_prompt_path)} "
            "--labels latest,halo-candidate "
            f"--tags halo,care-ai,{shlex.quote(focus_value)} "
            f"--commit-message {shlex.quote(f'HALO runtime candidate: {focus_value} improvement for {resolved_executor}')} "
            f"--json > {shlex.quote(str(create_json_path))}"
        ),
        "# Wait about 60s for label propagation, then rehydrate prompt cache using the prompt name only, no label suffix.",
        f"# Rehydrate `{environment}` for `{prompt_name}` with the approved rehydrate workflow.",
        (
            "uv run halo-careai lf-batch-recipe "
            f"{shlex.quote(str(trace_dir))} "
            f"--environment {shlex.quote(environment)} "
            f"--executor {shlex.quote(resolved_executor)} "
            f"--focus {shlex.quote(focus_value)} "
            f"> {shlex.quote(str(pull_recipe_path))}"
        ),
        f"# Run the generated pull script after candidate traffic exists: bash {shlex.quote(str(pull_recipe_path))}",
    ]
    comparison_commands = [
        (
            "uv run halo-careai candidate-evaluate "
            f"{shlex.quote(evidence_pack_path)} "
            f"{shlex.quote(str(candidate_trace_path))} "
            f"{shlex.quote(str(trace_dir))} "
            f"--runtime-check {shlex.quote(str(runtime_check_json_path))} "
            f"--focus {shlex.quote(focus_value)} "
            f"--executor {shlex.quote(resolved_executor)}"
        ),
        (
            "uv run halo-careai compare-audits "
            f"{shlex.quote(baseline_trace_path)} "
            f"{shlex.quote(str(candidate_trace_path))} "
            f"--json > {shlex.quote(str(compare_path))}"
        ),
        (
            "uv run halo-careai evidence-pack "
            f"{shlex.quote(baseline_trace_path)} "
            f"{shlex.quote(str(trace_dir / f'{_slugify(resolved_executor)}-{focus_value}-evidence.json'))} "
            f"--candidate {shlex.quote(str(candidate_trace_path))} "
            f"--focus {shlex.quote(focus_value)} "
            f"--executor {shlex.quote(resolved_executor)}"
        ),
    ]
    rollback_commands = [
        "# Approval required. Use if candidate must be rolled back.",
        (
            "lf api prompts create "
            f"--name {shlex.quote(prompt_name)} "
            "--type text "
            f"--prompt-file {shlex.quote(str(rollback_prompt_path))} "
            "--labels latest "
            "--tags halo,care-ai,rollback "
            f"--commit-message {shlex.quote(f'HALO rollback: restore pre-candidate {resolved_executor} prompt')} "
            f"--json > {shlex.quote(str(rollback_json_path))}"
        ),
        "# Wait about 60s, then rehydrate the same prompt name in the target environment.",
    ]
    return {
        "schema_version": 1,
        "purpose": "care-ai-halo-candidate-runtime-plan",
        "approval_required": True,
        "runtime_label_warning": (
            "`halo-candidate` is a review label only. Care AI runtime uses the manifest "
            "label, currently `latest` for this executor, unless the runtime manifest is "
            "explicitly overridden."
        ),
        "source_artifacts": {
            "candidate_review": str(candidate_review_path),
            "prompt_export": prompt_export_path,
            "candidate_prompt": candidate_prompt_path,
            "evidence_pack": evidence_pack_path,
            "baseline_trace": baseline_trace_path,
        },
        "scope": {
            "environment": environment,
            "executor": resolved_executor,
            "focus": focus_value,
        },
        "prompt": {
            "name": prompt_name,
            "source_version": prompt.get("source_version"),
            "source_labels": prompt.get("source_labels", []),
        },
        "candidate_review_summary": {
            "ready_for_human_review": review.get("ready_for_human_review"),
            "fingerprints": review.get("fingerprints", {}),
            "body_deltas": review.get("body_deltas", {}),
        },
        "approved_mutation_commands": mutation_commands,
        "comparison_commands": comparison_commands,
        "rollback_commands": rollback_commands,
        "do_not_run_without": [
            "Explicit approval to move the target prompt's runtime label.",
            "A rollback prompt file created from the current prompt export.",
            "A plan to rehydrate prompt cache after both candidate push and rollback.",
            "A candidate trace collection window with comparable scenario mix.",
        ],
    }


def render_candidate_runtime_plan_markdown(plan: dict[str, Any]) -> str:
    """Render a runtime plan without prompt bodies."""
    scope = plan["scope"]
    prompt = plan["prompt"]
    summary = plan["candidate_review_summary"]
    lines = [
        f"# HALO Candidate Runtime Plan: {scope['executor']} {scope['focus']}",
        "",
        "## Warning",
        "",
        plan["runtime_label_warning"],
        "",
        "## Scope",
        "",
        f"- Environment: `{scope['environment']}`",
        f"- Executor: `{scope['executor']}`",
        f"- Focus: `{scope['focus']}`",
        f"- Prompt: `{prompt['name']}`",
        f"- Source version: `{prompt.get('source_version') or 'unknown'}`",
        "",
        "## Candidate Review",
        "",
        f"- Ready for human review: `{summary.get('ready_for_human_review')}`",
        f"- Fingerprints: `{json.dumps(summary.get('fingerprints', {}), sort_keys=True)}`",
        f"- Body deltas: `{json.dumps(summary.get('body_deltas', {}), sort_keys=True)}`",
        "",
        "## Do Not Run Without",
        "",
        *[f"- {item}" for item in plan["do_not_run_without"]],
        "",
        "## Approved Mutation Commands",
        "",
        "```bash",
        *plan["approved_mutation_commands"],
        "```",
        "",
        "## Comparison Commands",
        "",
        "```bash",
        *plan["comparison_commands"],
        "```",
        "",
        "## Rollback Commands",
        "",
        "```bash",
        *plan["rollback_commands"],
        "```",
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_candidate_preflight(
    prompt_json_path: Path,
    candidate_review_path: Path,
    runtime_plan_path: Path,
) -> dict[str, Any]:
    """Validate current prompt state before running approved candidate mutations."""
    prompt_payload = _unwrap_lf_body(_load_json_file(prompt_json_path))
    review = _load_json_file(candidate_review_path)
    runtime_plan = _load_json_file(runtime_plan_path)
    current_body = _extract_text_prompt_body(prompt_payload)
    current_digest = _prompt_body_digest(current_body)
    current_prompt_name = _first_string_value(
        prompt_payload,
        ["name", "promptName", "prompt_name"],
    )
    current_version = _first_present_value(prompt_payload, ["version", "promptVersion"])
    current_labels = _string_list(prompt_payload.get("labels"))
    review_prompt = _dict_or_empty(review.get("prompt"))
    review_fingerprints = _dict_or_empty(review.get("fingerprints"))
    runtime_prompt = _dict_or_empty(runtime_plan.get("prompt"))
    runtime_summary = _dict_or_empty(runtime_plan.get("candidate_review_summary"))
    runtime_fingerprints = _dict_or_empty(runtime_summary.get("fingerprints"))
    approved_commands = "\n".join(_string_list(runtime_plan.get("approved_mutation_commands")))
    rollback_commands = "\n".join(_string_list(runtime_plan.get("rollback_commands")))
    review_checks = [check for check in review.get("checks", []) if isinstance(check, dict)]
    failed_review_checks = [
        _string_or_default(check.get("name"), "unknown")
        for check in review_checks
        if check.get("passed") is not True
    ]
    checks = [
        _review_check(
            "review_purpose",
            review.get("purpose") == "care-ai-halo-candidate-review",
            "Review JSON was produced by `candidate-review`.",
        ),
        _review_check(
            "runtime_plan_purpose",
            runtime_plan.get("purpose") == "care-ai-halo-candidate-runtime-plan",
            "Runtime plan JSON was produced by `candidate-runtime-plan`.",
        ),
        _review_check(
            "review_ready",
            review.get("ready_for_human_review") is True,
            "Candidate review is ready for human review.",
        ),
        _review_check(
            "review_checks_passed",
            not failed_review_checks,
            "Candidate review has no failed checks.",
        ),
        _review_check(
            "prompt_name_matches_review",
            current_prompt_name is not None and current_prompt_name == review_prompt.get("name"),
            "Current prompt export name matches candidate review.",
        ),
        _review_check(
            "prompt_name_matches_runtime_plan",
            current_prompt_name is not None and current_prompt_name == runtime_prompt.get("name"),
            "Current prompt export name matches runtime plan.",
        ),
        _review_check(
            "source_version_matches_review",
            current_version == review_prompt.get("source_version"),
            "Current prompt version still matches the reviewed source version.",
        ),
        _review_check(
            "source_version_matches_runtime_plan",
            current_version == runtime_prompt.get("source_version"),
            "Current prompt version still matches the planned runtime source version.",
        ),
        _review_check(
            "current_prompt_hash_matches_review",
            current_digest.get("sha256_12") == review_fingerprints.get("current_sha256_12"),
            "Current prompt body hash still matches the reviewed source prompt.",
        ),
        _review_check(
            "runtime_plan_fingerprints_match_review",
            runtime_fingerprints == review_fingerprints,
            "Runtime plan candidate-review fingerprints match the review JSON.",
        ),
        _review_check(
            "runtime_plan_approval_gated",
            runtime_plan.get("approval_required") is True,
            "Runtime plan requires explicit approval before mutation.",
        ),
        _review_check(
            "runtime_label_warning_present",
            "halo-candidate"
            in _string_or_default(
                runtime_plan.get("runtime_label_warning"),
                "",
            )
            and "latest"
            in _string_or_default(
                runtime_plan.get("runtime_label_warning"),
                "",
            ),
            "Runtime plan explains that `halo-candidate` is not the runtime label.",
        ),
        _review_check(
            "mutation_moves_latest_with_candidate_label",
            "--labels latest,halo-candidate" in approved_commands,
            "Approved mutation commands move `latest` and retain `halo-candidate`.",
        ),
        _review_check(
            "mutation_targets_prompt",
            bool(current_prompt_name) and current_prompt_name in approved_commands,
            "Approved mutation commands target the current prompt.",
        ),
        _review_check(
            "rollback_commands_present",
            "lf api prompts create" in rollback_commands and "--labels latest" in rollback_commands,
            "Rollback commands can restore the previous runtime `latest` prompt.",
        ),
    ]
    preflight_passed = all(check["passed"] for check in checks)
    return {
        "schema_version": 1,
        "purpose": "care-ai-halo-candidate-preflight",
        "preflight_passed": preflight_passed,
        "next_action": "ready_for_approval" if preflight_passed else "refresh_candidate_artifacts",
        "source_artifacts": {
            "prompt_export": str(prompt_json_path),
            "candidate_review": str(candidate_review_path),
            "runtime_plan": str(runtime_plan_path),
        },
        "prompt": {
            "name": current_prompt_name,
            "version": current_version,
            "labels": current_labels,
            "current_sha256_12": current_digest["sha256_12"],
            "review_source_version": review_prompt.get("source_version"),
            "runtime_plan_source_version": runtime_prompt.get("source_version"),
            "review_current_sha256_12": review_fingerprints.get("current_sha256_12"),
        },
        "review": {
            "ready_for_human_review": review.get("ready_for_human_review"),
            "failed_checks": failed_review_checks,
        },
        "runtime_plan": {
            "approval_required": runtime_plan.get("approval_required"),
            "scope": runtime_plan.get("scope", {}),
        },
        "checks": checks,
        "safety": {
            "raw_prompt_retained_in_preflight": False,
            "mutated_langfuse": False,
            "notes": [
                "Preflight stores prompt hashes and metadata only.",
                "A failed preflight means fetch latest again and regenerate candidate artifacts before mutation.",
            ],
        },
    }


def render_candidate_preflight_markdown(preflight: dict[str, Any]) -> str:
    """Render a candidate preflight packet without prompt bodies."""
    prompt = _dict_or_empty(preflight.get("prompt"))
    lines = [
        f"# HALO Candidate Preflight: {prompt.get('name') or 'unknown'}",
        "",
        "## Summary",
        "",
        f"- Preflight passed: `{preflight['preflight_passed']}`",
        f"- Next action: `{preflight['next_action']}`",
        f"- Prompt version: `{prompt.get('version') or 'unknown'}`",
        f"- Prompt labels: `{', '.join(_string_list(prompt.get('labels'))) or 'none'}`",
        f"- Current prompt sha: `{prompt.get('current_sha256_12') or 'unknown'}`",
        f"- Review source version: `{prompt.get('review_source_version') or 'unknown'}`",
        f"- Runtime plan source version: `{prompt.get('runtime_plan_source_version') or 'unknown'}`",
        "",
        "## Checks",
        "",
        "| Check | Passed | Detail |",
        "|---|---:|---|",
    ]
    for check in preflight["checks"]:
        lines.append(f"| `{check['name']}` | `{check['passed']}` | {check['detail']} |")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Full prompt bodies are not included in this preflight.",
            "- Langfuse was not mutated by this preflight.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_candidate_runtime_check(
    created_prompt_path: Path,
    candidate_review_path: Path,
    runtime_plan_path: Path,
    *,
    preflight_path: Path | None = None,
) -> dict[str, Any]:
    """Verify a created Langfuse prompt version matches the reviewed runtime candidate."""
    created_payload = _unwrap_lf_body(_load_json_file(created_prompt_path))
    review = _load_json_file(candidate_review_path)
    runtime_plan = _load_json_file(runtime_plan_path)
    preflight = _load_json_file(preflight_path) if preflight_path else {}
    if review.get("purpose") != "care-ai-halo-candidate-review":
        raise ValueError("candidate-runtime-check requires candidate-review JSON input")
    if runtime_plan.get("purpose") != "care-ai-halo-candidate-runtime-plan":
        raise ValueError("candidate-runtime-check requires candidate-runtime-plan JSON input")

    created_prompt_name = _first_string_value(
        created_payload,
        ["name", "promptName", "prompt_name"],
    )
    created_version = _first_present_value(created_payload, ["version", "promptVersion"])
    created_labels = _string_list(created_payload.get("labels"))
    created_tags = _string_list(created_payload.get("tags"))
    created_body_digest: dict[str, int | str] | None = None
    created_body_with_terminal_newline_digest: dict[str, int | str] | None = None
    try:
        created_body = _extract_text_prompt_body(created_payload)
        created_body_digest = _prompt_body_digest(created_body)
        if not created_body.endswith("\n"):
            created_body_with_terminal_newline_digest = _prompt_body_digest(f"{created_body}\n")
    except ValueError:
        created_body_digest = None

    review_prompt = _dict_or_empty(review.get("prompt"))
    review_fingerprints = _dict_or_empty(review.get("fingerprints"))
    runtime_prompt = _dict_or_empty(runtime_plan.get("prompt"))
    runtime_commands = "\n".join(_string_list(runtime_plan.get("approved_mutation_commands")))
    review_candidate_sha = review_fingerprints.get("candidate_sha256_12")
    body_hash_matches_review = created_body_digest is not None and (
        created_body_digest.get("sha256_12") == review_candidate_sha
        or (
            created_body_with_terminal_newline_digest is not None
            and created_body_with_terminal_newline_digest.get("sha256_12") == review_candidate_sha
        )
    )
    preflight_checks = _dict_or_empty(preflight)
    checks = [
        _review_check(
            "created_prompt_name_matches_review",
            created_prompt_name is not None and created_prompt_name == review_prompt.get("name"),
            "Created prompt response name matches the candidate review.",
        ),
        _review_check(
            "created_prompt_name_matches_runtime_plan",
            created_prompt_name is not None and created_prompt_name == runtime_prompt.get("name"),
            "Created prompt response name matches the runtime plan.",
        ),
        _review_check(
            "created_version_is_newer_than_review_source",
            _version_greater_than(created_version, review_prompt.get("source_version")),
            "Created prompt version is newer than the reviewed source version.",
        ),
        _review_check(
            "created_labels_include_runtime_latest",
            "latest" in created_labels,
            "Created prompt response includes the runtime `latest` label.",
        ),
        _review_check(
            "created_labels_include_candidate_label",
            "halo-candidate" in created_labels,
            "Created prompt response retains the `halo-candidate` tracking label.",
        ),
        _review_check(
            "created_body_hash_matches_reviewed_candidate",
            body_hash_matches_review,
            (
                "Created prompt body hash matches the reviewed candidate prompt body, "
                "allowing Langfuse text prompts to omit one terminal newline."
            ),
        ),
        _review_check(
            "runtime_plan_moved_latest",
            "--labels latest,halo-candidate" in runtime_commands,
            "Runtime plan mutation command moved `latest` and retained `halo-candidate`.",
        ),
        _review_check(
            "review_ready",
            review.get("ready_for_human_review") is True,
            "Candidate review was ready for human review.",
        ),
    ]
    if preflight_path is not None:
        checks.append(
            _review_check(
                "preflight_passed",
                preflight_checks.get("preflight_passed") is True,
                "Preflight passed before the runtime push.",
            )
        )
    runtime_check_passed = all(check["passed"] for check in checks)
    return {
        "schema_version": 1,
        "purpose": "care-ai-halo-candidate-runtime-check",
        "runtime_check_passed": runtime_check_passed,
        "next_action": (
            "rehydrate_prompt_cache_and_collect_candidate_traces"
            if runtime_check_passed
            else "stop_and_refresh_candidate_artifacts"
        ),
        "source_artifacts": {
            "created_prompt": str(created_prompt_path),
            "candidate_review": str(candidate_review_path),
            "runtime_plan": str(runtime_plan_path),
            "preflight": str(preflight_path) if preflight_path else None,
        },
        "prompt": {
            "name": created_prompt_name,
            "version": created_version,
            "labels": created_labels,
            "tags": created_tags,
            "created_at": _string_or_default(
                _first_present_value(created_payload, ["createdAt", "created_at"]),
                "",
            )
            or None,
            "source_version": review_prompt.get("source_version"),
            "candidate_sha256_12": created_body_digest.get("sha256_12")
            if created_body_digest
            else None,
            "review_candidate_sha256_12": review_fingerprints.get("candidate_sha256_12"),
        },
        "checks": checks,
        "safety": {
            "raw_prompt_retained_in_runtime_check": False,
            "mutated_langfuse": False,
            "notes": [
                "This check reads a local Langfuse create response and stores only metadata and hashes.",
                "Run prompt-cache rehydrate only after this check passes.",
            ],
        },
    }


def render_candidate_runtime_check_markdown(runtime_check: dict[str, Any]) -> str:
    """Render runtime push verification without prompt bodies."""
    prompt = _dict_or_empty(runtime_check.get("prompt"))
    lines = [
        f"# HALO Candidate Runtime Check: {prompt.get('name') or 'unknown'}",
        "",
        "## Summary",
        "",
        f"- Runtime check passed: `{runtime_check['runtime_check_passed']}`",
        f"- Next action: `{runtime_check['next_action']}`",
        f"- Prompt version: `{prompt.get('version') or 'unknown'}`",
        f"- Prompt labels: `{', '.join(_string_list(prompt.get('labels'))) or 'none'}`",
        f"- Candidate prompt sha: `{prompt.get('candidate_sha256_12') or 'unknown'}`",
        f"- Review candidate sha: `{prompt.get('review_candidate_sha256_12') or 'unknown'}`",
        "",
        "## Checks",
        "",
        "| Check | Passed | Detail |",
        "|---|---:|---|",
    ]
    for check in runtime_check["checks"]:
        lines.append(f"| `{check['name']}` | `{check['passed']}` | {check['detail']} |")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Full prompt bodies are not included in this runtime check.",
            "- This command does not mutate Langfuse or prompt cache state.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_candidate_decision(
    comparison_path: Path,
    *,
    runtime_plan_path: Path | None = None,
    focus: DiagnosticFocus = DiagnosticFocus.cost,
    min_candidate_trace_ratio: float = 0.8,
    candidate_traffic_note: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic promote/iterate/rollback decision from comparison output."""
    comparison = _load_json_file(comparison_path)
    runtime_plan = _load_json_file(runtime_plan_path) if runtime_plan_path else {}
    trace_counts = _dict_or_empty(comparison.get("trace_counts"))
    signal_deltas = _dict_or_empty(comparison.get("signal_deltas"))
    token_distributions = _dict_or_empty(comparison.get("token_distributions"))
    baseline_traces = _int_or_zero(trace_counts.get("baseline"))
    candidate_traces = _int_or_zero(trace_counts.get("candidate"))
    required_candidate_traces = (
        ceil(baseline_traces * min_candidate_trace_ratio) if baseline_traces else 1
    )
    enough_traces = candidate_traces >= required_candidate_traces
    regressions = _candidate_regressions(signal_deltas)
    target = _candidate_target_assessment(focus=focus, comparison=comparison)
    evidence_gate = _candidate_evidence_gate(comparison)
    candidate_trace_profile = _candidate_trace_profile_from_comparison(
        comparison,
        candidate_traffic_note=candidate_traffic_note,
    )

    if not enough_traces:
        decision = "collect_more_traces"
    elif not evidence_gate["passed"]:
        decision = "collect_more_traces"
    elif regressions:
        decision = "rollback_candidate"
    elif target["passed"]:
        decision = "promote_candidate"
    else:
        decision = "iterate_candidate"

    return {
        "schema_version": 1,
        "purpose": "care-ai-halo-candidate-decision",
        "source_artifacts": {
            "comparison": str(comparison_path),
            "runtime_plan": str(runtime_plan_path) if runtime_plan_path else None,
            "baseline_trace": comparison.get("baseline_path"),
            "candidate_trace": comparison.get("candidate_path"),
        },
        "focus": focus.value,
        "decision": decision,
        "approval_required": decision in {"promote_candidate", "rollback_candidate"},
        "trace_coverage": {
            "baseline": baseline_traces,
            "candidate": candidate_traces,
            "required_candidate_traces": required_candidate_traces,
            "candidate_to_baseline_ratio": round(
                candidate_traces / baseline_traces,
                6,
            )
            if baseline_traces
            else 0,
            "passed": enough_traces,
        },
        "evidence_gate": evidence_gate,
        "candidate_trace_profile": candidate_trace_profile,
        "target_assessment": target,
        "regressions": regressions,
        "signal_deltas": signal_deltas,
        "token_distributions": token_distributions,
        "runtime_boundary": {
            "decision_scope": "candidate runtime evaluation only",
            "production_label_allowed": False,
            "detail": (
                "`promote_candidate` means deterministic candidate checks passed. "
                "It does not approve a production label, production rollout, or prod cache rehydrate."
            ),
        },
        "recommended_next_steps": _candidate_decision_next_steps(
            decision,
            runtime_plan=runtime_plan,
        ),
        "safety": {
            "raw_trace_payloads_retained": False,
            "notes": [
                "Decision uses deterministic metadata and audit deltas only.",
                "Human review is still required before candidate promotion or rollback commands are run.",
                "A promote_candidate decision does not authorize applying a production label.",
            ],
        },
    }


def render_candidate_decision_markdown(decision: dict[str, Any]) -> str:
    """Render a candidate decision packet as Markdown."""
    coverage = decision["trace_coverage"]
    target = decision["target_assessment"]
    observed = _dict_or_empty(target.get("observed"))
    runtime_boundary = _dict_or_empty(decision.get("runtime_boundary"))
    candidate_profile = _dict_or_empty(decision.get("candidate_trace_profile"))
    lines = [
        f"# HALO Candidate Decision: {decision['decision']}",
        "",
        "## Summary",
        "",
        f"- Focus: `{decision['focus']}`",
        f"- Decision: `{decision['decision']}`",
        f"- Decision scope: `{runtime_boundary.get('decision_scope', 'candidate runtime evaluation only')}`",
        f"- Production label allowed: `{runtime_boundary.get('production_label_allowed', False)}`",
        f"- Approval required: `{decision['approval_required']}`",
        f"- Trace coverage passed: `{coverage['passed']}`",
        f"- Evidence gate passed: `{decision['evidence_gate']['passed']}`",
        (
            "- Trace counts: "
            f"`baseline={coverage['baseline']}, candidate={coverage['candidate']}, "
            f"required_candidate={coverage['required_candidate_traces']}`"
        ),
        f"- Target passed: `{target['passed']}`",
        f"- Target detail: {target['detail']}",
        f"- Input p95 direction: `{observed.get('input_direction_by_p95', 'n/a')}`",
        (
            "- Session input p95 direction: "
            f"`{observed.get('session_input_direction_by_p95', 'n/a')}`"
        ),
        f"- High-token span direction: `{observed.get('high_token_llm_spans_direction', 'n/a')}`",
        "",
        "## Candidate Trace Profile",
        "",
        f"- Total traces: `{candidate_profile.get('total_traces', 'unknown')}`",
        f"- Total spans: `{candidate_profile.get('total_spans', 'unknown')}`",
        f"- Token-bearing LLM spans: `{candidate_profile.get('token_bearing_llm_span_count', 'unknown')}`",
        f"- Tool spans: `{candidate_profile.get('tool_span_count', 'unknown')}`",
        f"- Request types: `{_format_counter_summary(candidate_profile.get('request_types'))}`",
        f"- User types: `{_format_counter_summary(candidate_profile.get('user_types'))}`",
        f"- Prompt names: `{_format_counter_summary(candidate_profile.get('prompt_names'))}`",
        f"- Prompt versions: `{_format_counter_summary(candidate_profile.get('prompt_versions'))}`",
        f"- Traffic note: {candidate_profile.get('traffic_note') or 'none'}",
        "",
        "## Regressions",
        "",
    ]
    if decision["regressions"]:
        lines.extend(
            [
                "| Signal | Direction | Delta | Detail |",
                "|---|---:|---:|---|",
                *[
                    (
                        f"| `{row['signal']}` | `{row['direction']}` | "
                        f"`{row['delta']}` | {row['detail']} |"
                    )
                    for row in decision["regressions"]
                ],
            ]
        )
    else:
        lines.append("- None detected.")
    lines.extend(
        [
            "",
            "## Recommended Next Steps",
            "",
            *[f"- {step}" for step in decision["recommended_next_steps"]],
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_candidate_evaluation(
    evidence_pack_path: Path,
    candidate_trace_path: Path,
    output_dir: Path,
    *,
    runtime_plan_path: Path | None = None,
    runtime_check_path: Path | None = None,
    prompt_snapshot_path: Path | None = None,
    focus: DiagnosticFocus | None = None,
    executor: str | None = None,
    min_candidate_trace_ratio: float = 0.8,
    top_n: int = 25,
    high_input_tokens: int = 32_000,
    high_output_tokens: int = 2_000,
    model: str = "gpt-5.4-mini",
    check_model: bool = False,
    use_gocode: bool = False,
    candidate_traffic_note: str | None = None,
) -> dict[str, Any]:
    """Write the full non-mutating post-candidate evaluation bundle."""
    baseline_evidence = _load_json_file(evidence_pack_path)
    if baseline_evidence.get("purpose") != "care-ai-halo-evidence-pack":
        raise ValueError("candidate-evaluate requires a baseline evidence-pack JSON input")
    baseline_trace_raw = baseline_evidence.get("trace_path")
    if not isinstance(baseline_trace_raw, str) or not baseline_trace_raw:
        raise ValueError("Baseline evidence pack does not include a trace_path")
    baseline_trace_path = Path(baseline_trace_raw)
    if not baseline_trace_path.exists():
        raise ValueError(
            f"Baseline trace file from evidence pack does not exist: {baseline_trace_path}"
        )

    scope = _dict_or_empty(baseline_evidence.get("scope"))
    resolved_focus = focus or _diagnostic_focus_from_value(scope.get("focus"))
    resolved_executor = executor or _string_or_default(scope.get("executor"), "")
    if not resolved_executor:
        resolved_executor = _primary_executor_from_trace(candidate_trace_path, top_n=top_n)
    if not resolved_executor:
        resolved_executor = _primary_executor_from_trace(baseline_trace_path, top_n=top_n)
    output_prefix = _slugify(resolved_executor or candidate_trace_path.stem)
    focus_slug = _slugify(resolved_focus.value)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_prompt_coverage = None
    if runtime_check_path is not None:
        runtime_check = _load_json_file(runtime_check_path)
        runtime_prompt_coverage = _candidate_runtime_prompt_coverage(
            candidate_trace_path,
            runtime_check=runtime_check,
        )
        if not runtime_prompt_coverage["passed"]:
            raise ValueError(runtime_prompt_coverage["detail"])

    comparison_path = output_dir / f"{output_prefix}-{focus_slug}-compare.json"
    evidence_output_path = output_dir / f"{output_prefix}-{focus_slug}-candidate-evidence-pack.json"
    decision_json_path = output_dir / f"{output_prefix}-{focus_slug}-candidate-decision.json"
    decision_markdown_path = output_dir / f"{output_prefix}-{focus_slug}-candidate-decision.md"
    manifest_path = output_dir / f"{output_prefix}-{focus_slug}-candidate-evaluation-manifest.json"

    comparison = compare_audits(
        baseline_trace_path,
        candidate_trace_path,
        high_input_tokens=high_input_tokens,
        high_output_tokens=high_output_tokens,
        top_n=top_n,
    )
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")

    candidate_evidence = build_evidence_pack(
        baseline_trace_path,
        candidate_path=candidate_trace_path,
        prompt_snapshot_path=prompt_snapshot_path,
        focus=resolved_focus,
        executor=resolved_executor or None,
        session_id=_string_or_default(scope.get("session_id"), "") or None,
        extra_question=_string_or_default(scope.get("question"), "") or None,
        top_n=top_n,
        model=model,
        check_model=check_model,
        use_gocode=use_gocode,
        high_input_tokens=high_input_tokens,
        high_output_tokens=high_output_tokens,
    )
    evidence_output_path.write_text(json.dumps(candidate_evidence, indent=2, sort_keys=True) + "\n")

    decision = build_candidate_decision(
        comparison_path,
        runtime_plan_path=runtime_plan_path,
        focus=resolved_focus,
        min_candidate_trace_ratio=min_candidate_trace_ratio,
        candidate_traffic_note=candidate_traffic_note,
    )
    decision_json_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    decision_markdown_path.write_text(render_candidate_decision_markdown(decision))

    candidate_section = _dict_or_empty(candidate_evidence.get("candidate"))
    candidate_safety = _dict_or_empty(candidate_section.get("safety"))
    manifest = {
        "schema_version": 1,
        "purpose": "care-ai-halo-candidate-evaluation",
        "source_artifacts": {
            "baseline_evidence_pack": str(evidence_pack_path),
            "baseline_trace": str(baseline_trace_path),
            "candidate_trace": str(candidate_trace_path),
            "runtime_plan": str(runtime_plan_path) if runtime_plan_path else None,
            "runtime_check": str(runtime_check_path) if runtime_check_path else None,
            "prompt_snapshot": str(prompt_snapshot_path) if prompt_snapshot_path else None,
        },
        "scope": {
            "focus": resolved_focus.value,
            "executor": resolved_executor or None,
            "top_n": top_n,
        },
        "artifacts": {
            "comparison": str(comparison_path),
            "candidate_evidence_pack": str(evidence_output_path),
            "decision_json": str(decision_json_path),
            "decision_markdown": str(decision_markdown_path),
            "manifest": str(manifest_path),
        },
        "comparison_summary": {
            "trace_counts": comparison.get("trace_counts", {}),
            "signal_deltas": comparison.get("signal_deltas", {}),
            "token_distributions": comparison.get("token_distributions", {}),
            "session_token_distributions": comparison.get("session_token_distributions", {}),
        },
        "decision": {
            "decision": decision["decision"],
            "approval_required": decision["approval_required"],
            "trace_coverage": decision["trace_coverage"],
            "evidence_gate": decision["evidence_gate"],
            "candidate_trace_profile": decision["candidate_trace_profile"],
            "target_assessment": decision["target_assessment"],
            "regression_count": len(decision["regressions"]),
            "runtime_boundary": decision["runtime_boundary"],
        },
        "runtime_prompt_coverage": runtime_prompt_coverage,
        "safety": {
            "baseline_safe_for_metadata_only_diagnosis": _dict_or_empty(
                candidate_evidence.get("safety")
            ).get("safe_for_metadata_only_diagnosis"),
            "candidate_safe_for_metadata_only_diagnosis": candidate_safety.get(
                "safe_for_metadata_only_diagnosis"
            ),
            "raw_trace_payloads_retained": False,
            "mutated_langfuse": False,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _candidate_runtime_prompt_coverage(
    candidate_trace_path: Path,
    *,
    runtime_check: dict[str, Any],
) -> dict[str, Any]:
    """Verify candidate traces include the prompt version that was actually pushed."""
    if runtime_check.get("purpose") != "care-ai-halo-candidate-runtime-check":
        raise ValueError("candidate runtime proof requires candidate-runtime-check JSON input")
    if runtime_check.get("runtime_check_passed") is not True:
        raise ValueError("candidate runtime proof requires a passing runtime check")

    prompt = _dict_or_empty(runtime_check.get("prompt"))
    expected_name = _string_or_default(prompt.get("name"), "")
    expected_version = prompt.get("version")
    if not expected_name or expected_version is None:
        raise ValueError("Runtime check does not include a prompt name and version")

    expected_version_key = str(expected_version)
    created_at = _parse_iso_datetime(_string_or_default(prompt.get("created_at"), ""))
    total_traces: set[str] = set()
    target_trace_ids: set[str] = set()
    target_versions: Counter[str] = Counter()
    missing_version_count = 0
    target_span_count = 0
    prompt_names: Counter[str] = Counter()
    earliest_target_start: datetime | None = None

    for record in _iter_jsonl_objects(candidate_trace_path):
        trace_id = _string_or_default(record.get("trace_id"), "")
        if trace_id:
            total_traces.add(trace_id)
        attributes = _dict_or_empty(record.get("attributes"))
        prompt_name = _string_or_default(attributes.get("care_ai.prompt_name"), "")
        if prompt_name:
            prompt_names[prompt_name] += 1
        if prompt_name != expected_name:
            continue
        target_span_count += 1
        if trace_id:
            target_trace_ids.add(trace_id)
        prompt_version = attributes.get("care_ai.prompt_version")
        if prompt_version is None:
            missing_version_count += 1
        else:
            target_versions[str(prompt_version)] += 1
        span_start = _parse_iso_datetime(_string_or_default(record.get("start_time"), ""))
        if span_start is not None and (
            earliest_target_start is None or span_start < earliest_target_start
        ):
            earliest_target_start = span_start

    wrong_versions = {
        version: count
        for version, count in sorted(target_versions.items())
        if version != expected_version_key
    }
    prompt_version_matched = target_versions.get(expected_version_key, 0) > 0 and not wrong_versions
    timestamp_proof = (
        not target_versions
        and missing_version_count > 0
        and created_at is not None
        and earliest_target_start is not None
        and earliest_target_start >= created_at
    )
    passed = target_span_count > 0 and (prompt_version_matched or timestamp_proof)
    if prompt_version_matched:
        detail = "Candidate traces prove the pushed runtime prompt version."
        proof_method = "prompt_version"
    elif timestamp_proof:
        detail = (
            "Candidate traces do not include promptVersion metadata, but all target prompt "
            "spans start after the created prompt timestamp."
        )
        proof_method = "trace_timestamp_after_runtime_push"
    else:
        detail = (
            "Candidate traces do not prove the pushed runtime prompt version. "
            f"Expected {expected_name}@{expected_version_key}; "
            f"observed target versions={dict(sorted(target_versions.items())) or {}}; "
            f"missing_version_spans={missing_version_count}; "
            f"created_at={prompt.get('created_at') or 'unknown'}; "
            f"earliest_target_start={earliest_target_start.isoformat() if earliest_target_start else 'unknown'}."
        )
        proof_method = "none"
    return {
        "passed": passed,
        "detail": detail,
        "proof_method": proof_method,
        "expected_prompt_name": expected_name,
        "expected_prompt_version": expected_version,
        "expected_created_at": prompt.get("created_at"),
        "target_prompt_span_count": target_span_count,
        "target_prompt_trace_count": len(target_trace_ids),
        "total_trace_count": len(total_traces),
        "observed_target_prompt_versions": dict(sorted(target_versions.items())),
        "unexpected_target_prompt_versions": wrong_versions,
        "missing_prompt_version_span_count": missing_version_count,
        "earliest_target_start": (
            earliest_target_start.isoformat() if earliest_target_start is not None else None
        ),
        "observed_prompt_names": dict(prompt_names.most_common(10)),
    }


def _iter_jsonl_objects(path: Path) -> Iterator[dict[str, Any]]:
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        yield value


def _parse_iso_datetime(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    if "." in normalized:
        prefix, suffix = normalized.split(".", 1)
        if "+" in suffix:
            fractional, offset = suffix.split("+", 1)
            normalized = f"{prefix}.{fractional[:6]}+{offset}"
        elif "-" in suffix:
            fractional, offset = suffix.split("-", 1)
            normalized = f"{prefix}.{fractional[:6]}-{offset}"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def build_candidate_loop_status(
    *,
    evidence_pack_path: Path | None = None,
    candidate_review_path: Path | None = None,
    runtime_plan_path: Path | None = None,
    preflight_path: Path | None = None,
    runtime_check_path: Path | None = None,
    evaluation_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Audit which pieces of the local Care AI HALO loop are proven."""
    evidence_payload = _load_json_file(evidence_pack_path) if evidence_pack_path else {}
    review_payload = _load_json_file(candidate_review_path) if candidate_review_path else {}
    runtime_payload = _load_json_file(runtime_plan_path) if runtime_plan_path else {}
    preflight_payload = _load_json_file(preflight_path) if preflight_path else {}
    runtime_check_payload = _load_json_file(runtime_check_path) if runtime_check_path else {}
    evaluation_payload = (
        _load_json_file(evaluation_manifest_path) if evaluation_manifest_path else {}
    )

    stages = [
        _loop_stage_baseline(evidence_pack_path, evidence_payload),
        _loop_stage_review(candidate_review_path, review_payload),
        _loop_stage_runtime_plan(runtime_plan_path, runtime_payload),
        _loop_stage_preflight(preflight_path, preflight_payload),
        _loop_stage_runtime_check(runtime_check_path, runtime_check_payload),
        _loop_stage_evaluation(evaluation_manifest_path, evaluation_payload),
    ]
    state, next_action = _candidate_loop_state(stages, evaluation_payload)
    local_artifact_chain_complete = all(stage["status"] == "passed" for stage in stages)
    decision = _dict_or_empty(evaluation_payload.get("decision"))
    decision_value = _string_or_default(decision.get("decision"), "")
    return {
        "schema_version": 1,
        "purpose": "care-ai-halo-loop-status",
        "state": state,
        "next_action": next_action,
        "external_approval_required": state == "ready_for_approved_candidate_push"
        or bool(decision.get("approval_required")),
        "local_artifact_chain_complete": local_artifact_chain_complete,
        "loop_complete": state == "candidate_evaluated"
        and decision_value == "promote_candidate"
        and decision.get("approval_required") is False,
        "source_artifacts": {
            "evidence_pack": str(evidence_pack_path) if evidence_pack_path else None,
            "candidate_review": str(candidate_review_path) if candidate_review_path else None,
            "runtime_plan": str(runtime_plan_path) if runtime_plan_path else None,
            "preflight": str(preflight_path) if preflight_path else None,
            "runtime_check": str(runtime_check_path) if runtime_check_path else None,
            "evaluation_manifest": (
                str(evaluation_manifest_path) if evaluation_manifest_path else None
            ),
        },
        "scope": _candidate_loop_scope(
            evidence_payload=evidence_payload,
            runtime_payload=runtime_payload,
            runtime_check_payload=runtime_check_payload,
            evaluation_payload=evaluation_payload,
        ),
        "stages": stages,
        "decision": decision,
        "safety": {
            "raw_prompt_retained": False,
            "raw_trace_payloads_retained": False,
            "mutated_langfuse": False,
            "notes": [
                "Loop status reads local metadata artifacts only.",
                "A ready_for_approved_candidate_push state is not completion. It means the next step is an explicitly approved Langfuse latest-label move, prompt cache rehydrate, and candidate traffic collection.",
            ],
        },
    }


def render_candidate_loop_status_markdown(status: dict[str, Any]) -> str:
    """Render local HALO loop status without prompt or trace payloads."""
    scope = _dict_or_empty(status.get("scope"))
    lines = [
        "# HALO Care AI Loop Status",
        "",
        "## Summary",
        "",
        f"- State: `{status['state']}`",
        f"- Next action: {status['next_action']}",
        f"- External approval required: `{status['external_approval_required']}`",
        f"- Local artifact chain complete: `{status['local_artifact_chain_complete']}`",
        f"- Loop complete: `{status['loop_complete']}`",
        f"- Focus: `{scope.get('focus') or 'unknown'}`",
        f"- Executor: `{scope.get('executor') or 'unknown'}`",
        "",
        "## Stages",
        "",
        "| Stage | Status | Detail |",
        "|---|---:|---|",
    ]
    for stage in status["stages"]:
        lines.append(f"| `{stage['name']}` | `{stage['status']}` | {stage['detail']} |")
    decision = _dict_or_empty(status.get("decision"))
    if decision:
        lines.extend(
            [
                "",
                "## Decision",
                "",
                f"- Decision: `{decision.get('decision') or 'unknown'}`",
                f"- Approval required: `{decision.get('approval_required')}`",
                f"- Regression count: `{decision.get('regression_count', 0)}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- This status report reads local metadata artifacts only.",
            "- It does not mutate Langfuse, Care AI, or prompt cache state.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _candidate_regressions(signal_deltas: dict[str, Any]) -> list[dict[str, Any]]:
    guardrail_signals = [
        "planned_missing_tool_calls",
        "repeated_tool_calls",
        "transfer_without_diagnostic_tool",
        "tool_errors",
    ]
    regressions: list[dict[str, Any]] = []
    for signal in guardrail_signals:
        row = _dict_or_empty(signal_deltas.get(signal))
        if row.get("direction") != "regressed":
            continue
        regressions.append(
            {
                "signal": signal,
                "direction": row.get("direction"),
                "delta": row.get("delta", 0),
                "per_trace_delta": row.get("per_trace_delta", 0),
                "detail": "Guardrail signal increased in candidate traces.",
            }
        )
    return regressions


def _candidate_target_assessment(
    *,
    focus: DiagnosticFocus,
    comparison: dict[str, Any],
) -> dict[str, Any]:
    signal_deltas = _dict_or_empty(comparison.get("signal_deltas"))
    token_distributions = _dict_or_empty(comparison.get("token_distributions"))
    if focus == DiagnosticFocus.cost:
        input_distribution = _dict_or_empty(token_distributions.get("input"))
        session_token_distributions = _dict_or_empty(comparison.get("session_token_distributions"))
        session_input_distribution = _dict_or_empty(session_token_distributions.get("input"))
        high_token_delta = _dict_or_empty(signal_deltas.get("high_token_llm_spans"))
        p95_direction = input_distribution.get("direction_by_p95")
        session_p95_direction = session_input_distribution.get("direction_by_p95")
        high_token_direction = high_token_delta.get("direction")
        passed = (
            p95_direction == "improved"
            and session_p95_direction == "improved"
            and high_token_direction
            in {
                "improved",
                "unchanged",
            }
        )
        return {
            "passed": passed,
            "detail": (
                "Cost target requires improved LLM-span and session input-token p95, "
                "plus no increase in `high_token_llm_spans`."
            ),
            "observed": {
                "input_direction_by_p95": p95_direction,
                "session_input_direction_by_p95": session_p95_direction,
                "high_token_llm_spans_direction": high_token_direction,
                "input_p95_delta": _dict_or_empty(input_distribution.get("delta")).get("p95"),
                "session_input_p95_delta": _dict_or_empty(
                    session_input_distribution.get("delta")
                ).get("p95"),
                "high_token_llm_spans_delta": high_token_delta.get("delta"),
            },
        }
    focus_signal = {
        DiagnosticFocus.tool_errors: "tool_errors",
        DiagnosticFocus.routing: "transfer_without_diagnostic_tool",
        DiagnosticFocus.loops: "repeated_tool_calls",
    }.get(focus)
    if focus_signal is not None:
        row = _dict_or_empty(signal_deltas.get(focus_signal))
        return {
            "passed": row.get("direction") == "improved",
            "detail": f"Target requires `{focus_signal}` to improve.",
            "observed": {
                "signal": focus_signal,
                "direction": row.get("direction"),
                "delta": row.get("delta"),
            },
        }
    return {
        "passed": not _candidate_regressions(signal_deltas),
        "detail": "Overview target requires no guardrail regressions.",
        "observed": {},
    }


def _candidate_evidence_gate(comparison: dict[str, Any]) -> dict[str, Any]:
    """Reject candidate sets that are trace-count complete but semantically empty."""
    token_distributions = _dict_or_empty(comparison.get("token_distributions"))
    input_distribution = _dict_or_empty(token_distributions.get("input"))
    baseline_input = _dict_or_empty(input_distribution.get("baseline"))
    candidate_input = _dict_or_empty(input_distribution.get("candidate"))
    baseline_llm_span_count = _int_or_zero(baseline_input.get("count"))
    candidate_llm_span_count = _int_or_zero(candidate_input.get("count"))
    passed = not (baseline_llm_span_count > 0 and candidate_llm_span_count == 0)
    return {
        "passed": passed,
        "detail": (
            "Candidate trace set includes token-bearing LLM spans."
            if passed
            else (
                "Candidate trace set has zero token-bearing LLM spans while the baseline has "
                f"{baseline_llm_span_count}. Collect traces from a path that actually invokes "
                "the agent before making a promotion decision."
            )
        ),
        "baseline_llm_span_count": baseline_llm_span_count,
        "candidate_llm_span_count": candidate_llm_span_count,
    }


def _candidate_trace_profile_from_comparison(
    comparison: dict[str, Any],
    *,
    candidate_traffic_note: str | None = None,
) -> dict[str, Any]:
    candidate_trace = _string_or_default(comparison.get("candidate_path"), "")
    if not candidate_trace:
        return _empty_candidate_trace_profile(
            reason="Comparison does not include a candidate_path.",
            candidate_traffic_note=candidate_traffic_note,
        )
    candidate_path = Path(candidate_trace)
    if not candidate_path.exists():
        return _empty_candidate_trace_profile(
            reason=f"Candidate trace path does not exist: {candidate_path}",
            candidate_traffic_note=candidate_traffic_note,
        )
    return _candidate_trace_profile(
        candidate_path,
        candidate_traffic_note=candidate_traffic_note,
    )


def _empty_candidate_trace_profile(
    *,
    reason: str,
    candidate_traffic_note: str | None = None,
) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "traffic_note": candidate_traffic_note,
        "total_traces": 0,
        "total_spans": 0,
        "token_bearing_llm_span_count": 0,
        "tool_span_count": 0,
        "agent_execution_span_count": 0,
        "request_types": {},
        "user_types": {},
        "executors": {},
        "prompt_names": {},
        "prompt_versions": {},
    }


def _candidate_trace_profile(
    candidate_trace_path: Path,
    *,
    candidate_traffic_note: str | None = None,
) -> dict[str, Any]:
    """Summarize candidate traces without retaining raw payloads."""
    trace_ids: set[str] = set()
    request_types: Counter[str] = Counter()
    user_types: Counter[str] = Counter()
    executors: Counter[str] = Counter()
    prompt_names: Counter[str] = Counter()
    prompt_versions: Counter[str] = Counter()
    total_spans = 0
    token_bearing_llm_spans = 0
    tool_spans = 0
    agent_execution_spans = 0

    for record in _iter_jsonl_objects(candidate_trace_path):
        total_spans += 1
        trace_id = _string_or_default(record.get("trace_id"), "")
        if trace_id:
            trace_ids.add(trace_id)
        name = _string_or_default(record.get("name"), "")
        attributes = _dict_or_empty(record.get("attributes"))
        _count_if_present(request_types, attributes.get("care_ai.request_type"))
        _count_if_present(user_types, attributes.get("care_ai.user_type"))
        _count_if_present(executors, attributes.get("care_ai.session_executor_name"))
        _count_if_present(prompt_names, attributes.get("care_ai.prompt_name"))
        prompt_version = attributes.get("care_ai.prompt_version")
        if prompt_version is not None:
            prompt_versions[str(prompt_version)] += 1

        trace_kind = _string_or_default(attributes.get("care_ai.trace_kind"), "")
        if _has_positive_llm_tokens(attributes):
            token_bearing_llm_spans += 1
        if trace_kind == "tool_call" or name.startswith("tool_call: "):
            tool_spans += 1
        if trace_kind == "agent_execution" or name == "agent_execution":
            agent_execution_spans += 1

    return {
        "available": True,
        "trace_path": str(candidate_trace_path),
        "traffic_note": candidate_traffic_note,
        "total_traces": len(trace_ids),
        "total_spans": total_spans,
        "token_bearing_llm_span_count": token_bearing_llm_spans,
        "tool_span_count": tool_spans,
        "agent_execution_span_count": agent_execution_spans,
        "request_types": dict(request_types.most_common(10)),
        "user_types": dict(user_types.most_common(10)),
        "executors": dict(executors.most_common(10)),
        "prompt_names": dict(prompt_names.most_common(10)),
        "prompt_versions": dict(prompt_versions.most_common(10)),
        "notes": [
            "Profile is derived from sanitized candidate trace metadata only.",
            "If the traffic note is absent, do not infer authenticated shopper parity from this artifact.",
        ],
    }


def _count_if_present(counter: Counter[str], value: object) -> None:
    text = _string_or_default(value, "")
    if text:
        counter[text] += 1


def _has_positive_llm_tokens(attributes: dict[str, Any]) -> bool:
    for key in (
        "llm.token_count.prompt",
        "llm.token_count.completion",
        "llm.token_count.total",
    ):
        value = attributes.get(key)
        if isinstance(value, int) and value > 0:
            return True
        if isinstance(value, float) and value > 0:
            return True
    return False


def _format_counter_summary(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return ", ".join(f"{key}={count}" for key, count in value.items())


def _candidate_decision_next_steps(
    decision: str,
    *,
    runtime_plan: dict[str, Any],
) -> list[str]:
    rollback_commands = _string_list(runtime_plan.get("rollback_commands"))
    comparison_commands = _string_list(runtime_plan.get("comparison_commands"))
    if decision == "promote_candidate":
        return [
            "Review representative candidate traces for qualitative task completion.",
            (
                "If approved, keep the approved runtime `latest` candidate in place and "
                "continue monitoring. Do not apply a production label from this decision."
            ),
            "Archive the comparison, evidence pack, and candidate decision artifacts together.",
        ]
    if decision == "rollback_candidate":
        steps = [
            "Do not promote. Candidate guardrail regressions were detected.",
            "Run the runtime plan rollback commands after approval.",
        ]
        if rollback_commands:
            steps.append("Rollback command block is available in the runtime plan artifact.")
        return steps
    if decision == "collect_more_traces":
        steps = ["Collect more candidate traces before making a promotion decision."]
        if comparison_commands:
            steps.append("Then rerun the runtime plan comparison commands.")
        return steps
    return [
        "Do not promote yet. Target metric did not improve enough without regressions.",
        "Use the HALO evidence pack and candidate traces to refine the prompt/config change.",
    ]


def _int_or_zero(value: object) -> int:
    return value if isinstance(value, int) else 0


def _version_greater_than(candidate: object, baseline: object) -> bool:
    return isinstance(candidate, int) and isinstance(baseline, int) and candidate > baseline


def _loop_stage(
    name: str,
    *,
    path: Path | None,
    status: str,
    detail: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path) if path else None,
        "status": status,
        "detail": detail,
        "evidence": evidence or {},
    }


def _loop_stage_baseline(path: Path | None, payload: dict[str, Any]) -> dict[str, Any]:
    if path is None:
        return _loop_stage(
            "baseline_evidence",
            path=None,
            status="missing",
            detail="Run `halo-careai evidence-pack` or `halo-careai local-pipeline` on sanitized baseline traces.",
        )
    if payload.get("purpose") != "care-ai-halo-evidence-pack":
        return _loop_stage(
            "baseline_evidence",
            path=path,
            status="failed",
            detail="Input is not a Care AI HALO evidence pack.",
        )
    inspect_report = _dict_or_empty(payload.get("inspect"))
    safety = _dict_or_empty(payload.get("safety"))
    trace_count = _int_or_zero(inspect_report.get("total_traces"))
    safe = safety.get("safe_for_metadata_only_diagnosis") is True
    if trace_count < 1:
        return _loop_stage(
            "baseline_evidence",
            path=path,
            status="failed",
            detail="Evidence pack contains no traces.",
        )
    if not safe:
        return _loop_stage(
            "baseline_evidence",
            path=path,
            status="failed",
            detail="Evidence pack is not safe for metadata-only diagnosis.",
            evidence={"total_traces": trace_count},
        )
    return _loop_stage(
        "baseline_evidence",
        path=path,
        status="passed",
        detail="Baseline metadata evidence is present and sanitized.",
        evidence={
            "total_traces": trace_count,
            "total_spans": inspect_report.get("total_spans", 0),
            "audit_counts": _dict_or_empty(_dict_or_empty(payload.get("audit")).get("counts")),
        },
    )


def _loop_stage_review(path: Path | None, payload: dict[str, Any]) -> dict[str, Any]:
    if path is None:
        return _loop_stage(
            "candidate_review",
            path=None,
            status="missing",
            detail="Run `candidate-prompt-file` and `candidate-review --format json`.",
        )
    if payload.get("purpose") != "care-ai-halo-candidate-review":
        return _loop_stage(
            "candidate_review",
            path=path,
            status="failed",
            detail="Input is not a candidate review JSON artifact.",
        )
    failed_checks = [
        _string_or_default(check.get("name"), "unknown")
        for check in payload.get("checks", [])
        if isinstance(check, dict) and check.get("passed") is not True
    ]
    if payload.get("ready_for_human_review") is not True or failed_checks:
        return _loop_stage(
            "candidate_review",
            path=path,
            status="failed",
            detail="Candidate review is not ready.",
            evidence={"failed_checks": failed_checks},
        )
    prompt = _dict_or_empty(payload.get("prompt"))
    fingerprints = _dict_or_empty(payload.get("fingerprints"))
    return _loop_stage(
        "candidate_review",
        path=path,
        status="passed",
        detail="Candidate prompt review is ready for human approval.",
        evidence={
            "prompt": prompt.get("name"),
            "source_version": prompt.get("source_version"),
            "current_sha256_12": fingerprints.get("current_sha256_12"),
            "candidate_sha256_12": fingerprints.get("candidate_sha256_12"),
        },
    )


def _loop_stage_runtime_plan(path: Path | None, payload: dict[str, Any]) -> dict[str, Any]:
    if path is None:
        return _loop_stage(
            "runtime_plan",
            path=None,
            status="missing",
            detail="Run `candidate-runtime-plan --format json`.",
        )
    if payload.get("purpose") != "care-ai-halo-candidate-runtime-plan":
        return _loop_stage(
            "runtime_plan",
            path=path,
            status="failed",
            detail="Input is not a candidate runtime plan JSON artifact.",
        )
    approved_commands = "\n".join(_string_list(payload.get("approved_mutation_commands")))
    rollback_commands = "\n".join(_string_list(payload.get("rollback_commands")))
    passed = (
        payload.get("approval_required") is True
        and "--labels latest,halo-candidate" in approved_commands
        and "lf api prompts create" in rollback_commands
        and "--labels latest" in rollback_commands
    )
    if not passed:
        return _loop_stage(
            "runtime_plan",
            path=path,
            status="failed",
            detail="Runtime plan is missing approval gating, latest-label move, or rollback commands.",
        )
    return _loop_stage(
        "runtime_plan",
        path=path,
        status="passed",
        detail="Runtime plan is approval-gated and includes candidate plus rollback commands.",
        evidence={"scope": _dict_or_empty(payload.get("scope"))},
    )


def _loop_stage_preflight(path: Path | None, payload: dict[str, Any]) -> dict[str, Any]:
    if path is None:
        return _loop_stage(
            "preflight",
            path=None,
            status="missing",
            detail="Fetch `latest` again and run `candidate-preflight` before any approved mutation.",
        )
    if payload.get("purpose") != "care-ai-halo-candidate-preflight":
        return _loop_stage(
            "preflight",
            path=path,
            status="failed",
            detail="Input is not a candidate preflight JSON artifact.",
        )
    if payload.get("preflight_passed") is not True:
        failed_checks = [
            _string_or_default(check.get("name"), "unknown")
            for check in payload.get("checks", [])
            if isinstance(check, dict) and check.get("passed") is not True
        ]
        return _loop_stage(
            "preflight",
            path=path,
            status="failed",
            detail="Preflight failed. Refresh current prompt and regenerate candidate artifacts.",
            evidence={"failed_checks": failed_checks},
        )
    prompt = _dict_or_empty(payload.get("prompt"))
    return _loop_stage(
        "preflight",
        path=path,
        status="passed",
        detail="Reviewed candidate still matches the current runtime prompt version and hash.",
        evidence={
            "prompt": prompt.get("name"),
            "version": prompt.get("version"),
            "current_sha256_12": prompt.get("current_sha256_12"),
        },
    )


def _loop_stage_runtime_check(path: Path | None, payload: dict[str, Any]) -> dict[str, Any]:
    if path is None:
        return _loop_stage(
            "runtime_check",
            path=None,
            status="missing",
            detail="After approved prompt push writes created JSON, run `candidate-runtime-check`.",
        )
    if payload.get("purpose") != "care-ai-halo-candidate-runtime-check":
        return _loop_stage(
            "runtime_check",
            path=path,
            status="failed",
            detail="Input is not a candidate runtime check JSON artifact.",
        )
    if payload.get("runtime_check_passed") is not True:
        failed_checks = [
            _string_or_default(check.get("name"), "unknown")
            for check in payload.get("checks", [])
            if isinstance(check, dict) and check.get("passed") is not True
        ]
        return _loop_stage(
            "runtime_check",
            path=path,
            status="failed",
            detail="Runtime prompt push check failed. Do not rehydrate or collect candidate traces.",
            evidence={"failed_checks": failed_checks},
        )
    prompt = _dict_or_empty(payload.get("prompt"))
    return _loop_stage(
        "runtime_check",
        path=path,
        status="passed",
        detail="Created prompt version matches the reviewed candidate and includes runtime labels.",
        evidence={
            "prompt": prompt.get("name"),
            "version": prompt.get("version"),
            "labels": prompt.get("labels", []),
            "candidate_sha256_12": prompt.get("candidate_sha256_12"),
        },
    )


def _loop_stage_evaluation(path: Path | None, payload: dict[str, Any]) -> dict[str, Any]:
    if path is None:
        return _loop_stage(
            "candidate_evaluation",
            path=None,
            status="missing",
            detail="After approved candidate traffic exists, run `candidate-evaluate`.",
        )
    if payload.get("purpose") != "care-ai-halo-candidate-evaluation":
        return _loop_stage(
            "candidate_evaluation",
            path=path,
            status="failed",
            detail="Input is not a candidate evaluation manifest.",
        )
    decision = _dict_or_empty(payload.get("decision"))
    if not _string_or_default(decision.get("decision"), ""):
        return _loop_stage(
            "candidate_evaluation",
            path=path,
            status="failed",
            detail="Candidate evaluation manifest does not include a decision.",
        )
    return _loop_stage(
        "candidate_evaluation",
        path=path,
        status="passed",
        detail="Candidate traces have been evaluated against baseline.",
        evidence={
            "decision": decision.get("decision"),
            "approval_required": decision.get("approval_required"),
            "evidence_gate": decision.get("evidence_gate"),
            "regression_count": decision.get("regression_count"),
        },
    )


def _candidate_loop_state(
    stages: list[dict[str, Any]],
    evaluation_payload: dict[str, Any],
) -> tuple[str, str]:
    by_name = {stage["name"]: stage for stage in stages}
    ordered_names = [
        "baseline_evidence",
        "candidate_review",
        "runtime_plan",
        "preflight",
        "runtime_check",
        "candidate_evaluation",
    ]
    for name in ordered_names:
        stage = by_name[name]
        if stage["status"] == "failed":
            return f"{name}_failed", stage["detail"]
        if stage["status"] == "missing":
            if name == "baseline_evidence":
                return "missing_baseline_evidence", stage["detail"]
            if name == "candidate_review":
                return (
                    "baseline_ready",
                    "Create a candidate prompt file and candidate review from the baseline evidence.",
                )
            if name == "runtime_plan":
                return (
                    "candidate_review_ready",
                    "Create a runtime plan JSON so the approved mutation and rollback commands are explicit.",
                )
            if name == "preflight":
                return (
                    "runtime_plan_ready",
                    "Fetch `latest` again and run preflight before any approved mutation.",
                )
            if name == "runtime_check":
                return (
                    "ready_for_approved_candidate_push",
                    "External approval required: move the runtime `latest` label, then run `candidate-runtime-check` before rehydrate or trace collection.",
                )
            return (
                "candidate_runtime_verified",
                "Rehydrate prompt cache, collect candidate traffic, then run `candidate-evaluate`.",
            )
    decision = _dict_or_empty(evaluation_payload.get("decision"))
    decision_value = _string_or_default(decision.get("decision"), "unknown")
    if decision_value == "promote_candidate":
        return (
            "candidate_evaluated",
            (
                "Candidate meets deterministic target checks. Human approval is still required "
                "before treating this as candidate-promoted. This does not authorize a production label."
            ),
        )
    if decision_value == "rollback_candidate":
        return (
            "candidate_evaluated",
            "Candidate regressed guardrails. Run rollback commands only after explicit approval.",
        )
    if decision_value == "collect_more_traces":
        evidence_gate = _dict_or_empty(decision.get("evidence_gate"))
        if evidence_gate.get("passed") is False:
            return (
                "candidate_evaluated",
                "Candidate traces are not decision-grade. Collect token-bearing candidate traces and rerun `candidate-evaluate`.",
            )
        return (
            "candidate_evaluated",
            "Candidate trace coverage is too low. Collect more candidate traces and rerun `candidate-evaluate`.",
        )
    return (
        "candidate_evaluated",
        "Candidate did not pass deterministic promotion targets. Iterate the candidate prompt/config change.",
    )


def _candidate_loop_scope(
    *,
    evidence_payload: dict[str, Any],
    runtime_payload: dict[str, Any],
    runtime_check_payload: dict[str, Any],
    evaluation_payload: dict[str, Any],
) -> dict[str, Any]:
    evidence_scope = _dict_or_empty(evidence_payload.get("scope"))
    runtime_scope = _dict_or_empty(runtime_payload.get("scope"))
    runtime_check_prompt = _dict_or_empty(runtime_check_payload.get("prompt"))
    evaluation_scope = _dict_or_empty(evaluation_payload.get("scope"))
    return {
        "focus": evaluation_scope.get("focus")
        or runtime_scope.get("focus")
        or evidence_scope.get("focus"),
        "executor": evaluation_scope.get("executor")
        or runtime_scope.get("executor")
        or evidence_scope.get("executor"),
        "environment": runtime_scope.get("environment"),
        "runtime_prompt_version": runtime_check_prompt.get("version"),
    }


def _diagnostic_focus_from_value(value: object) -> DiagnosticFocus:
    if isinstance(value, DiagnosticFocus):
        return value
    if isinstance(value, str) and value:
        try:
            return DiagnosticFocus(value)
        except ValueError as exc:
            raise ValueError(f"Unknown diagnostic focus in evidence pack: {value!r}") from exc
    return DiagnosticFocus.overview


def _primary_executor_from_trace(trace_path: Path, *, top_n: int) -> str:
    report = inspect_halo_jsonl(trace_path, top_n=top_n)
    executors = report.get("executor_names")
    if not isinstance(executors, list) or not executors:
        return ""
    first = executors[0]
    if not isinstance(first, dict):
        return ""
    return _string_or_default(first.get("name"), "")


def _candidate_review_evidence_pack_path(source_artifacts: dict[str, Any]) -> str:
    metadata_path = source_artifacts.get("candidate_metadata")
    if not isinstance(metadata_path, str) or not metadata_path:
        return "<evidence-pack.json>"
    try:
        metadata = _load_json_file(Path(metadata_path))
    except (OSError, ValueError, json.JSONDecodeError):
        return "<evidence-pack.json>"
    metadata_sources = _dict_or_empty(metadata.get("source_artifacts"))
    return _string_or_default(metadata_sources.get("evidence_pack"), "<evidence-pack.json>")


def _candidate_review_baseline_path(evidence_pack_path: str) -> str:
    if evidence_pack_path == "<evidence-pack.json>":
        return "<baseline-traces.sanitized.jsonl>"
    try:
        evidence_pack = _load_json_file(Path(evidence_pack_path))
    except (OSError, ValueError, json.JSONDecodeError):
        return "<baseline-traces.sanitized.jsonl>"
    return _string_or_default(
        evidence_pack.get("trace_path"),
        "<baseline-traces.sanitized.jsonl>",
    )


def _extract_candidate_addendum(candidate_body: str) -> tuple[str, dict[str, int]]:
    begin_marker = "[HALO candidate addendum begins]"
    end_marker = "[HALO candidate addendum ends]"
    begin_count = candidate_body.count(begin_marker)
    end_count = candidate_body.count(end_marker)
    if begin_count != 1 or end_count != 1:
        return "", {"begin": begin_count, "end": end_count}
    after_begin = candidate_body.split(begin_marker, 1)[1]
    text = after_begin.split(end_marker, 1)[0].strip()
    return text, {"begin": begin_count, "end": end_count}


def _review_check(name: str, passed: bool, detail: str) -> dict[str, bool | str]:
    return {"name": name, "passed": passed, "detail": detail}


def _candidate_create_curl_report(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "bytes": 0, "contains_prompt_name": False}
    if not path.exists():
        return {"path": str(path), "exists": False, "bytes": 0, "contains_prompt_name": False}
    text = path.read_text(errors="replace")
    return {
        "path": str(path),
        "exists": True,
        "bytes": len(text.encode()),
        "contains_prompt_name": "c1/airo-care/" in text,
    }


def _extract_text_prompt_body(payload: dict[str, Any]) -> str:
    value = _extract_prompt_value(payload)
    if not isinstance(value, str):
        raise ValueError("Only text Langfuse prompts can be converted into candidate prompt files")
    return value


def _merge_candidate_addendum(current_body: str, addendum: str) -> str:
    if _candidate_addendum_present(current_body, addendum):
        return current_body if current_body.endswith("\n") else current_body + "\n"
    return "\n\n".join(
        [
            current_body.rstrip(),
            "[HALO candidate addendum begins]",
            addendum.strip(),
            "[HALO candidate addendum ends]\n",
        ]
    )


def _candidate_addendum_present(current_body: str, addendum: str) -> bool:
    return addendum.strip() in current_body


def _prompt_body_digest(value: str) -> dict[str, int | str]:
    raw = value.encode()
    digest = sha256(raw).hexdigest()
    return {
        "sha256": digest,
        "sha256_12": digest[:12],
        "bytes": len(raw),
        "line_count": value.count("\n") + (1 if value else 0),
    }


def load_prompt_snapshot(path: Path) -> dict[str, Any]:
    """Load an existing prompt snapshot or build one from a raw prompt export."""
    payload = _load_json_file(path)
    if payload.get("purpose") == "care-ai-halo-prompt-snapshot":
        return payload
    return build_prompt_snapshot(path)


def _unwrap_lf_body(payload: dict[str, Any]) -> dict[str, Any]:
    body = payload.get("body")
    if isinstance(body, dict) and (
        "prompt" in body
        or "name" in body
        or "version" in body
        or payload.get("ok") is not None
        or payload.get("status") is not None
    ):
        return body
    return payload


def _extract_prompt_value(payload: dict[str, Any]) -> object:
    if "prompt" not in payload:
        raise ValueError("Prompt JSON does not contain a `prompt` field")
    return payload["prompt"]


def _fingerprint_prompt_body(value: object) -> tuple[str, str, int, int, str]:
    if isinstance(value, str):
        body_text = value
        raw = value.encode()
        line_count = value.count("\n") + (1 if value else 0)
        value_type = "text"
    else:
        body_text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        raw = body_text.encode()
        line_count = 0
        value_type = type(value).__name__
    digest = sha256(raw).hexdigest()
    return digest, digest[:12], len(raw), line_count, value_type


def _config_snapshot(value: object) -> tuple[str | None, list[str]]:
    if not isinstance(value, dict):
        return None, []
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return sha256(raw).hexdigest()[:12], sorted(str(key) for key in value.keys())


def _first_present_value(payload: dict[str, Any], keys: list[str]) -> object:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _string_or_default(value: object, default: str) -> str:
    return value if isinstance(value, str) and value else default


def _dict_or_empty(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def care_ai_surface_for_executor(executor: str | None) -> dict[str, str]:
    """Return deterministic Care AI prompt and code surface for known executors."""
    if not executor:
        return {}
    surface = _CARE_AI_EXECUTOR_SURFACES.get(executor)
    return dict(surface) if surface is not None else {}


def _extract_recommended_experiment(report_text: str) -> str:
    marker_candidates = [
        "### Recommended next experiment",
        "## Recommended next experiment",
        "Recommended next experiment",
        "## Proposed prompt or configuration improvement",
        "## 4) Proposed prompt or configuration improvement",
        "## Proposed prompt / configuration improvement",
        "## 2) Proposed prompt / configuration improvement",
        "### Suggested improvement",
        "### Improvement direction",
        "### Improvement ideas",
    ]
    for marker in marker_candidates:
        start = report_text.find(marker)
        if start < 0:
            continue
        following = report_text[start:]
        if marker.startswith("## "):
            next_heading = following.find("\n## ", len(marker))
        else:
            next_heading = following.find("\n### ", len(marker))
            if next_heading < 0:
                next_heading = following.find("\n## ", len(marker))
        excerpt = following[:next_heading] if next_heading >= 0 else following
        return _sanitize_diagnosis_excerpt_text(excerpt.strip())
    return ""


def _experiment_title(*, focus: str, executor: str, audit_counts: dict[str, Any]) -> str:
    if focus == DiagnosticFocus.cost.value and audit_counts.get("high_token_llm_spans", 0):
        return f"Reduce high input-token spend in {executor}"
    if focus == DiagnosticFocus.routing.value:
        return f"Validate routing behavior for {executor}"
    if focus == DiagnosticFocus.tool_errors.value:
        return f"Reduce tool error rate for {executor}"
    if focus == DiagnosticFocus.loops.value:
        return f"Reduce repeated exploration in {executor}"
    return f"Validate HALO finding for {executor}"


def _experiment_hypothesis(*, focus: str, executor: str, audit_counts: dict[str, Any]) -> str:
    if focus == DiagnosticFocus.cost.value:
        high_token_count = audit_counts.get("high_token_llm_spans", 0)
        return (
            f"{executor} is carrying more context into `gpt-5.4` calls than needed. "
            f"Reducing or summarizing prior conversation/tool context should lower the "
            f"{high_token_count} high-token LLM spans without increasing transfers or "
            "reducing task completion."
        )
    if focus == DiagnosticFocus.routing.value:
        return (
            f"{executor} may be choosing a transfer or specialist path before enough "
            "diagnostic evidence is gathered. A prompt/config change should reduce "
            "unsupported transfers while preserving legitimate escalation."
        )
    if focus == DiagnosticFocus.tool_errors.value:
        return (
            f"{executor} can recover more consistently from failing tools if the harness "
            "adds clearer retry, fallback, or escalation policy."
        )
    if focus == DiagnosticFocus.loops.value:
        return (
            f"{executor} can reduce repeated exploration by adding stricter stop conditions "
            "after equivalent tool evidence has already been collected."
        )
    return (
        f"{executor} has a recurring trace pattern that needs validation against a "
        "candidate prompt or configuration change."
    )


def _candidate_change(
    *,
    focus: str,
    executor: str,
    care_ai_surface: dict[str, Any] | None = None,
) -> str:
    surface = care_ai_surface or {}
    surface_clause = _surface_change_clause(surface)
    if focus == DiagnosticFocus.cost.value:
        return (
            f"Create a candidate for `{executor}` that caps or summarizes non-essential prior "
            "conversation/tool history before each LLM call. Keep tool definitions and "
            f"current-turn evidence intact.{surface_clause}"
        )
    if focus == DiagnosticFocus.routing.value:
        return (
            f"Create a candidate prompt/config variant for `{executor}` that names the required "
            f"diagnostic evidence before transfer or specialist routing.{surface_clause}"
        )
    if focus == DiagnosticFocus.tool_errors.value:
        return (
            f"Create a candidate for `{executor}` that gives explicit fallback behavior for the "
            f"observed failing tools.{surface_clause}"
        )
    if focus == DiagnosticFocus.loops.value:
        return (
            f"Create a candidate for `{executor}` that stops repeating a tool call when the same "
            f"tool/input fingerprint has already been observed in the trace.{surface_clause}"
        )
    return (
        f"Create a scoped candidate for `{executor}` and compare against the baseline trace set."
        f"{surface_clause}"
    )


def _surface_change_clause(surface: dict[str, Any]) -> str:
    prompt_name = surface.get("prompt_name")
    executor_path = surface.get("session_executor_path")
    if not isinstance(prompt_name, str) or not prompt_name:
        return ""
    if isinstance(executor_path, str) and executor_path:
        return (
            f" Test the prompt `{prompt_name}` first; inspect `{executor_path}` only if the "
            "candidate requires runtime context or tool-wiring changes."
        )
    return f" Test the prompt `{prompt_name}` first."


def _candidate_prompt_addendum(
    *,
    focus: str,
    executor: str,
    evidence_summary: dict[str, Any] | None = None,
    diagnosis_excerpt: str = "",
) -> str:
    evidence_lines = _candidate_evidence_lines(
        focus=focus,
        evidence_summary=_dict_or_empty(evidence_summary),
        diagnosis_excerpt=diagnosis_excerpt,
    )
    if focus == DiagnosticFocus.cost.value:
        lines = [
            "HALO candidate: context budget and routing discipline",
            "",
            f"This candidate applies to `{executor}` only.",
            *evidence_lines,
            "Before each routing or customer response, use the smallest context needed for the current turn.",
            "Carry forward only decision-critical facts from prior turns and tool results: product or domain, verified status, user intent, unresolved blocker, and next action.",
            "Do not re-read, restate, or re-evaluate full prior tool payloads when a concise fact already supports the routing decision.",
            "When a specialist tool has returned a structured result, summarize that result into the next action instead of performing another high-context manager pass solely to justify the same route.",
            "If transfer_to_human is required, call it once with a concise factual summary. Do not add another routing pass unless new evidence changes the decision.",
            "Keep current-turn evidence, tool definitions, and required diagnostic details intact.",
        ]
        return "\n".join(lines)
    if focus == DiagnosticFocus.routing.value:
        return "\n".join(
            [
                "HALO candidate: evidence-gated routing",
                "",
                f"This candidate applies to `{executor}` only.",
                *evidence_lines,
                "Before transfer or specialist routing, identify the minimum diagnostic evidence required for that path.",
                "Use an available diagnostic tool before transfer when the tool can verify the customer issue.",
                "Transfer only when the required backend capability is unavailable, the customer explicitly needs a human task, or the diagnostic result requires human escalation.",
            ]
        )
    if focus == DiagnosticFocus.loops.value:
        return "\n".join(
            [
                "HALO candidate: repeat-call stop condition",
                "",
                f"This candidate applies to `{executor}` only.",
                *evidence_lines,
                "Do not call the same tool with equivalent input after it has already returned usable evidence.",
                "If repeated evidence is needed, state the new fact being checked before calling the tool again.",
            ]
        )
    if focus == DiagnosticFocus.tool_errors.value:
        return "\n".join(
            [
                "HALO candidate: tool-error fallback discipline",
                "",
                f"This candidate applies to `{executor}` only.",
                *evidence_lines,
                "When a tool fails, use the failure status to choose one fallback: retry with corrected input, use a lower-risk diagnostic path, or transfer with a factual summary.",
                "Do not continue tool exploration without new evidence after a repeated failure.",
            ]
        )
    return "\n".join(
        [
            "HALO candidate: scoped harness improvement",
            "",
            f"This candidate applies to `{executor}` only.",
            *evidence_lines,
            "Apply the smallest prompt or configuration change that directly targets the HALO finding.",
            "Preserve tool schemas, current-turn evidence, and existing escalation rules.",
        ]
    )


def _candidate_evidence_lines(
    *,
    focus: str,
    evidence_summary: dict[str, Any],
    diagnosis_excerpt: str,
) -> list[str]:
    lines: list[str] = []
    audit_counts = _dict_or_empty(evidence_summary.get("audit_counts"))
    input_dist = _dict_or_empty(evidence_summary.get("llm_input_token_distribution"))
    session_input_dist = _dict_or_empty(evidence_summary.get("session_input_token_distribution"))
    top_sessions = evidence_summary.get("top_sessions_by_input_tokens")
    if focus == DiagnosticFocus.cost.value:
        lines.extend(
            [
                "",
                "Observed HALO evidence to address:",
                (
                    f"- `{audit_counts.get('high_token_llm_spans', 0)}` high-input-token "
                    f"LLM spans across `{evidence_summary.get('total_traces', 0)}` traces."
                ),
                (
                    f"- Input-token median `{input_dist.get('median', 0)}`, "
                    f"p95 `{input_dist.get('p95', 0)}`, max `{input_dist.get('max', 0)}`."
                ),
                (
                    f"- Session input-token median `{session_input_dist.get('median', 0)}`, "
                    f"p95 `{session_input_dist.get('p95', 0)}`, max "
                    f"`{session_input_dist.get('max', 0)}`."
                ),
                (
                    f"- Tool errors `{audit_counts.get('tool_errors', 0)}`, repeated tool calls "
                    f"`{audit_counts.get('repeated_tool_calls', 0)}`, planned-missing tool calls "
                    f"`{audit_counts.get('planned_missing_tool_calls', 0)}`."
                ),
            ]
        )
        if isinstance(top_sessions, list) and top_sessions and isinstance(top_sessions[0], dict):
            top_session = top_sessions[0]
            lines.append(
                f"- Highest-cost session used `{top_session.get('input_tokens', 0)}` input "
                f"tokens across `{top_session.get('trace_count', 0)}` traces and "
                f"`{top_session.get('span_count', 0)}` spans."
            )
    elif focus == DiagnosticFocus.routing.value:
        lines.extend(
            [
                "",
                "Observed HALO evidence to address:",
                (
                    "- Transfer-without-diagnostic-tool signals "
                    f"`{audit_counts.get('transfer_without_diagnostic_tool', 0)}`."
                ),
                f"- Planned-missing tool calls `{audit_counts.get('planned_missing_tool_calls', 0)}`.",
            ]
        )
    elif focus == DiagnosticFocus.loops.value:
        lines.extend(
            [
                "",
                "Observed HALO evidence to address:",
                f"- Repeated tool-call signals `{audit_counts.get('repeated_tool_calls', 0)}`.",
            ]
        )
    elif focus == DiagnosticFocus.tool_errors.value:
        lines.extend(
            [
                "",
                "Observed HALO evidence to address:",
                f"- Tool-error signals `{audit_counts.get('tool_errors', 0)}`.",
            ]
        )
    direction_lines = _diagnosis_direction_lines(diagnosis_excerpt)
    if direction_lines:
        lines.extend(["", "HALO diagnosis direction:"])
        lines.extend(f"- {line}" for line in direction_lines)
    if lines:
        lines.append("")
    return lines


def _diagnosis_direction_lines(report_excerpt: str, *, max_lines: int = 4) -> list[str]:
    if not report_excerpt:
        return []
    lines: list[str] = []
    interesting = False
    for raw_line in report_excerpt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if (
            "improvement direction" in lower
            or "suggested improvement" in lower
            or "improvement ideas" in lower
        ):
            interesting = True
            continue
        if line.startswith("#") or line.startswith("|"):
            continue
        if line.startswith("-"):
            line = line[1:].strip()
            lower = line.lower()
        if not line:
            continue
        if any(
            blocked in lower
            for blocked in [
                "likely surface",
                "session-executor",
                "care-ai-agents/",
                "**prompt:**",
                "owning surface",
                "surface to change",
            ]
        ):
            continue
        if not interesting and not any(
            token in lower
            for token in [
                "reduce",
                "summar",
                "limit",
                "context",
                "routing",
                "diagnostic",
                "transfer",
                "candidate",
            ]
        ):
            continue
        cleaned = _sanitize_diagnosis_addendum_line(line)
        if cleaned:
            lines.append(cleaned)
        if len(lines) >= max_lines:
            break
    return lines


def _sanitize_diagnosis_addendum_line(line: str) -> str:
    cleaned = re.sub(r"https?://\S+", "[redacted-url]", line)
    cleaned = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[redacted-email]", cleaned)
    cleaned = re.sub(r"`([0-9a-f]{16,})`", "`redacted-id`", cleaned)
    cleaned = re.sub(r"\b[0-9a-f]{24,}\b", "redacted-id", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:240].rstrip()


def _sanitize_diagnosis_excerpt_text(text: str) -> str:
    sanitized_lines = [
        _sanitize_diagnosis_addendum_line(line) if line.strip() else ""
        for line in text.splitlines()
    ]
    return "\n".join(sanitized_lines).strip()


def _candidate_langfuse_commands(
    *,
    prompt_name: str,
    focus: str,
    executor: str,
    evidence_pack_path: str,
) -> list[str]:
    current_path = f".halo-careai/prompts/{_slugify(executor)}-current.json"
    snapshot_path = f".halo-careai/prompts/{_slugify(executor)}-current.snapshot.json"
    candidate_path = f".halo-careai/prompts/{_slugify(executor)}-{_slugify(focus)}-candidate.txt"
    candidate_metadata_path = (
        f".halo-careai/prompts/{_slugify(executor)}-{_slugify(focus)}-candidate.metadata.json"
    )
    create_curl_path = (
        f".halo-careai/prompts/{_slugify(executor)}-{_slugify(focus)}-candidate.create.curl"
    )
    review_path = f".halo-careai/prompts/{_slugify(executor)}-{_slugify(focus)}-candidate-review.md"
    review_json_path = (
        f".halo-careai/prompts/{_slugify(executor)}-{_slugify(focus)}-candidate-review.json"
    )
    runtime_plan_path = (
        f".halo-careai/prompts/{_slugify(executor)}-{_slugify(focus)}-runtime-plan.md"
    )
    runtime_plan_json_path = (
        f".halo-careai/prompts/{_slugify(executor)}-{_slugify(focus)}-runtime-plan.json"
    )
    preflight_path = (
        f".halo-careai/prompts/{_slugify(executor)}-{_slugify(focus)}-candidate-preflight.md"
    )
    preflight_json_path = (
        f".halo-careai/prompts/{_slugify(executor)}-{_slugify(focus)}-candidate-preflight.json"
    )
    preflight_status_path = (
        f".halo-careai/prompts/{_slugify(executor)}-{_slugify(focus)}-loop-status.md"
    )
    created_json_path = (
        f".halo-careai/prompts/{_slugify(executor)}-{_slugify(focus)}-candidate.created.json"
    )
    runtime_check_path = (
        f".halo-careai/prompts/{_slugify(executor)}-{_slugify(focus)}-runtime-check.md"
    )
    runtime_check_json_path = (
        f".halo-careai/prompts/{_slugify(executor)}-{_slugify(focus)}-runtime-check.json"
    )
    candidate_runtime_dir = f".halo-careai/export/candidate-runtime/{_slugify(executor)}"
    candidate_trace_path = f"{candidate_runtime_dir}/{_slugify(executor)}-traces.sanitized.jsonl"
    evaluation_manifest_path = (
        f"{candidate_runtime_dir}/{_slugify(executor)}-{_slugify(focus)}"
        "-candidate-evaluation-manifest.json"
    )
    evaluation_status_path = (
        f"{candidate_runtime_dir}/{_slugify(executor)}-{_slugify(focus)}-loop-status.md"
    )
    return [
        "mkdir -p .halo-careai/prompts",
        (
            f"lf api prompts get {shlex.quote(prompt_name)} "
            f"--label latest --json > {shlex.quote(current_path)}"
        ),
        (
            "uv run halo-careai prompt-snapshot "
            f"{shlex.quote(current_path)} {shlex.quote(snapshot_path)}"
        ),
        (
            "uv run halo-careai candidate-prompt-file "
            f"{shlex.quote(current_path)} "
            f"{shlex.quote(evidence_pack_path)} "
            f"{shlex.quote(candidate_path)} "
            f"--metadata-output {shlex.quote(candidate_metadata_path)}"
        ),
        f"# Review {shlex.quote(candidate_path)} before running any command that creates a Langfuse prompt version.",
        (
            "lf api prompts create "
            f"--name {shlex.quote(prompt_name)} "
            f"--type text "
            f"--prompt-file {shlex.quote(candidate_path)} "
            f"--labels halo-candidate "
            f"--tags halo,care-ai,{shlex.quote(focus)} "
            f"--commit-message {shlex.quote(f'HALO candidate: {focus} improvement for {executor}')} "
            f"--curl > {shlex.quote(create_curl_path)}"
        ),
        (
            "uv run halo-careai candidate-review "
            f"{shlex.quote(current_path)} "
            f"{shlex.quote(candidate_path)} "
            f"{shlex.quote(candidate_metadata_path)} "
            f"{shlex.quote(review_path)} "
            f"--create-curl {shlex.quote(create_curl_path)}"
        ),
        (
            "uv run halo-careai candidate-review "
            f"{shlex.quote(current_path)} "
            f"{shlex.quote(candidate_path)} "
            f"{shlex.quote(candidate_metadata_path)} "
            f"{shlex.quote(review_json_path)} "
            f"--create-curl {shlex.quote(create_curl_path)} "
            "--format json"
        ),
        (
            "uv run halo-careai candidate-runtime-plan "
            f"{shlex.quote(review_json_path)} "
            f"{shlex.quote(runtime_plan_path)} "
            f"--environment test "
            f"--executor {shlex.quote(executor)} "
            f"--focus {shlex.quote(focus)}"
        ),
        (
            "uv run halo-careai candidate-runtime-plan "
            f"{shlex.quote(review_json_path)} "
            f"{shlex.quote(runtime_plan_json_path)} "
            f"--environment test "
            f"--executor {shlex.quote(executor)} "
            f"--focus {shlex.quote(focus)} "
            "--format json"
        ),
        "# Fetch latest again before preflight so stale candidate artifacts are caught.",
        (
            f"lf api prompts get {shlex.quote(prompt_name)} "
            f"--label latest --json > {shlex.quote(current_path)}"
        ),
        (
            "uv run halo-careai candidate-preflight "
            f"{shlex.quote(current_path)} "
            f"{shlex.quote(review_json_path)} "
            f"{shlex.quote(runtime_plan_json_path)} "
            f"{shlex.quote(preflight_path)}"
        ),
        (
            "uv run halo-careai candidate-preflight "
            f"{shlex.quote(current_path)} "
            f"{shlex.quote(review_json_path)} "
            f"{shlex.quote(runtime_plan_json_path)} "
            f"{shlex.quote(preflight_json_path)} "
            "--format json"
        ),
        (
            "uv run halo-careai loop-status "
            f"--evidence-pack {shlex.quote(evidence_pack_path)} "
            f"--candidate-review {shlex.quote(review_json_path)} "
            f"--runtime-plan {shlex.quote(runtime_plan_json_path)} "
            f"--preflight {shlex.quote(preflight_json_path)} "
            f"--output {shlex.quote(preflight_status_path)}"
        ),
        "# After the approved runtime push writes the created prompt JSON:",
        (
            "uv run halo-careai candidate-runtime-check "
            f"{shlex.quote(created_json_path)} "
            f"{shlex.quote(review_json_path)} "
            f"{shlex.quote(runtime_plan_json_path)} "
            f"{shlex.quote(runtime_check_path)} "
            f"--preflight {shlex.quote(preflight_json_path)}"
        ),
        (
            "uv run halo-careai candidate-runtime-check "
            f"{shlex.quote(created_json_path)} "
            f"{shlex.quote(review_json_path)} "
            f"{shlex.quote(runtime_plan_json_path)} "
            f"{shlex.quote(runtime_check_json_path)} "
            f"--preflight {shlex.quote(preflight_json_path)} "
            "--format json"
        ),
        "# After approved candidate traffic exists and the runtime pull script has produced sanitized traces:",
        (
            "uv run halo-careai candidate-evaluate "
            f"{shlex.quote(evidence_pack_path)} "
            f"{shlex.quote(candidate_trace_path)} "
            f"{shlex.quote(candidate_runtime_dir)} "
            f"--runtime-plan {shlex.quote(runtime_plan_json_path)} "
            f"--runtime-check {shlex.quote(runtime_check_json_path)} "
            f"--focus {shlex.quote(focus)} "
            f"--executor {shlex.quote(executor)}"
        ),
        (
            "uv run halo-careai loop-status "
            f"--evidence-pack {shlex.quote(evidence_pack_path)} "
            f"--candidate-review {shlex.quote(review_json_path)} "
            f"--runtime-plan {shlex.quote(runtime_plan_json_path)} "
            f"--preflight {shlex.quote(preflight_json_path)} "
            f"--runtime-check {shlex.quote(runtime_check_json_path)} "
            f"--evaluation {shlex.quote(evaluation_manifest_path)} "
            f"--output {shlex.quote(evaluation_status_path)}"
        ),
    ]


def _success_metrics(*, focus: str) -> list[str]:
    metrics = [
        "Candidate trace set has comparable trace count and scenario mix.",
        "No increase in `tool_errors`, `planned_missing_tool_calls`, or repeated tool-call signals.",
    ]
    if focus == DiagnosticFocus.cost.value:
        metrics.extend(
            [
                "Lower total `inference.llm.input_tokens` per trace.",
                "Lower median and p95 `inference.llm.input_tokens` across LLM spans.",
                "Lower count of `high_token_llm_spans` at the same audit threshold.",
                "No increase in `transfer_without_diagnostic_tool` rate.",
            ]
        )
    elif focus == DiagnosticFocus.routing.value:
        metrics.append("Lower unsupported transfer or misroute rate in trace review.")
    elif focus == DiagnosticFocus.tool_errors.value:
        metrics.append("Lower `tool_errors` count and clear recovery behavior in trace review.")
    elif focus == DiagnosticFocus.loops.value:
        metrics.append("Lower `repeated_tool_calls` count and shorter trace paths.")
    return metrics


def _guardrails(*, focus: str) -> list[str]:
    guardrails = [
        "Do not change production prompts based only on sanitized metadata.",
        "Do not recommend a model swap unless trace evidence isolates model capability as the failure mode.",
    ]
    if focus == DiagnosticFocus.cost.value:
        guardrails.append(
            "Do not remove current-turn evidence or tool schemas while trimming context."
        )
    return guardrails


def _verification_commands(*, executor: str, focus: str, baseline_path: str) -> list[str]:
    candidate_path = ".halo-careai/candidate-traces.sanitized.jsonl"
    return [
        f"uv run halo-careai inspect {shlex.quote(baseline_path)} --json",
        f"uv run halo-careai audit {shlex.quote(baseline_path)} --json",
        (f"uv run halo-careai compare-audits {shlex.quote(baseline_path)} {candidate_path} --json"),
        (
            "uv run halo-careai evidence-pack "
            f"{shlex.quote(baseline_path)} .halo-careai/evidence/{executor}-{focus}-candidate.json "
            f"--candidate {candidate_path} --focus {shlex.quote(focus)} --executor {shlex.quote(executor)}"
        ),
    ]


def build_prompt_evidence_summary(
    trace_path: Path,
    *,
    top_n: int = 10,
    high_input_tokens: int = 32_000,
    high_output_tokens: int = 2_000,
) -> str:
    """Return a compact metadata-only evidence summary for the HALO prompt."""
    inspect_report = inspect_halo_jsonl(trace_path, top_n=top_n)
    audit_report = _metadata_safe_audit_report(
        audit_halo_jsonl(
            trace_path,
            high_input_tokens=high_input_tokens,
            high_output_tokens=high_output_tokens,
            top_n=top_n,
        )
    )
    safety_report = trace_safety_report(trace_path, top_n=top_n)
    payload = {
        "inspect": {
            "total_traces": inspect_report["total_traces"],
            "total_spans": inspect_report["total_spans"],
            "error_trace_count": inspect_report["error_trace_count"],
            "session_count": inspect_report["session_count"],
            "ucid_count": inspect_report["ucid_count"],
            "total_input_tokens": inspect_report["total_input_tokens"],
            "total_output_tokens": inspect_report["total_output_tokens"],
            "session_trace_distribution": inspect_report["session_trace_distribution"],
            "session_span_distribution": inspect_report["session_span_distribution"],
            "session_input_token_distribution": inspect_report["session_input_token_distribution"],
            "session_output_token_distribution": inspect_report[
                "session_output_token_distribution"
            ],
            "top_sessions_by_input_tokens": inspect_report["top_sessions_by_input_tokens"],
            "llm_input_token_distribution": inspect_report["llm_input_token_distribution"],
            "llm_output_token_distribution": inspect_report["llm_output_token_distribution"],
            "model_names": inspect_report["model_names"],
            "executor_names": inspect_report["executor_names"],
            "tool_names": inspect_report["tool_names"],
            "planned_tool_names": inspect_report["planned_tool_names"],
            "status_codes": inspect_report["status_codes"],
        },
        "audit_counts": audit_report["counts"],
        "audit_examples": {
            "planned_missing_tool_calls": audit_report["planned_missing_tool_calls"],
            "repeated_tool_calls": audit_report["repeated_tool_calls"],
            "transfer_without_diagnostic_tool": audit_report["transfer_without_diagnostic_tool"],
            "tool_errors": audit_report["tool_errors"],
            "high_token_llm_spans": audit_report["high_token_llm_spans"],
        },
        "safety": {
            "safe_for_metadata_only_diagnosis": safety_report["safe_for_metadata_only_diagnosis"],
            "raw_payload_attribute_count": safety_report["raw_payload_attribute_count"],
            "raw_status_message_count": safety_report["raw_status_message_count"],
            "possible_raw_identifier_attribute_count": safety_report[
                "possible_raw_identifier_attribute_count"
            ],
        },
    }
    return "\n".join(
        [
            "Deterministic local evidence summary (metadata-only, generated before HALO):",
            "```json",
            json.dumps(payload, indent=2, sort_keys=True),
            "```",
            "Use this summary as the starting point. In summary tool mode, detail tools like view_trace, view_spans, search_trace, search_span, and run_code are intentionally unavailable.",
        ]
    )


def sanitize_halo_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    keep_status_message: bool = False,
    keep_identifiers: bool = False,
) -> dict[str, Any]:
    """Write a metadata-only trace file suitable for lower-risk HALO diagnosis."""
    input_spans = _iter_halo_spans(input_path)
    trace_id_map = {
        trace_id: _hashed_identifier(trace_id, length=32)
        for trace_id in {span.trace_id for span in input_spans}
    }
    span_id_map = {
        span_id: _hashed_identifier(span_id, length=16)
        for span_id in {span.span_id for span in input_spans}
    }
    spans: list[SpanRecord] = []
    redacted_attribute_count = 0
    redacted_identifier_count = 0
    redacted_status_message_count = 0

    for span in input_spans:
        data = span.model_dump()
        if not keep_identifiers:
            data["trace_id"] = trace_id_map[span.trace_id]
            data["span_id"] = span_id_map[span.span_id]
            if span.parent_span_id:
                data["parent_span_id"] = span_id_map.get(
                    span.parent_span_id,
                    _hashed_identifier(span.parent_span_id, length=16),
                )

        attrs = dict(data["attributes"])
        for key in _SENSITIVE_ATTRIBUTE_KEYS:
            if key not in attrs:
                continue
            sha, byte_count = _fingerprint_value(attrs.pop(key))
            safe_key = key.replace(".", "_")
            attrs[f"care_ai.redacted.{safe_key}.sha256_12"] = sha
            attrs[f"care_ai.redacted.{safe_key}.bytes"] = byte_count
            redacted_attribute_count += 1
        if not keep_identifiers:
            for key in _SENSITIVE_IDENTIFIER_ATTRIBUTE_KEYS:
                value = attrs.get(key)
                if not isinstance(value, str) or not value:
                    continue
                attrs[key] = f"redacted:{_hashed_identifier(value, length=12)}"
                redacted_identifier_count += 1

        status = dict(data["status"])
        message = status.get("message")
        if not keep_status_message and isinstance(message, str) and message:
            sha, byte_count = _fingerprint_value(message)
            attrs["care_ai.redacted.status_message.sha256_12"] = sha
            attrs["care_ai.redacted.status_message.bytes"] = byte_count
            status["message"] = ""
            redacted_status_message_count += 1

        data["attributes"] = attrs
        data["status"] = status
        spans.append(SpanRecord.model_validate(data))

    count = write_halo_jsonl(spans, output_path)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "spans_written": count,
        "redacted_attribute_count": redacted_attribute_count,
        "redacted_identifier_count": redacted_identifier_count,
        "redacted_status_message_count": redacted_status_message_count,
        "redacted_attribute_keys": list(_SENSITIVE_ATTRIBUTE_KEYS),
        "redacted_identifier_attribute_keys": list(_SENSITIVE_IDENTIFIER_ATTRIBUTE_KEYS),
        "kept_identifiers": keep_identifiers,
        "kept_status_message": keep_status_message,
    }


def _compare_count(
    baseline_count: int,
    candidate_count: int,
    baseline_traces: int,
    candidate_traces: int,
) -> dict[str, float | int | str]:
    baseline_rate = baseline_count / baseline_traces if baseline_traces else 0.0
    candidate_rate = candidate_count / candidate_traces if candidate_traces else 0.0
    delta = candidate_count - baseline_count
    rate_delta = candidate_rate - baseline_rate
    if rate_delta < 0:
        direction = "improved"
    elif rate_delta > 0:
        direction = "regressed"
    else:
        direction = "unchanged"
    return {
        "baseline": baseline_count,
        "candidate": candidate_count,
        "delta": delta,
        "baseline_per_trace": round(baseline_rate, 6),
        "candidate_per_trace": round(candidate_rate, 6),
        "per_trace_delta": round(rate_delta, 6),
        "direction": direction,
    }


def _token_distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "total": 0,
            "min": 0,
            "median": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "mean": 0,
        }

    ordered = sorted(values)
    total = sum(ordered)
    return {
        "count": len(ordered),
        "total": total,
        "min": ordered[0],
        "median": _median(ordered),
        "p95": _nearest_rank_percentile(ordered, 95),
        "p99": _nearest_rank_percentile(ordered, 99),
        "max": ordered[-1],
        "mean": round(total / len(ordered), 2),
    }


def _median(ordered_values: list[int]) -> float | int:
    midpoint = len(ordered_values) // 2
    if len(ordered_values) % 2:
        return ordered_values[midpoint]
    return round((ordered_values[midpoint - 1] + ordered_values[midpoint]) / 2, 2)


def _nearest_rank_percentile(ordered_values: list[int], percentile: int) -> int:
    index = max(0, min(len(ordered_values) - 1, ceil(percentile / 100 * len(ordered_values)) - 1))
    return ordered_values[index]


def _compare_token_distribution(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    numeric_keys = ["count", "total", "min", "median", "p95", "p99", "max", "mean"]
    deltas = {
        key: round(float(candidate.get(key, 0)) - float(baseline.get(key, 0)), 2)
        for key in numeric_keys
    }
    p95_delta = deltas["p95"]
    if p95_delta < 0:
        direction = "improved"
    elif p95_delta > 0:
        direction = "regressed"
    else:
        direction = "unchanged"
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": deltas,
        "direction_by_p95": direction,
    }


def _iter_halo_spans(trace_path: Path) -> list[SpanRecord]:
    spans: list[SpanRecord] = []
    with trace_path.open("rb") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if stripped:
                spans.append(SpanRecord.model_validate_json(stripped))
    return spans


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, str) and entry]


def _trace_executors(spans: list[SpanRecord]) -> list[str]:
    executors = {
        value
        for span in spans
        if isinstance(value := span.attributes.get("care_ai.session_executor_name"), str) and value
    }
    return sorted(executors)


def _session_key(attrs: dict[str, Any]) -> str:
    for key in ("langfuse.session_id", "care_ai.conversation_id"):
        value = attrs.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown-session"


def _session_summary(
    sessions: dict[str, dict[str, Any]],
    *,
    top_n: int,
) -> dict[str, Any]:
    traces_per_session: list[int] = []
    spans_per_session: list[int] = []
    input_tokens_per_session: list[int] = []
    output_tokens_per_session: list[int] = []
    reported_tools_per_session: list[int] = []
    executed_tools_per_session: list[int] = []
    planned_tools_per_session: list[int] = []
    top_rows: list[dict[str, Any]] = []
    for session_id, session in sessions.items():
        trace_count = len(session["trace_ids"])
        span_count = int(session["span_count"])
        input_tokens = int(session["input_tokens"])
        output_tokens = int(session["output_tokens"])
        reported_tools = int(session["reported_tool_calls"])
        executed_tools = int(session["executed_tool_calls"])
        planned_tools = int(session["planned_tool_calls"])
        traces_per_session.append(trace_count)
        spans_per_session.append(span_count)
        input_tokens_per_session.append(input_tokens)
        output_tokens_per_session.append(output_tokens)
        reported_tools_per_session.append(reported_tools)
        executed_tools_per_session.append(executed_tools)
        planned_tools_per_session.append(planned_tools)
        top_rows.append(
            {
                "session_sha256_12": _fingerprint_value(session_id)[0],
                "trace_count": trace_count,
                "span_count": span_count,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reported_tool_calls": reported_tools,
                "executed_tool_calls": executed_tools,
                "planned_tool_calls": planned_tools,
                "error_trace_count": len(session["error_trace_ids"]),
            }
        )
    top_rows.sort(
        key=lambda row: (
            int(row["input_tokens"]),
            int(row["span_count"]),
            int(row["trace_count"]),
        ),
        reverse=True,
    )
    return {
        "trace_distribution": _token_distribution(traces_per_session),
        "span_distribution": _token_distribution(spans_per_session),
        "input_token_distribution": _token_distribution(input_tokens_per_session),
        "output_token_distribution": _token_distribution(output_tokens_per_session),
        "reported_tool_call_distribution": _token_distribution(reported_tools_per_session),
        "executed_tool_call_distribution": _token_distribution(executed_tools_per_session),
        "planned_tool_call_distribution": _token_distribution(planned_tools_per_session),
        "top_by_input_tokens": top_rows[:top_n],
    }


def _first_string_value(attrs: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = attrs.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _fingerprint_value(value: object) -> tuple[str, int]:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    raw = text.encode()
    return sha256(raw).hexdigest()[:12], len(raw)


def _hashed_identifier(value: str, *, length: int) -> str:
    return sha256(f"care-ai-halo:{value}".encode()).hexdigest()[:length]


def _counter_update(counter: Counter[str], value: object) -> None:
    if isinstance(value, str) and value:
        counter.update([value])


def _counter_update_unique(counter: Counter[str], values: list[object]) -> None:
    seen = {value for value in values if isinstance(value, str) and value}
    counter.update(seen)


def _set_update(values: set[str], value: object) -> None:
    if isinstance(value, str) and value:
        values.add(value)


def _top_counts(counter: Counter[str], top_n: int) -> list[dict[str, int | str]]:
    return [{"name": name, "count": count} for name, count in counter.most_common(top_n)]


def _render_count_table(title: str, rows: list[dict[str, int | str]]) -> None:
    table = Table(title=title)
    table.add_column("Name")
    table.add_column("Count", justify="right")
    for row in rows:
        table.add_row(str(row["name"]), str(row["count"]))
    console.print(table)


def _render_audit_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    table = Table(title=title)
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*[_format_cell(row.get(column)) for column in columns])
    console.print(table)


def _format_cell(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return "" if value is None else str(value)


def _metadata_safe_audit_report(report: dict[str, Any]) -> dict[str, Any]:
    """Remove raw status message text from an audit report before packaging it."""
    safe_report = json.loads(json.dumps(report))
    for row in safe_report.get("tool_errors", []):
        message = row.pop("status_message", "")
        sha, byte_count = _fingerprint_value(message)
        row["status_message_sha256_12"] = sha
        row["status_message_bytes"] = byte_count
    return safe_report


def _render_comparison_table(title: str, rows: dict[str, Any]) -> None:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Per Trace Delta", justify="right")
    table.add_column("Direction")
    for name, row in rows.items():
        table.add_row(
            name,
            str(row["baseline"]),
            str(row["candidate"]),
            str(row["delta"]),
            str(row["per_trace_delta"]),
            str(row["direction"]),
        )
    console.print(table)


def _render_token_distribution_comparison(title: str, rows: dict[str, Any]) -> None:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Baseline Median", justify="right")
    table.add_column("Candidate Median", justify="right")
    table.add_column("Median Delta", justify="right")
    table.add_column("Baseline P95", justify="right")
    table.add_column("Candidate P95", justify="right")
    table.add_column("P95 Delta", justify="right")
    table.add_column("Direction")
    for name, row in rows.items():
        baseline = row["baseline"]
        candidate = row["candidate"]
        delta = row["delta"]
        table.add_row(
            name,
            str(baseline["median"]),
            str(candidate["median"]),
            str(delta["median"]),
            str(baseline["p95"]),
            str(candidate["p95"]),
            str(delta["p95"]),
            str(row["direction_by_p95"]),
        )
    console.print(table)


def doctor_report(
    trace_path: Path | None = None,
    *,
    model: str = "gpt-5.4-mini",
    check_model: bool = False,
    use_gocode: bool = False,
) -> dict[str, Any]:
    """Return local readiness checks for the Care AI HALO loop."""
    with _model_provider_env(use_gocode=use_gocode):
        api_key = os.environ.get("OPENAI_API_KEY")
        report: dict[str, Any] = {
            "model_provider": "gocode" if use_gocode else "environment",
            "openai_api_key": "set" if api_key else "missing",
            "gocode_codex_api_key": (
                "set" if os.environ.get("GOCODE_CODEX_API_KEY") else "missing"
            ),
            "openai_base_url": _openai_base_url_status(use_gocode=use_gocode),
            "sandbox": _sandbox_status(),
            "model": model,
            "model_access": "not_checked",
        }
        if check_model:
            report["model_access"] = _check_model_access(model)
        if trace_path is not None:
            summary = inspect_halo_jsonl(trace_path, top_n=10)
            audit_report = audit_halo_jsonl(trace_path, top_n=10)
            report["trace"] = {
                "path": str(trace_path),
                "total_traces": summary["total_traces"],
                "total_spans": summary["total_spans"],
                "session_count": summary["session_count"],
                "ucid_count": summary["ucid_count"],
                "session_trace_distribution": summary["session_trace_distribution"],
                "session_input_token_distribution": summary["session_input_token_distribution"],
                "tool_names": summary["tool_names"],
                "planned_tool_names": summary["planned_tool_names"],
                "audit_counts": audit_report["counts"],
            }
        return report


@contextmanager
def _model_provider_env(*, use_gocode: bool) -> Iterator[None]:
    """Temporarily route OpenAI-compatible calls through GoCode when requested."""
    if not use_gocode:
        yield
        return

    previous_openai_key = os.environ.get("OPENAI_API_KEY")
    previous_openai_base_url = os.environ.get("OPENAI_BASE_URL")
    try:
        if os.environ.get("GOCODE_CODEX_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.environ["GOCODE_CODEX_API_KEY"]
        os.environ["OPENAI_BASE_URL"] = _GOCODE_OPENAI_BASE_URL
        yield
    finally:
        _restore_env_var("OPENAI_API_KEY", previous_openai_key)
        _restore_env_var("OPENAI_BASE_URL", previous_openai_base_url)


def _restore_env_var(key: str, previous_value: str | None) -> None:
    if previous_value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = previous_value


def _openai_base_url_status(*, use_gocode: bool) -> str:
    value = os.environ.get("OPENAI_BASE_URL")
    if use_gocode and value == _GOCODE_OPENAI_BASE_URL:
        return "gocode"
    return "set" if value else "default"


def _sandbox_status() -> str:
    sandbox_logger = logging.getLogger("engine.sandbox.sandbox")
    was_disabled = sandbox_logger.disabled
    sandbox_logger.disabled = True
    try:
        with redirect_stderr(StringIO()):
            return "available" if Sandbox.get() is not None else "unavailable"
    finally:
        sandbox_logger.disabled = was_disabled


def _check_model_access(model: str) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        return "skipped_missing_api_key"
    try:
        from openai import OpenAI

        client = OpenAI()
        client.responses.create(
            model=model,
            input="Reply with OK only.",
            max_output_tokens=16,
        )
    except Exception as exc:  # pragma: no cover - exact SDK exception varies by provider
        return f"failed:{type(exc).__name__}"
    return "ok"


def _is_authentication_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return type(exc).__name__ == "AuthenticationError" or "invalid_api_key" in text


@app.command("doctor")
def doctor(
    trace_path: Path | None = typer.Argument(
        None,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Optional HALO-compatible JSONL trace file to inspect and audit.",
    ),
    model: str = typer.Option("gpt-5.4-mini", "--model", "-m"),
    check_model: bool = typer.Option(
        False,
        "--check-model",
        help="Make a live model API request to validate credentials.",
    ),
    use_gocode: bool = typer.Option(
        False,
        "--gocode",
        help="Use Tommy's GoCode OpenAI-compatible provider for the live model check.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Check local readiness before running Care AI HALO diagnosis."""
    report = doctor_report(
        trace_path,
        model=model,
        check_model=check_model,
        use_gocode=use_gocode,
    )
    if as_json:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return

    table = Table(title="Care AI HALO Doctor")
    table.add_column("Check")
    table.add_column("Status")
    for key in [
        "model_provider",
        "openai_api_key",
        "gocode_codex_api_key",
        "openai_base_url",
        "sandbox",
        "model",
        "model_access",
    ]:
        table.add_row(key, str(report[key]))
    console.print(table)
    if trace := report.get("trace"):
        trace_table = Table(title="Trace")
        trace_table.add_column("Metric")
        trace_table.add_column("Value")
        for key in ["path", "total_traces", "total_spans", "session_count", "ucid_count"]:
            trace_table.add_row(key, str(trace[key]))
        trace_table.add_row(
            "session_trace_distribution",
            json.dumps(trace["session_trace_distribution"], sort_keys=True),
        )
        trace_table.add_row("audit_counts", json.dumps(trace["audit_counts"], sort_keys=True))
        console.print(trace_table)


@app.command("sanitize")
def sanitize(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Input HALO-compatible JSONL trace file.",
    ),
    output_path: Path = typer.Argument(
        ...,
        dir_okay=False,
        writable=True,
        help="Output metadata-only HALO-compatible JSONL trace file.",
    ),
    keep_status_message: bool = typer.Option(
        False,
        "--keep-status-message",
        help="Keep OTel status.message text instead of redacting it.",
    ),
    keep_identifiers: bool = typer.Option(
        False,
        "--keep-identifiers",
        help="Keep original trace, span, session, UCID, request, and tool-call identifiers.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Redact raw inputs and outputs while preserving trace structure."""
    report = sanitize_halo_jsonl(
        input_path,
        output_path,
        keep_status_message=keep_status_message,
        keep_identifiers=keep_identifiers,
    )
    if as_json:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return

    console.print(
        " | ".join(
            [
                f"wrote={report['spans_written']}",
                f"redacted_attributes={report['redacted_attribute_count']}",
                f"redacted_identifiers={report['redacted_identifier_count']}",
                f"redacted_status_messages={report['redacted_status_message_count']}",
                f"output={report['output_path']}",
            ]
        )
    )


@app.command("inspect")
def inspect(
    trace_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="HALO-compatible JSONL trace file, usually produced by convert-langfuse.",
    ),
    top: int = typer.Option(10, "--top", min=1, max=50, help="Rows per top-count table."),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Inspect a converted Care AI trace file locally without running an LLM."""
    summary = inspect_halo_jsonl(trace_path, top_n=top)
    if as_json:
        typer.echo(json.dumps(summary, indent=2, sort_keys=True))
        return

    console.print(f"Trace file: {summary['trace_path']}")
    console.print(
        " | ".join(
            [
                f"traces={summary['total_traces']}",
                f"spans={summary['total_spans']}",
                f"error_traces={summary['error_trace_count']}",
                f"sessions={summary['session_count']}",
                f"ucids={summary['ucid_count']}",
                f"input_tokens={summary['total_input_tokens']}",
                f"output_tokens={summary['total_output_tokens']}",
                f"session_input_p95={summary['session_input_token_distribution']['p95']}",
            ]
        )
    )
    for title, key in [
        ("Projects", "project_ids"),
        ("Services", "service_names"),
        ("Observation Kinds", "observation_kinds"),
        ("Models", "model_names"),
        ("Agents", "agent_names"),
        ("Executors", "executor_names"),
        ("Tools", "tool_names"),
        ("Planned Tool Calls", "planned_tool_names"),
        ("Tool Errors", "tool_error_names"),
        ("Statuses", "status_codes"),
    ]:
        rows = summary[key]
        if rows:
            _render_count_table(title, rows)


@app.command("audit")
def audit(
    trace_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="HALO-compatible JSONL trace file, usually produced by convert-langfuse.",
    ),
    high_input_tokens: int = typer.Option(
        32_000,
        "--high-input-tokens",
        min=1,
        help="Flag LLM spans with at least this many input tokens.",
    ),
    high_output_tokens: int = typer.Option(
        2_000,
        "--high-output-tokens",
        min=1,
        help="Flag LLM spans with at least this many output tokens.",
    ),
    top: int = typer.Option(25, "--top", min=1, max=100, help="Rows per finding type."),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Mechanically audit converted Care AI traces without running an LLM."""
    report = audit_halo_jsonl(
        trace_path,
        high_input_tokens=high_input_tokens,
        high_output_tokens=high_output_tokens,
        top_n=top,
    )
    if as_json:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return

    console.print(f"Trace file: {report['trace_path']}")
    console.print(
        " | ".join(
            [
                f"traces={report['total_traces']}",
                f"planned_missing={report['counts']['planned_missing_tool_calls']}",
                f"repeated_tools={report['counts']['repeated_tool_calls']}",
                f"transfer_no_diagnostic={report['counts']['transfer_without_diagnostic_tool']}",
                f"tool_errors={report['counts']['tool_errors']}",
                f"high_token_llm={report['counts']['high_token_llm_spans']}",
            ]
        )
    )
    table_specs = [
        (
            "Planned Tool Calls Missing Executed TOOL Span",
            "planned_missing_tool_calls",
            ["trace_id", "planned_tool_name", "llm_span_ids", "executed_tool_names"],
        ),
        (
            "Repeated Tool Calls With Same Input Fingerprint",
            "repeated_tool_calls",
            ["trace_id", "tool_name", "count", "span_ids", "input_sha256_12"],
        ),
        (
            "Transfer To Human Without Prior Diagnostic Tool In Same Trace",
            "transfer_without_diagnostic_tool",
            ["trace_id", "transfer_span_ids", "planned_tool_names", "executor_names"],
        ),
        (
            "Tool Errors",
            "tool_errors",
            ["trace_id", "span_id", "tool_name", "status_message"],
        ),
        (
            "High Token LLM Spans",
            "high_token_llm_spans",
            ["trace_id", "span_id", "model_name", "input_tokens", "output_tokens"],
        ),
    ]
    for title, key, columns in table_specs:
        rows = report[key]
        if rows:
            _render_audit_table(title, rows, columns)


@app.command("compare-audits")
def compare_audits_command(
    baseline_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Baseline HALO-compatible JSONL trace file.",
    ),
    candidate_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Candidate HALO-compatible JSONL trace file after a prompt/config change.",
    ),
    high_input_tokens: int = typer.Option(
        32_000,
        "--high-input-tokens",
        min=1,
        help="Flag LLM spans with at least this many input tokens.",
    ),
    high_output_tokens: int = typer.Option(
        2_000,
        "--high-output-tokens",
        min=1,
        help="Flag LLM spans with at least this many output tokens.",
    ),
    top: int = typer.Option(25, "--top", min=1, max=100, help="Rows per finding type."),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Compare deterministic audit signals before and after a harness change."""
    report = compare_audits(
        baseline_path,
        candidate_path,
        high_input_tokens=high_input_tokens,
        high_output_tokens=high_output_tokens,
        top_n=top,
    )
    if as_json:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return

    console.print(f"Baseline: {report['baseline_path']}")
    console.print(f"Candidate: {report['candidate_path']}")
    trace_counts = report["trace_counts"]
    console.print(
        " | ".join(
            [
                f"baseline_traces={trace_counts['baseline']}",
                f"candidate_traces={trace_counts['candidate']}",
                f"trace_delta={trace_counts['delta']}",
            ]
        )
    )
    _render_comparison_table("Audit Signal Delta", report["signal_deltas"])
    _render_comparison_table("Token Total Delta", report["token_totals"])
    _render_token_distribution_comparison(
        "LLM Token Distribution Delta",
        report["token_distributions"],
    )
    _render_token_distribution_comparison(
        "Session Token Distribution Delta",
        report["session_token_distributions"],
    )


@app.command("evidence-pack")
def evidence_pack_command(
    trace_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="Baseline HALO-compatible JSONL trace file.",
    ),
    output_path: Path = typer.Argument(
        ...,
        dir_okay=False,
        writable=True,
        help="Output JSON evidence pack.",
    ),
    candidate_path: Path | None = typer.Option(
        None,
        "--candidate",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Optional candidate trace set to compare against the baseline.",
    ),
    prompt_snapshot_path: Path | None = typer.Option(
        None,
        "--prompt-snapshot",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Optional metadata-only prompt snapshot produced by `prompt-snapshot`.",
    ),
    focus: DiagnosticFocus = typer.Option(
        DiagnosticFocus.overview,
        "--focus",
        "-f",
        case_sensitive=False,
        help="Care AI diagnostic lens for the included HALO prompt.",
    ),
    executor: str | None = typer.Option(
        None,
        "--executor",
        help="Prioritize one care_ai.session_executor_name, such as airo-care-orchestrator.",
    ),
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        help="Restrict prompt instructions to one langfuse.session_id / care_ai.conversation_id.",
    ),
    question: str | None = typer.Option(
        None,
        "--question",
        "-q",
        help="Additional trace-data question to append to the selected focus prompt.",
    ),
    top: int = typer.Option(25, "--top", min=1, max=100, help="Rows per finding type."),
    high_input_tokens: int = typer.Option(
        32_000,
        "--high-input-tokens",
        min=1,
        help="Flag LLM spans with at least this many input tokens.",
    ),
    high_output_tokens: int = typer.Option(
        2_000,
        "--high-output-tokens",
        min=1,
        help="Flag LLM spans with at least this many output tokens.",
    ),
    model: str = typer.Option("gpt-5.4-mini", "--model", "-m"),
    check_model: bool = typer.Option(
        False,
        "--check-model",
        help="Make a live model API request in the included doctor report.",
    ),
    use_gocode: bool = typer.Option(
        False,
        "--gocode",
        help="Use Tommy's GoCode OpenAI-compatible provider for the live model check.",
    ),
) -> None:
    """Write a metadata-safe evidence bundle before running HALO diagnosis."""
    pack = build_evidence_pack(
        trace_path,
        candidate_path=candidate_path,
        prompt_snapshot_path=prompt_snapshot_path,
        focus=focus,
        executor=executor,
        session_id=session_id,
        extra_question=question,
        top_n=top,
        model=model,
        check_model=check_model,
        use_gocode=use_gocode,
        high_input_tokens=high_input_tokens,
        high_output_tokens=high_output_tokens,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n")
    safety = pack["safety"]
    console.print(
        " | ".join(
            [
                f"wrote={output_path}",
                f"traces={pack['inspect']['total_traces']}",
                f"spans={pack['inspect']['total_spans']}",
                f"safe_metadata={safety['safe_for_metadata_only_diagnosis']}",
                f"audit_counts={json.dumps(pack['audit']['counts'], sort_keys=True)}",
            ]
        )
    )


def build_diagnostic_prompt(
    *,
    focus: DiagnosticFocus,
    executor: str | None = None,
    session_id: str | None = None,
    extra_question: str | None = None,
) -> str:
    """Build a Care AI diagnostic prompt that asks HALO about trace data only."""
    scope = []
    if executor:
        scope.append(f"Prioritize spans where care_ai.session_executor_name equals {executor!r}.")
    if session_id:
        scope.append(
            "Restrict the analysis to the conversation/session identified by "
            f"langfuse.session_id or care_ai.conversation_id equal to {session_id!r}."
        )
    scope_text = "\n".join(f"- {line}" for line in scope) if scope else "- Use the full dataset."

    focus_prompts = {
        DiagnosticFocus.overview: (
            "Find the top recurring harness-level failure patterns across the dataset. "
            "Group by concrete trace evidence: tool names, status codes, repeated outputs, "
            "agent names, and token outliers. For each pattern, cite trace_ids and explain "
            "why it appears systemic rather than isolated."
        ),
        DiagnosticFocus.tool_errors: (
            "Analyze TOOL spans. Count traces with STATUS_CODE_ERROR tool spans, group "
            "literal error strings or failing output.value payloads when present, and show "
            "whether the agent recovered after each error. If the trace was sanitized, use "
            "status codes, redaction fingerprints, and byte counts instead of claiming exact "
            "error text. Cite trace_ids, span_ids, tool.name, and care_ai.tool_call_id."
        ),
        DiagnosticFocus.routing: (
            "Analyze agent and tool selection. Look for misrouting, unnecessary "
            "agent-as-tool invocations, missing specialist calls, and cases where "
            "transfer_to_human happened before the agent tried an available diagnostic "
            "path. Cite trace_ids, tool.name values, target agent metadata, and the "
            "sequence that supports each claim."
        ),
        DiagnosticFocus.loops: (
            "Find inefficient exploration loops. Look for repeated tool calls with "
            "similar arguments, repeated LLM turns without new evidence, refusal or "
            "clarification loops, and long traces whose later turns do not change the "
            "state. Cite trace_ids, span_ids, repeated arguments, and turn sequence."
        ),
        DiagnosticFocus.cost: (
            "Analyze token and model usage. Identify traces with unusually high "
            "inference.llm.input_tokens or output_tokens, models involved, and whether "
            "the extra spend is tied to tool loops, large context, or repeated routing. "
            "Cite trace_ids and numeric token totals."
        ),
        DiagnosticFocus.mcp: (
            "Analyze MCP and customer-context usage. Look for failed, missing, or "
            "redundant calls to product, domain, case-management, and diagnostic tools. "
            "Group by tool.name and output.value patterns when raw payloads are available. "
            "If the trace was sanitized, group by metadata, redaction fingerprints, and byte "
            "counts instead of claiming literal payload snippets."
        ),
    }
    question = focus_prompts[focus]
    if extra_question:
        question = f"{question}\n\nAdditional question: {extra_question}"

    harness_context = build_harness_context(executor=executor)

    return "\n".join(
        [
            "You are analyzing Care AI Langfuse traces that were converted into HALO JSONL.",
            "Stay inside the trace evidence. Do not invent code paths, file paths, prompts, or fixes.",
            "Use trace_ids, span_ids, langfuse.session_id, care_ai.conversation_id, and care_ai.ucid when available.",
            "Care AI Langfuse GENERATION.input often contains only the latest user message, not the full system prompt or prior turns. Do not treat sparse GENERATION input as proof of missing context.",
            "For LLM spans, care_ai.llm_planned_tool_names lists function_call names parsed from GENERATION.output when present. Compare those planned calls against actual TOOL spans before claiming a tool call was skipped or hallucinated.",
            "If the trace or evidence summary says payloads, status messages, or identifiers were redacted, do not describe token counts, hashes, ids, or byte counts as literal payload snippets. Say raw payload text is unavailable and cite the metadata that remains.",
            "",
            "Care AI harness context:",
            harness_context,
            "",
            "Expected answer shape:",
            "1. Evidence table with pattern, count, trace_ids, span_ids, executor, tool or agent names, and observed metadata. Include literal payload snippets only when raw payload text is actually present.",
            "2. Proposed prompt or configuration improvement with the exact Care AI surface to change.",
            "3. Verification plan that names the eval, trace sample, or live-test evidence needed before shipping.",
            "4. No-change calls for patterns where trace evidence is too weak or conflicts with known harness context.",
            "",
            "Scope:",
            scope_text,
            "",
            "Task:",
            question,
        ]
    )


def build_harness_context(*, executor: str | None = None) -> str:
    """Return Care AI topology facts that bound HALO's recommendations."""
    normalized_executor = executor or ""
    include_variants = not executor or normalized_executor in _VARIANT_EXECUTORS
    include_help_center_variants = (
        not executor or normalized_executor in _HELP_CENTER_VARIANT_EXECUTORS
    )
    prod_tools = ", ".join(f"`{tool}`" for tool in _PROD_ORCHESTRATOR_TOOLS)

    lines = [
        "- Care AI customer-facing orchestrators run through `care-ai-agents` using the OpenAI Agents SDK.",
        "- Prompt bodies are managed by the Prompt Management API backed by Langfuse. Runtime fetches `:latest`; repo code is the integration layer, not the prompt body source of truth.",
        "- The production `airo-care-orchestrator` manager prompt is `c1/airo-care/orchestrator/orchestrator`.",
        f"- Production manager tools are {prod_tools}, plus productGraph MCP tools discovered at runtime.",
        "- `domain_and_dns_tool` and `domain_lifecycle_tool` are agents-as-tools via `agent.asTool()`, not SDK handoffs.",
        "- The DNS agent prompt is `c1/airo-care/orchestrator/dns-agent`; its direct tools include `dns_resolve`, `http_probe`, `ssl_check`, and `whois_lookup`.",
        "- The Domain Lifecycle agent prompt is `c1/airo-care/orchestrator/domain-lifecycle-agent`; it is MCP-only for domain context, and case creation is disabled in current orchestrator executors.",
        "- All orchestrator-family customer-facing agents currently use `gpt-5.4` with temperature 0.2 unless trace evidence proves otherwise.",
    ]
    surface = care_ai_surface_for_executor(executor)
    if surface:
        lines.append(
            " ".join(
                [
                    f"- Selected executor `{executor}` maps to prompt",
                    f"`{surface['prompt_name']}` and session-executor wiring",
                    f"`{surface['session_executor_path']}`.",
                ]
            )
        )
    if include_variants:
        variant_tools = ", ".join(f"`{tool}`" for tool in _VARIANT_SPECIALIST_TOOLS)
        lines.extend(
            [
                "- Test, sandbox, TCF, and email-MCP-bot orchestrator variants can add `general_support_tool`, `hosting_tool`, `billing_tool`, and `email_diagnostic_tool`.",
                f"- Variant specialist tools to consider when the executor is one of {_format_code_list(sorted(_VARIANT_EXECUTORS))}: {variant_tools}.",
                "- Variant prompt names use `c1/airo-care/<variant>/...`; do not map variant findings onto the production prompt without confirming the executor.",
            ]
        )
    if include_help_center_variants:
        help_center_tools = ", ".join(f"`{tool}`" for tool in _HELP_CENTER_VARIANT_TOOLS)
        lines.append(
            " ".join(
                [
                    "- `orchestrator-test-tcf` and `orchestrator-sandbox` can also add",
                    f"help-center agents-as-tools: {help_center_tools}.",
                ]
            )
        )
    lines.extend(
        [
            "- Treat `transfer_to_human` differently from agents-as-tools. A human transfer may be correct when the harness has no diagnostic path or required backend capability.",
            "- If recommending a prompt change, say which prompt likely owns the behavior. If recommending a code/config change, say which wiring surface is implicated. Do not claim a concrete file path unless a trace attribute contains it.",
        ]
    )
    return "\n".join(lines)


def _format_code_list(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values)


@app.command("diagnose")
def diagnose(
    trace_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        dir_okay=False,
        help="HALO-compatible JSONL trace file, usually produced by convert-langfuse.",
    ),
    focus: DiagnosticFocus = typer.Option(
        DiagnosticFocus.overview,
        "--focus",
        "-f",
        case_sensitive=False,
        help="Care AI diagnostic lens.",
    ),
    executor: str | None = typer.Option(
        None,
        "--executor",
        help="Prioritize one care_ai.session_executor_name, such as airo-care-orchestrator.",
    ),
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        help="Restrict prompt instructions to one langfuse.session_id / care_ai.conversation_id.",
    ),
    question: str | None = typer.Option(
        None,
        "--question",
        "-q",
        help="Additional trace-data question to append to the selected focus prompt.",
    ),
    model: str = typer.Option("gpt-5.4-mini", "--model", "-m"),
    max_depth: int = typer.Option(2, "--max-depth", min=0),
    max_turns: int = typer.Option(12, "--max-turns", min=1),
    max_parallel: int = typer.Option(2, "--max-parallel", min=1),
    refusal_retries: int = typer.Option(
        0,
        "--refusal-retries",
        min=0,
        help="Retry an agent model request this many times when the model refuses.",
    ),
    reasoning_effort: str | None = typer.Option(
        None,
        "--reasoning-effort",
        help=(
            "Reasoning effort forwarded to root, subagent, and synthesis calls. One of: "
            f"{', '.join(REASONING_EFFORT_CHOICES)}."
        ),
    ),
    telemetry: bool = typer.Option(
        False,
        "--telemetry/--no-telemetry",
        help="Emit HALO's own traces while diagnosing.",
    ),
    timeout_seconds: int | None = typer.Option(
        180,
        "--timeout-seconds",
        min=1,
        help="Abort the HALO diagnosis after this many seconds. Use 0 is not supported.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        dir_okay=False,
        writable=True,
        help="Write the final HALO diagnosis answer to this file.",
    ),
    events_path: Path | None = typer.Option(
        None,
        "--events-jsonl",
        dir_okay=False,
        writable=True,
        help="Write durable HALO agent events to this JSONL file.",
    ),
    tool_mode: TraceToolMode = typer.Option(
        TraceToolMode.summary,
        "--tool-mode",
        case_sensitive=False,
        help=(
            "Trace tool set for HALO. summary disables view/search/run_code detail "
            "tools so diagnosis starts from the local evidence summary."
        ),
    ),
    include_evidence: bool = typer.Option(
        True,
        "--include-evidence/--no-include-evidence",
        help="Prepend local metadata-only inspect/audit/safety summary to the HALO prompt.",
    ),
    evidence_top: int = typer.Option(
        10,
        "--evidence-top",
        min=1,
        max=50,
        help="Rows per local evidence finding included in the HALO prompt.",
    ),
    use_gocode: bool = typer.Option(
        False,
        "--gocode",
        help="Route HALO model calls through Tommy's GoCode OpenAI-compatible provider.",
    ),
    print_prompt: bool = typer.Option(
        False,
        "--print-prompt",
        help="Print the generated diagnostic prompt and exit without running HALO.",
    ),
) -> None:
    """Run HALO with a Care AI trace-diagnostic prompt."""
    prompt = build_diagnosis_prompt_with_evidence(
        trace_path,
        focus=focus,
        executor=executor,
        session_id=session_id,
        extra_question=question,
        include_evidence=include_evidence,
        evidence_top=evidence_top,
    )
    if print_prompt:
        typer.echo(prompt)
        return

    try:
        with _model_provider_env(use_gocode=use_gocode):
            run_trace(
                trace_path=trace_path,
                prompt=prompt,
                model=model,
                max_depth=max_depth,
                max_turns=max_turns,
                max_parallel=max_parallel,
                refusal_retries=refusal_retries,
                reasoning_effort=reasoning_effort,
                telemetry=telemetry,
                trace_detail_tools_enabled=tool_mode == TraceToolMode.full,
                run_code_enabled=tool_mode == TraceToolMode.full,
                timeout_seconds=timeout_seconds,
                output_path=output_path,
                events_path=events_path,
            )
    except TimeoutError as exc:
        typer.echo(
            "HALO diagnosis timed out before producing a final answer. "
            "Try `--tool-mode summary --reasoning-effort low --max-turns 4`, "
            "or raise `--timeout-seconds` when provider latency is expected.",
            err=True,
        )
        raise typer.Exit(1) from exc
    except Exception as exc:
        if _is_authentication_error(exc):
            typer.echo(
                "HALO diagnosis could not authenticate with the model provider. "
                "Run `uv run halo-careai doctor --check-model --gocode` to validate GoCode, "
                "or `uv run halo-careai doctor --check-model` for the current environment.",
                err=True,
            )
            raise typer.Exit(1) from exc
        raise


if __name__ == "__main__":
    app()
