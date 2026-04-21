"""Minimal Claude Agent SDK + OTEL wiring. Copy-paste into any Claude Agent SDK app."""
import os
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

DEFAULT_OTLP_ENDPOINT = "http://localhost:4318"

# Env vars the Claude Code CLI subprocess reads. setdefault so users can override
# any of them in their shell or .env file.
_CLAUDE_ENV = {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",   # distributed trace spans
    "OTEL_TRACES_EXPORTER": "otlp",
    "OTEL_METRICS_EXPORTER": "otlp",
    "OTEL_LOGS_EXPORTER": "otlp",
    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    "OTEL_TRACES_EXPORT_INTERVAL": "1000",         # faster flush for short runs
    # Content capture — on by default so demo traces are useful as fixtures.
    "OTEL_LOG_USER_PROMPTS": "1",                  # include prompt text
    "OTEL_LOG_TOOL_DETAILS": "1",                  # include bash commands, file paths
    "OTEL_LOG_TOOL_CONTENT": "1",                  # include raw tool I/O (truncated at 60 KB)
    "OTEL_LOG_RAW_API_BODIES": "1",                # include Anthropic API request/response bodies
}


def setup_tracing(service_name: str = "halo-claude-agents-demo") -> TracerProvider:
    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", DEFAULT_OTLP_ENDPOINT)
    for k, v in _CLAUDE_ENV.items():
        os.environ.setdefault(k, v)

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    endpoint = f"{os.environ['OTEL_EXPORTER_OTLP_ENDPOINT']}/v1/traces"
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    return provider


@contextmanager
def agent_span(name: str, model: str):
    tracer = trace.get_tracer("halo-claude-agents-demo")
    with tracer.start_as_current_span(
        name,
        attributes={
            "openinference.span.kind": "AGENT",
            "llm.model_name": model,
        },
    ) as span:
        yield span
