"""Minimal OpenInference + OTEL wiring. Copy-paste into any OpenAI Agents SDK app."""
import os
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

DEFAULT_OTLP_ENDPOINT = "http://localhost:4318"


def setup_tracing(service_name: str = "halo-openai-agents-demo") -> TracerProvider:
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    base = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", DEFAULT_OTLP_ENDPOINT).rstrip("/")
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=f"{base}/v1/traces")))
    OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)
    return provider
