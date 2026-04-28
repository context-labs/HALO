"""Wire the openai-agents SDK to write inference.net-format JSONL traces."""
import os

from agents import add_trace_processor

from inference_otlp_exporter import ExportContext, InferenceOtlpFileProcessor

DEFAULT_OUTPUT_PATH = "traces.jsonl.gz"


def setup_tracing(
    service_name: str = "halo-openai-agents-demo",
    project_id: str = "halo-demo",
) -> InferenceOtlpFileProcessor:
    path = os.getenv("HALO_TRACES_PATH", DEFAULT_OUTPUT_PATH)
    ctx = ExportContext(project_id=project_id, service_name=service_name)
    processor = InferenceOtlpFileProcessor(path, ctx=ctx)
    add_trace_processor(processor)
    return processor
