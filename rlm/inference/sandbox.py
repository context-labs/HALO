"""Sandboxed Python execution for the ``run_code`` tool.

Runs user (LLM-provided) code in a fresh subprocess with a wall-clock
timeout and a stdout size cap. The subprocess imports the dataset layer
and pre-binds:

* ``store``: the ``IndexStore`` for the dataset
* ``reader``: a ``TraceReader`` already opened on the source JSONL
* ``descriptor``: the ``DatasetDescriptor``
* a handful of convenience imports

This gives the analyst an RLM-style "write whatever query you need"
escape hatch that the eight fixed tools can't always express, without
letting that code touch the host process's memory, open sockets, or
outlive its timeout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from dataset import DatasetDescriptor, HFMapping, OpenInferenceMapping


# Pre-bound helpers + an eval loop. The parent passes the code via a
# ``--code-file`` argv; output is pickled to stdout as JSON so we don't
# have to trust ``print`` being text-clean.
_RUNNER = r'''
from __future__ import annotations
import io, json, sys, traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

descriptor_path = Path(sys.argv[1])
jsonl_path = Path(sys.argv[2])
index_path = Path(sys.argv[3])
code_path = Path(sys.argv[4])

sys.path.insert(0, str(Path(sys.argv[5])))  # project root so ``import dataset`` works

from dataset import IndexStore  # noqa: E402
from dataset.reader import TraceReader  # noqa: E402
import json as _json
raw = _json.loads(descriptor_path.read_text())

# Re-hydrate the descriptor from a JSON payload that mirrors
# ``DatasetDescriptor``'s shape. The parent always serializes the full
# structure (``mapping``, ``primary_metric``, ``labels``) so defaults
# here are just safety rails.
from dataset import (  # noqa: E402
    DatasetDescriptor, HFMapping, Label, Metric, OpenInferenceMapping,
)

m_raw = raw.get("mapping") or {}
if m_raw.get("kind") == "openinference":
    mapping = OpenInferenceMapping(id_attribute=m_raw.get("id_attribute"))
else:
    mapping = HFMapping(
        id_field=m_raw.get("id_field", "id"),
        query_field=m_raw.get("query_field", "query"),
        messages_field=m_raw.get("messages_field", "messages"),
        final_answer_field=m_raw.get("final_answer_field"),
        documents_field=m_raw.get("documents_field"),
        document_path_field=m_raw.get("document_path_field", "path"),
        usage_field=m_raw.get("usage_field", "metadata.usage"),
        turns_field=m_raw.get("turns_field", "metadata.turns_used"),
        tool_calls_total_field=m_raw.get("tool_calls_total_field", "metadata.total_tool_calls"),
        tool_errors_field=m_raw.get("tool_errors_field", "metadata.tool_errors"),
    )

def _mk_metric(d):
    return Metric(
        name=d["name"],
        source=d["source"],
        kind=d.get("kind", "score_01"),
        display_name=d.get("display_name"),
        higher_is_better=d.get("higher_is_better", True),
        perfect_threshold=d.get("perfect_threshold", 0.999),
        zero_threshold=d.get("zero_threshold", 0.001),
    )

pm_raw = raw.get("primary_metric")
primary_metric = _mk_metric(pm_raw) if pm_raw else None
secondary_metrics = [_mk_metric(m) for m in (raw.get("secondary_metrics") or [])]

labels = [
    Label(name=l["name"], source=l["source"], display_name=l.get("display_name"))
    for l in (raw.get("labels") or [])
]

descriptor = DatasetDescriptor(
    id=raw["id"],
    name=raw["name"],
    source_path=Path(raw["source_path"]),
    mapping=mapping,
    source_model=raw.get("source_model"),
    description=raw.get("description"),
    primary_metric=primary_metric,
    secondary_metrics=secondary_metrics,
    ground_truth_source=raw.get("ground_truth_source"),
    labels=labels,
    seed_questions=list(raw.get("seed_questions") or []),
)

store = IndexStore.load(index_path, descriptor)
reader = TraceReader(jsonl_path)
reader.__enter__()

user_code = code_path.read_text()

out_buf = io.StringIO()
err_buf = io.StringIO()
ns = {
    "store": store,
    "reader": reader,
    "descriptor": descriptor,
    "json": json,
    "__name__": "__halo_run_code__",
}
_envelope = {"stdout": "", "stderr": "", "error": None}
try:
    with redirect_stdout(out_buf), redirect_stderr(err_buf):
        exec(compile(user_code, "<run_code>", "exec"), ns)
except SystemExit as e:
    _envelope["error"] = f"SystemExit: {e.code}"
except BaseException:
    tb = traceback.format_exc()
    _envelope["error"] = tb
finally:
    reader.close()

_envelope["stdout"] = out_buf.getvalue()
_envelope["stderr"] = err_buf.getvalue()
# Capture expression-values the user may have set under well-known names
# — often more ergonomic than ``print``.
for k in ("result", "out", "ans"):
    if k in ns:
        try:
            _envelope.setdefault("returned", {})[k] = json.loads(json.dumps(ns[k], default=str))
        except Exception:
            _envelope.setdefault("returned", {})[k] = repr(ns[k])

print("__HALO_RESULT__" + json.dumps(_envelope, default=str))
'''


def run_code(
    *,
    code: str,
    descriptor: DatasetDescriptor,
    index_path: Path,
    project_root: Path,
    timeout_s: float = 10.0,
    max_output_bytes: int = 100_000,
) -> dict[str, Any]:
    """Execute ``code`` in a subprocess with the dataset pre-loaded.

    The subprocess inherits the parent's environment but runs in a
    working directory of ``/tmp`` so relative paths in user code can't
    accidentally clobber the project tree. ``timeout_s`` is a hard kill;
    ``max_output_bytes`` caps stdout so a runaway print loop can't blow
    up the reply.
    """
    import tempfile

    # Resolve to absolute paths since the subprocess runs with cwd=/tmp.
    index_path = index_path.resolve()
    project_root = project_root.resolve()
    source_path = descriptor.source_path
    if not source_path.is_absolute():
        source_path = (project_root / source_path).resolve()

    # Serialize the descriptor so the child can rebuild it without
    # importing ``catalog.<mod>`` (which would re-execute module code).
    if isinstance(descriptor.mapping, OpenInferenceMapping):
        mapping_payload: dict[str, Any] = {
            "kind": "openinference",
            "id_attribute": descriptor.mapping.id_attribute,
        }
    else:
        assert isinstance(descriptor.mapping, HFMapping)
        hm = descriptor.mapping
        mapping_payload = {
            "kind": "hf",
            "id_field": hm.id_field,
            "query_field": hm.query_field,
            "messages_field": hm.messages_field,
            "final_answer_field": hm.final_answer_field,
            "documents_field": hm.documents_field,
            "document_path_field": hm.document_path_field,
            "usage_field": hm.usage_field,
            "turns_field": hm.turns_field,
            "tool_calls_total_field": hm.tool_calls_total_field,
            "tool_errors_field": hm.tool_errors_field,
        }

    def _metric_payload(m: Any) -> dict[str, Any]:
        return {
            "name": m.name,
            "source": m.source,
            "kind": m.kind,
            "display_name": m.display_name,
            "higher_is_better": m.higher_is_better,
            "perfect_threshold": m.perfect_threshold,
            "zero_threshold": m.zero_threshold,
        }

    pm = descriptor.primary_metric
    primary_metric_payload = _metric_payload(pm) if pm is not None else None
    secondary_metric_payload = [_metric_payload(m) for m in descriptor.secondary_metrics]

    desc_payload = {
        "id": descriptor.id,
        "name": descriptor.name,
        "source_path": str(source_path),
        "source_model": descriptor.source_model,
        "description": descriptor.description,
        "mapping": mapping_payload,
        "primary_metric": primary_metric_payload,
        "secondary_metrics": secondary_metric_payload,
        "ground_truth_source": descriptor.ground_truth_source,
        "labels": [
            {"name": lbl.name, "source": lbl.source, "display_name": lbl.display_name}
            for lbl in descriptor.labels
        ],
        "seed_questions": list(descriptor.seed_questions),
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        runner_py = tmp_path / "_runner.py"
        runner_py.write_text(_RUNNER)
        code_py = tmp_path / "_user_code.py"
        code_py.write_text(code)
        desc_json = tmp_path / "_descriptor.json"
        desc_json.write_text(json.dumps(desc_payload))

        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(runner_py),
                    str(desc_json),
                    str(source_path),
                    str(index_path),
                    str(code_py),
                    str(project_root),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd="/tmp",
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired as e:
            return {
                "error": f"run_code timed out after {timeout_s}s",
                "stdout": (e.stdout or "")[:max_output_bytes] if e.stdout else "",
                "stderr": (e.stderr or "")[:max_output_bytes] if e.stderr else "",
            }

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    # Pull out our marker; anything before it is user ``print`` output,
    # anything in the marker payload is the structured result.
    marker = "__HALO_RESULT__"
    idx = stdout.rfind(marker)
    if idx < 0:
        return {
            "error": "no result sentinel in subprocess output",
            "stdout": stdout[:max_output_bytes],
            "stderr": stderr[:max_output_bytes],
            "returncode": proc.returncode,
        }
    try:
        structured = json.loads(stdout[idx + len(marker):])
    except json.JSONDecodeError as e:
        return {
            "error": f"failed to parse subprocess result JSON: {e}",
            "stdout": stdout[:max_output_bytes],
            "stderr": stderr[:max_output_bytes],
        }

    structured_stdout = structured.get("stdout", "")
    if len(structured_stdout) > max_output_bytes:
        structured["stdout"] = (
            structured_stdout[:max_output_bytes]
            + f"\n... [truncated {len(structured_stdout) - max_output_bytes} bytes]"
        )
    return structured
