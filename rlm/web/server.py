"""FastAPI server exposing the halo RLM agent over HTTP + SSE.

The server is dataset-agnostic: it loads every registered dataset from
``catalog/`` at startup and serves all routes scoped by ``dataset_id``.
"""

from __future__ import annotations

import base64
import importlib
import json
import os
import random
import re
import secrets
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import anyio
import asyncio
from loguru import logger

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from dataset import IndexStore, TraceReader
from dataset.autodetect import descriptor_to_python, infer_descriptor
from dataset.indexer import build_index
from inference.config import InferenceConfig
from inference.harness import run_agent
from inference.tools import ToolContext, tool_inspect_trace
from registry import DatasetRegistry, build_registry, register_live
from web.conversations import ConversationStore
from web.history import HistoryStore

_INGEST_LOCK = threading.Lock()

# One asyncio.Lock per conversation id — taken for the full duration of a
# run so concurrent /ask requests on the same conv_id serialize instead of
# interleaving mutations to conv.messages / conv.result_store.
_CONV_LOCKS: dict[str, asyncio.Lock] = {}


def _conv_lock(conv_id: str) -> asyncio.Lock:
    lock = _CONV_LOCKS.get(conv_id)
    if lock is None:
        lock = asyncio.Lock()
        _CONV_LOCKS[conv_id] = lock
    return lock


class IngestRequest(BaseModel):
    """Body for POST /api/datasets/ingest."""

    path: str = Field(..., description="Absolute path to the JSONL file on the server.")
    id: str | None = Field(None, description="URL-safe slug; auto from filename if absent.")
    name: str | None = Field(None, description="Display name; defaults to the filename stem.")
    source_model: str | None = None
    description: str | None = None
    sample: int = Field(200, ge=10, le=2000)
    build_index: bool = True
    force: bool = False


def _sanitize_slug(s: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return slug or "dataset"

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _entry_summary(did: str, registry: DatasetRegistry) -> dict[str, Any]:
    entry = registry.entries[did]
    d = entry.descriptor
    return {
        "id": d.id,
        "name": d.name,
        "description": d.description,
        "source_model": d.source_model,
        "source_path": str(d.source_path),
        "is_indexed": entry.is_indexed,
        "label_names": d.label_names,
        "has_outcome": d.primary_metric is not None,
        "has_documents": d.has_documents,
        "outcome_display_name": d.primary_metric.label if d.primary_metric else None,
        "format": d.format,
    }


def build_app(config: InferenceConfig | None = None) -> FastAPI:
    cfg = config or InferenceConfig()
    cfg.init()
    registry = build_registry(cfg.index_dir)

    if not registry.entries:
        raise SystemExit(
            "No datasets registered. Drop a descriptor file into catalog/ first."
        )

    # Eagerly load every indexed dataset; unindexed ones stay as placeholders.
    for did, entry in registry.entries.items():
        if entry.is_indexed:
            try:
                registry.load_store(did)
            except FileNotFoundError:
                pass

    history_path = cfg.index_dir / "history.jsonl"
    history = HistoryStore(history_path)
    conversations = ConversationStore(cfg.index_dir / "conversations")

    app = FastAPI(title="halo", docs_url=None, redoc_url=None)

    # HTTP Basic Auth gate (when both env vars are set).
    auth_user = os.environ.get("HALO_AUTH_USER")
    auth_pass = os.environ.get("HALO_AUTH_PASS")
    if auth_user and auth_pass:
        @app.middleware("http")
        async def basic_auth(request: Request, call_next: Any) -> Response:
            header = request.headers.get("authorization", "")
            ok = False
            if header.lower().startswith("basic "):
                try:
                    decoded = base64.b64decode(header[6:]).decode("utf-8", "ignore")
                    u, _, p = decoded.partition(":")
                    ok = secrets.compare_digest(u, auth_user) and secrets.compare_digest(p, auth_pass)
                except Exception:
                    ok = False
            if not ok:
                return Response(
                    status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="halo"'},
                    content="Authentication required",
                )
            return await call_next(request)

    catalog_dir = Path(__file__).resolve().parent.parent / "catalog"
    upload_dir = cfg.index_dir / "uploads"

    # Allowlist of roots a client may ingest from. Anything else is
    # rejected to prevent a client from reading arbitrary server files
    # (``POST /api/datasets/ingest`` takes a path parameter).
    _env_extra = os.environ.get("HALO_INGEST_ROOTS", "")
    _extra_roots = [Path(p).resolve() for p in _env_extra.split(":") if p]
    allowed_ingest_roots: list[Path] = [upload_dir.resolve(), *_extra_roots]

    def _validate_ingest_path(src_path: Path) -> Path:
        """Resolve ``src_path`` and fail if it escapes every allowed root."""
        try:
            resolved = src_path.resolve(strict=False)
        except Exception as e:
            raise HTTPException(400, f"invalid path: {e}") from e
        for root in allowed_ingest_roots:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue
        roots_str = " or ".join(str(r) for r in allowed_ingest_roots)
        raise HTTPException(
            403,
            f"path {src_path} is outside the allowed ingest roots ({roots_str}); "
            f"set HALO_INGEST_ROOTS=/path1:/path2 to widen.",
        )

    def _ingest_from_path(
        req: IngestRequest, src_path: Path,
    ) -> dict[str, Any]:
        """Run the shared inference → write descriptor → (optionally) build
        index path. Called by both the JSON-path and multipart-upload
        endpoints. Holds a lock so two concurrent ingests can't clobber.
        """
        src_path = _validate_ingest_path(src_path)
        if not src_path.exists():
            raise HTTPException(400, f"source not found: {src_path}")

        with _INGEST_LOCK:
            descriptor, report = infer_descriptor(
                src_path,
                dataset_id=req.id,
                name=req.name,
                source_model=req.source_model,
                description=req.description,
                sample_size=req.sample,
            )

            descriptor_path = catalog_dir / f"{descriptor.id}.py"
            if descriptor_path.exists() and not req.force:
                raise HTTPException(
                    409,
                    f"descriptor {descriptor.id!r} already exists. Pass "
                    f"'force': true to overwrite, or pick a different 'id'.",
                )
            descriptor_path.write_text(descriptor_to_python(descriptor))

            # Invalidate any cached import of this module so register_live
            # picks up the freshly-written file.
            import sys as _sys
            _sys.modules.pop(f"catalog.{descriptor.id}", None)
            importlib.invalidate_caches()

            entry = register_live(registry, descriptor)

            built = 0
            if req.build_index:
                index_path = entry.index_path
                index_path.parent.mkdir(parents=True, exist_ok=True)
                built = build_index(src_path, descriptor, index_path)
                # Drop any stale store so the next /overview loads the
                # newly-built index.
                entry.store = None

            return {
                "dataset_id": descriptor.id,
                "descriptor_path": str(descriptor_path),
                "source_path": str(src_path),
                "index_path": str(entry.index_path),
                "indexed": req.build_index,
                "records_indexed": built,
                "inferred": report.fields,
                "notes": report.notes,
                "summary": _entry_summary(descriptor.id, registry),
            }

    # -------- metadata --------

    _models_cache: dict[str, Any] = {"ids": None, "fetched_at": 0.0, "error": None}

    @app.get("/api/models")
    def list_models() -> JSONResponse:
        """List chat-completion models available from the configured LiteLLM
        endpoint. Cached for 5 minutes so a flaky upstream doesn't stall the
        UI on every render."""
        now = time.time()
        if _models_cache["ids"] is not None and (now - _models_cache["fetched_at"]) < 300:
            return JSONResponse({
                "models": _models_cache["ids"],
                "cached": True,
                "analyst_default": cfg.model,
                "synth_default": cfg.synth_model,
            })

        try:
            import openai
            base_url = os.environ.get("LITELLM_BASE_URL", "https://litellm.inference.cool/v1")
            api_key = os.environ.get("LITELLM_API_KEY")
            if not api_key:
                raise RuntimeError("LITELLM_API_KEY not set")
            client = openai.OpenAI(base_url=base_url, api_key=api_key)
            page = client.models.list()
            ids = sorted(m.id for m in page.data)
            _models_cache["ids"] = ids
            _models_cache["fetched_at"] = now
            _models_cache["error"] = None
        except Exception as e:
            logger.exception("failed to list LiteLLM models")
            _models_cache["error"] = str(e)
            # Fall back to whatever we cached last, or the configured defaults.
            fallback = _models_cache["ids"] or sorted({cfg.model, cfg.synth_model})
            return JSONResponse({
                "models": fallback,
                "cached": _models_cache["ids"] is not None,
                "error": str(e),
                "analyst_default": cfg.model,
                "synth_default": cfg.synth_model,
            })

        return JSONResponse({
            "models": ids,
            "cached": False,
            "analyst_default": cfg.model,
            "synth_default": cfg.synth_model,
        })

    @app.get("/api/datasets")
    def list_datasets() -> JSONResponse:
        return JSONResponse({
            "datasets": [_entry_summary(did, registry) for did in registry.ids()],
            "default_id": cfg.default_dataset_id if cfg.default_dataset_id in registry.entries
                           else (registry.ids()[0] if registry.ids() else None),
            # Bubble catalog-load errors up so the UI can show bad
            # descriptors instead of silently dropping them.
            "catalog_errors": registry.errors,
        })

    def _require_indexed(dataset_id: str) -> IndexStore:
        if dataset_id not in registry.entries:
            raise HTTPException(404, f"unknown dataset '{dataset_id}'")
        try:
            return registry.load_store(dataset_id)
        except FileNotFoundError as e:
            raise HTTPException(409, str(e))

    @app.get("/api/datasets/{dataset_id}/overview")
    def dataset_overview(dataset_id: str) -> JSONResponse:
        store = _require_indexed(dataset_id)
        d = registry.entries[dataset_id].descriptor
        return JSONResponse({
            **_entry_summary(dataset_id, registry),
            "analyst_model": cfg.model,
            "synth_model": cfg.synth_model,
            "indexed_traces": len(store),
            "overview": store.overview(),
            "seed_questions": d.seed_questions,
        })

    @app.get("/api/datasets/{dataset_id}/traces")
    def list_traces(
        dataset_id: str,
        limit: int = Query(20, ge=1, le=100),
        outcome: str | None = Query(None, description="zero | partial | perfect | any"),
        seed: int = 42,
    ) -> JSONResponse:
        store = _require_indexed(dataset_id)
        d = registry.entries[dataset_id].descriptor

        # Reject unknown ``outcome`` values so the client sees a clear 400
        # instead of getting silent "all rows" behaviour (fix #18).
        if outcome not in (None, "", "any", "perfect", "partial", "zero"):
            raise HTTPException(
                400, f"invalid outcome {outcome!r}; expected one of: perfect, partial, zero, any",
            )
        bucket = outcome if outcome in {"perfect", "partial", "zero"} and d.primary_metric is not None else None
        rows = store.filter(outcome_bucket=bucket)
        total = len(rows)
        if len(rows) > limit:
            rng = random.Random(seed)
            rows = rng.sample(rows, limit)
        return JSONResponse({
            "total": total,
            "returned": len(rows),
            "traces": [asdict(r) for r in rows],
        })

    @app.get("/api/datasets/{dataset_id}/traces/{trace_id}")
    def get_trace(dataset_id: str, trace_id: str) -> JSONResponse:
        store = _require_indexed(dataset_id)
        descriptor = registry.entries[dataset_id].descriptor
        if store.lookup(trace_id) is None:
            raise HTTPException(404, "unknown trace id")
        with TraceReader(descriptor.source_path) as reader:
            ctx = ToolContext(
                descriptor=descriptor,
                store=store,
                reader=reader,
                synth_model=cfg.synth_model,
                synth_trace_cap=cfg.synth_trace_cap,
                synth_chars_per_trace=cfg.synth_chars_per_trace,
                sample_cap=cfg.sample_cap,
            )
            data = tool_inspect_trace(ctx, id=trace_id, content_chars=800)
        return JSONResponse(data)

    # -------- agent run --------

    @app.get("/api/datasets/{dataset_id}/ask")
    async def ask(
        dataset_id: str,
        question: str = Query(..., min_length=3, max_length=2000),
        conversation_id: str | None = None,
        model: str | None = None,
        synth_model: str | None = None,
        max_turns: int | None = None,
    ) -> EventSourceResponse:
        store = _require_indexed(dataset_id)
        descriptor = registry.entries[dataset_id].descriptor

        per_cfg = InferenceConfig()
        per_cfg.init()
        per_cfg.default_dataset_id = dataset_id
        per_cfg.model = model or cfg.model
        per_cfg.synth_model = synth_model or cfg.synth_model
        per_cfg.max_turns = max_turns or cfg.max_turns
        per_cfg.sample_cap = cfg.sample_cap
        per_cfg.synth_trace_cap = cfg.synth_trace_cap
        per_cfg.synth_chars_per_trace = cfg.synth_chars_per_trace

        try:
            conv = conversations.get_or_create(conversation_id, dataset_id)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

        async def event_stream() -> AsyncIterator[dict[str, Any]]:
            buffered: list[dict[str, Any]] = []
            final: dict[str, Any] | None = None
            started_at = time.time()
            # The agent generator is synchronous and each ``next()`` call
            # blocks on LLM HTTP requests for seconds. If we iterated it
            # directly here, the event loop would freeze — every other
            # HTTP endpoint (history, traces, dataset list) would queue
            # behind the LLM call and the UI would appear to hang. So we
            # push each step onto a worker thread via ``anyio.to_thread``
            # and await the result; the loop stays free.
            _sentinel = object()

            def _next_event(gen: Any) -> Any:
                try:
                    return next(gen)
                except StopIteration:
                    return _sentinel

            # Serialize concurrent /ask requests that share the same
            # conv_id. Without this, both runs mutate ``conv.messages`` and
            # ``conv.result_store`` in parallel and the final ``save()``
            # persists a torn state.
            lock = _conv_lock(conv.id)
            persist_errors: list[str] = []
            try:
                await lock.acquire()
                # Snapshot the pre-run messages length so we can truncate a
                # dangling ``user`` message if the run errors out before
                # producing any assistant reply.
                messages_len_before = len(conv.messages)
                run_completed_cleanly = False
                try:
                    with TraceReader(descriptor.source_path) as reader:
                        gen = run_agent(
                            question, per_cfg,
                            descriptor=descriptor, store=store, reader=reader,
                            # Pass the conversation's mutable messages list so
                            # run_agent continues prior context and appends the
                            # new turn's assistant/tool messages in-place.
                            messages=conv.messages,
                            # Share the persistent result_store so older turns'
                            # ``r_N`` keys still resolve via ``inspect_result``.
                            result_store=conv.result_store,
                            next_result_key=conv.next_result_key,
                        )
                        while True:
                            event = await anyio.to_thread.run_sync(_next_event, gen)
                            if event is _sentinel:
                                break
                            if event.kind == "start":
                                event.data["conversation_id"] = conv.id
                            buffered.append({"kind": event.kind, "data": event.data})
                            if event.kind == "final":
                                final = event.data
                            yield {
                                "event": event.kind,
                                "data": json.dumps(event.data, ensure_ascii=False, default=str),
                            }
                    run_completed_cleanly = True
                finally:
                    completed_at = time.time()
                    # If the run errored out before emitting an assistant
                    # reply, roll back the user message so the persisted
                    # conversation doesn't end on a dangling ``user`` turn
                    # (which would break role-alternation on next resume).
                    if not run_completed_cleanly:
                        produced_assistant = any(
                            m.get("role") == "assistant"
                            for m in conv.messages[messages_len_before:]
                        )
                        if not produced_assistant:
                            del conv.messages[messages_len_before:]
                    try:
                        content = (final or {}).get("content")
                        conv.turns.append({
                            "question": question,
                            "started_at": started_at,
                            "completed_at": completed_at,
                            "final_preview": content[:240] if isinstance(content, str) else "",
                            "turns_used": (final or {}).get("turns_used"),
                            # Store the per-turn delta, not the run's
                            # cumulative total, so summing conv.turns reflects
                            # the actual spend (fix #11).
                            "cost": (final or {}).get("total_cost"),
                        })
                        if conv.result_store:
                            max_n = max(
                                (int(k.split("_")[-1]) for k in conv.result_store
                                 if k.startswith("r_") and k.split("_")[-1].isdigit()),
                                default=conv.next_result_key - 1,
                            )
                            conv.next_result_key = max(conv.next_result_key, max_n + 1)
                        conversations.save(conv)
                    except Exception as e:  # noqa: BLE001
                        logger.exception(
                            "failed to persist conversation {} after run",
                            conv.id,
                        )
                        persist_errors.append(f"conversation save failed: {type(e).__name__}: {e}")
                    try:
                        history.append(
                            dataset_id=dataset_id,
                            question=question,
                            events=buffered,
                            final=final,
                            analyst_model=per_cfg.model,
                            synth_model=per_cfg.synth_model,
                            started_at=started_at,
                            completed_at=completed_at,
                            conversation_id=conv.id,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.exception("failed to persist history row for run")
                        persist_errors.append(f"history append failed: {type(e).__name__}: {e}")
                    lock.release()
            except asyncio.CancelledError:
                # Client disconnected. Don't try to yield more — the stream
                # is already gone. Persistence still ran above, so nothing
                # is lost except the trailing ``warn`` event.
                raise

            # Emit persist errors OUTSIDE the finally so a GeneratorExit
            # from a client disconnect doesn't crash on ``yield`` after
            # close. If the client is still connected, they'll see it.
            if persist_errors:
                yield {
                    "event": "warn",
                    "data": json.dumps({"persist_errors": persist_errors}),
                }

        return EventSourceResponse(event_stream())

    # -------- conversations --------

    @app.get("/api/conversations")
    def list_conversations(dataset_id: str | None = None) -> JSONResponse:
        return JSONResponse({
            "conversations": [c.summary() for c in conversations.list(dataset_id)],
        })

    @app.get("/api/conversations/{conv_id}")
    def get_conversation(conv_id: str) -> JSONResponse:
        conv = conversations.get(conv_id)
        if conv is None:
            raise HTTPException(404, "unknown conversation id")
        # Deep-copy under the store's lock so a concurrent /ask appending
        # to the same ``messages``/``turns`` list doesn't leave the
        # JSONResponse serializer looking at a torn half-written dict
        # (fix #12).
        import copy
        snapshot = copy.deepcopy({
            "messages": conv.messages,
            "turns": conv.turns,
        })
        return JSONResponse({**conv.summary(), **snapshot})

    @app.delete("/api/conversations/{conv_id}")
    def delete_conversation(conv_id: str) -> JSONResponse:
        if not conversations.delete(conv_id):
            raise HTTPException(404, "unknown conversation id")
        return JSONResponse({"deleted": conv_id})

    # -------- ingest --------

    @app.post("/api/datasets/ingest")
    def ingest_by_path(body: IngestRequest) -> JSONResponse:
        """Register a dataset from a server-side JSONL path."""
        src_path = Path(body.path).expanduser().resolve()
        return JSONResponse(_ingest_from_path(body, src_path))

    @app.post("/api/datasets/upload")
    async def ingest_by_upload(
        file: UploadFile = File(..., description="The JSONL file to ingest."),
        id: str | None = Form(None),
        name: str | None = Form(None),
        source_model: str | None = Form(None),
        description: str | None = Form(None),
        sample: int = Form(200),
        build_index: bool = Form(True),
        force: bool = Form(False),
    ) -> JSONResponse:
        """Upload a JSONL file, then ingest it like /api/datasets/ingest."""
        upload_dir.mkdir(parents=True, exist_ok=True)
        original = file.filename or "uploaded.jsonl"
        safe_stem = _sanitize_slug(Path(original).stem) or "uploaded"
        target = upload_dir / f"{safe_stem}.jsonl"
        # Stream the upload to disk to avoid memory blow-ups on large files.
        written = 0
        with target.open("wb") as out:
            while True:
                chunk = await file.read(1 << 20)  # 1 MiB at a time
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
        req = IngestRequest(
            path=str(target),
            id=id or safe_stem,
            name=name or safe_stem,
            source_model=source_model,
            description=description,
            sample=sample,
            build_index=build_index,
            force=force,
        )
        result = _ingest_from_path(req, target)
        result["uploaded_bytes"] = written
        return JSONResponse(result)

    # -------- history --------

    @app.get("/api/history")
    def list_history(
        limit: int = Query(100, ge=1, le=500),
        dataset_id: str | None = None,
    ) -> JSONResponse:
        rows = history.list(limit=limit)
        if dataset_id:
            rows = [r for r in rows if r.get("dataset_id") == dataset_id]
        return JSONResponse({"runs": rows})

    @app.get("/api/history/{run_id}")
    def get_history(run_id: str) -> JSONResponse:
        row = history.get(run_id)
        if row is None:
            raise HTTPException(404, "unknown run id")
        return JSONResponse(row)

    @app.delete("/api/history/{run_id}")
    def delete_history(run_id: str) -> JSONResponse:
        if not history.remove(run_id):
            raise HTTPException(404, "unknown run id")
        return JSONResponse({"deleted": run_id})

    # -------- delete dataset --------

    @app.delete("/api/datasets/{dataset_id}")
    def delete_dataset(
        dataset_id: str,
        keep_index: bool = Query(False, description="Leave the index JSONL on disk."),
        keep_history: bool = Query(False, description="Leave this dataset's history rows on disk."),
    ) -> JSONResponse:
        """Unregister a dataset and remove its descriptor + index + uploaded file.

        The original source JSONL is *not* touched if it lives outside
        ``data/uploads/``; we only own files we wrote. Uploaded JSONLs
        under ``data/uploads/`` are removed alongside the index.
        """
        if dataset_id not in registry.entries:
            raise HTTPException(404, f"unknown dataset '{dataset_id}'")
        entry = registry.entries[dataset_id]
        removed: dict[str, Any] = {"dataset_id": dataset_id, "removed": []}

        with _INGEST_LOCK:
            descriptor_file = catalog_dir / f"{dataset_id}.py"
            if descriptor_file.exists():
                descriptor_file.unlink()
                removed["removed"].append(str(descriptor_file))
            if not keep_index and entry.index_path.exists():
                entry.index_path.unlink()
                removed["removed"].append(str(entry.index_path))

            # If the source JSONL was an upload we own, clean it up too.
            try:
                src = entry.descriptor.source_path.resolve()
                if src.is_relative_to(upload_dir.resolve()) and src.exists():
                    src.unlink()
                    removed["removed"].append(str(src))
            except (AttributeError, ValueError):
                # is_relative_to is 3.9+; resolve() can raise. Ignore.
                pass

            import sys as _sys
            _sys.modules.pop(f"catalog.{dataset_id}", None)
            registry.entries.pop(dataset_id, None)

            if not keep_history:
                removed["history_rows_removed"] = history.remove_for_dataset(dataset_id)

            # Clear saved conversations pointing at this dataset so future
            # ``get_or_create`` requests don't revive orphan state (fix #5).
            removed["conversations_removed"] = conversations.clear_for_dataset(dataset_id)

        return JSONResponse(removed)

    # -------- static UI --------

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    app = build_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
