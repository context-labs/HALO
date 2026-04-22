"""Conversation store: disk-persisted, in-memory cached, thread-safe.

A *conversation* is the state needed to continue talking to the analyst —
specifically the growing ``messages`` list (OpenAI chat format) plus the
dataset it's tied to. When the client sends a new question with an existing
``conversation_id``, the server looks up the conversation, appends the new
user message, lets ``run_agent`` mutate the list with assistant/tool turns,
and keeps the updated list for the next request.

Each conversation is persisted as ``<index_dir>/conversations/<id>.json``
so sessions survive process restarts. Load-all-on-boot is fine for the
scale we target (hundreds of conversations tops); if that ever stops being
enough we can switch to a SQLite store without touching callers.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class Conversation:
    """One conversation's mutable state."""

    id: str
    dataset_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    # Per-turn summary rows for the sidebar list view (without raw events).
    turns: list[dict[str, Any]] = field(default_factory=list)
    # Full tool-result JSON keyed by the short ``r_N`` handle emitted in
    # compacted tool messages. Persisted so ``inspect_result`` still
    # resolves across server restarts.
    result_store: dict[str, Any] = field(default_factory=dict)
    next_result_key: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def last_question(self) -> str | None:
        for m in reversed(self.messages):
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    return content
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "last_question": self.last_question,
            "turn_count": len(self.turns),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ConversationStore:
    """Thread-safe disk-backed store of conversations.

    Writes are atomic via ``<file>.tmp`` → ``rename``. Reads short-circuit
    through an in-memory cache; the disk copy is the source of truth.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._data: OrderedDict[str, Conversation] = OrderedDict()
        self._lock = threading.Lock()
        self._load_all()

    # -- disk --

    def _path(self, conv_id: str) -> Path:
        return self.base_dir / f"{conv_id}.json"

    def _persist(self, conv: Conversation) -> None:
        path = self._path(conv.id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(asdict(conv), ensure_ascii=False, default=str))
            tmp.replace(path)
        except Exception:  # pragma: no cover - disk errors
            logger.exception("failed to persist conversation {}", conv.id)

    def _delete_file(self, conv_id: str) -> None:
        path = self._path(conv_id)
        try:
            if path.exists():
                path.unlink()
        except Exception:  # pragma: no cover
            logger.exception("failed to delete conversation file {}", conv_id)

    def _load_all(self) -> None:
        loaded = 0
        for p in sorted(self.base_dir.glob("*.json")):
            try:
                raw = json.loads(p.read_text())
                conv = Conversation(
                    id=raw.get("id") or p.stem,
                    dataset_id=raw.get("dataset_id", ""),
                    messages=raw.get("messages") or [],
                    turns=raw.get("turns") or [],
                    result_store=raw.get("result_store") or {},
                    next_result_key=int(raw.get("next_result_key") or 0),
                    created_at=float(raw.get("created_at") or 0.0),
                    updated_at=float(raw.get("updated_at") or 0.0),
                )
                self._data[conv.id] = conv
                loaded += 1
            except Exception:  # pragma: no cover
                logger.exception("failed to load conversation file {}", p)
        # Keep the internal OrderedDict sorted by updated_at ascending so
        # list() returns newest first without a re-sort on each call.
        ids_by_mtime = sorted(self._data.keys(),
                              key=lambda k: self._data[k].updated_at)
        for cid in ids_by_mtime:
            self._data.move_to_end(cid)
        if loaded:
            logger.info("ConversationStore loaded {} conversations from {}",
                        loaded, self.base_dir)

    # -- public API --

    def get(self, conv_id: str) -> Conversation | None:
        with self._lock:
            conv = self._data.get(conv_id)
            if conv is not None:
                self._data.move_to_end(conv_id)
            return conv

    def create(self, dataset_id: str, conv_id: str | None = None) -> Conversation:
        now = time.time()
        cid = conv_id or uuid.uuid4().hex[:12]
        with self._lock:
            if cid in self._data:
                self._data.move_to_end(cid)
                return self._data[cid]
            conv = Conversation(
                id=cid, dataset_id=dataset_id, messages=[],
                created_at=now, updated_at=now,
            )
            self._data[cid] = conv
        self._persist(conv)
        return conv

    def get_or_create(self, conv_id: str | None, dataset_id: str) -> Conversation:
        if conv_id:
            existing = self.get(conv_id)
            if existing is not None:
                if existing.dataset_id != dataset_id:
                    raise ValueError(
                        f"conversation {conv_id!r} belongs to dataset "
                        f"{existing.dataset_id!r}, not {dataset_id!r}"
                    )
                return existing
        return self.create(dataset_id, conv_id)

    def save(self, conv: Conversation) -> None:
        """Persist after an in-place mutation (messages/turns append).

        The server calls this at the end of each run so the updated
        message list + new turn summary reach disk atomically.
        """
        with self._lock:
            conv.updated_at = time.time()
            self._data[conv.id] = conv
            self._data.move_to_end(conv.id)
        self._persist(conv)

    def delete(self, conv_id: str) -> bool:
        with self._lock:
            removed = self._data.pop(conv_id, None) is not None
        if removed:
            self._delete_file(conv_id)
        return removed

    def list(self, dataset_id: str | None = None) -> list[Conversation]:
        with self._lock:
            items = list(self._data.values())
        if dataset_id is not None:
            items = [c for c in items if c.dataset_id == dataset_id]
        return sorted(items, key=lambda c: c.updated_at, reverse=True)

    def clear_for_dataset(self, dataset_id: str) -> int:
        with self._lock:
            to_remove = [cid for cid, c in self._data.items()
                         if c.dataset_id == dataset_id]
            for cid in to_remove:
                self._data.pop(cid, None)
        for cid in to_remove:
            self._delete_file(cid)
        return len(to_remove)
