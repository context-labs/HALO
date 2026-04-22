"""Persist completed agent runs so users can replay past questions.

Each run is a single JSONL line containing the question, the full event
stream, and a small summary (model / turns / cost) for the list view.

Performance
-----------
The JSONL grows unbounded (~100 KB per run) and the UI refetches
``/api/history`` after every completed run. Re-reading and re-parsing the
entire file on each call added up fast, so the store keeps an in-memory
cache of *summaries* (the small part, without ``events``/``final``).
Full rows (including events) still come from disk when requested — they're
only fetched when the user actually opens a run.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any



class HistoryStore:
    """Append-only JSONL store with an in-memory summary cache."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._summary_cache: list[dict[str, Any]] | None = None

    # -- cache plumbing --

    def _load_summaries_from_disk(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(self._summary(row))
        rows.sort(key=lambda r: r.get("started_at", 0), reverse=True)
        return rows

    def _ensure_cache(self) -> list[dict[str, Any]]:
        # Caller must hold ``self._lock``.
        if self._summary_cache is None:
            self._summary_cache = self._load_summaries_from_disk()
        return self._summary_cache

    def _invalidate_cache(self) -> None:
        # Caller must hold ``self._lock``.
        self._summary_cache = None

    # -- write --

    def append(
        self,
        *,
        question: str,
        events: list[dict[str, Any]],
        final: dict[str, Any] | None,
        analyst_model: str,
        synth_model: str,
        started_at: float,
        completed_at: float,
        dataset_id: str | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Write one run and return its index row (without events)."""
        run_id = uuid.uuid4().hex[:12]

        tool_calls = sum(1 for e in events if e.get("kind") == "tool_call")
        tokens = sum(int(((e.get("data") or {}).get("tokens") or {}).get("total") or 0)
                     for e in events if e.get("kind") == "usage")
        # Prefer ``final.total_cost`` — it includes synth/tool sub-LLM
        # calls (``ctx.cost``) whereas summing per-turn usage events only
        # covers the top-level analyst LLM, under-reporting runs that used
        # ``synthesize``. Fall back to the usage sum for error paths that
        # never emitted ``final`` (fix #10).
        usage_cost = sum(float((e.get("data") or {}).get("cost") or 0)
                         for e in events if e.get("kind") == "usage")
        cost = float((final or {}).get("total_cost") or usage_cost)

        final_preview = ""
        if final:
            content = final.get("content") or ""
            if isinstance(content, str):
                final_preview = content[:240]

        row = {
            "id": run_id,
            "dataset_id": dataset_id,
            "conversation_id": conversation_id,
            "question": question,
            "analyst_model": analyst_model,
            "synth_model": synth_model,
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_s": round(completed_at - started_at, 2),
            "turns_used": (final or {}).get("turns_used"),
            "tool_calls_made": (final or {}).get("tool_calls_made") or tool_calls,
            "tokens": tokens,
            "cost": round(cost, 6),
            "final_preview": final_preview,
            "events": events,
            "final": final,
        }
        summary = self._summary(row)
        with self._lock:
            with self.path.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str))
                f.write("\n")
            # Keep cache in sync without rereading the file.
            cache = self._ensure_cache()
            cache.insert(0, summary)
        return summary

    @staticmethod
    def _summary(row: dict[str, Any]) -> dict[str, Any]:
        """Row shape returned for the list view (no events / final body)."""
        out = {k: v for k, v in row.items() if k not in {"events", "final"}}
        return out

    # -- read --

    def list(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent runs (summary shape), newest first.

        Served from the in-memory cache; the file is only read on a miss
        (first call after startup, or after ``_invalidate_cache``).
        """
        with self._lock:
            cache = self._ensure_cache()
            return list(cache[:limit])

    def get(self, run_id: str) -> dict[str, Any] | None:
        """Return the full row (including events) for one run.

        Always reads from disk — the cache holds summaries only.
        """
        if not self.path.exists():
            return None
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("id") == run_id:
                    return row
        return None

    # -- delete --

    def remove(self, run_id: str) -> bool:
        """Drop the row with id ``run_id``. Rewrites the file atomically."""
        if not self.path.exists():
            return False
        removed = False
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with self._lock:
            with self.path.open() as src, tmp_path.open("w") as dst:
                for line in src:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        row = json.loads(stripped)
                    except json.JSONDecodeError:
                        dst.write(line if line.endswith("\n") else line + "\n")
                        continue
                    if row.get("id") == run_id:
                        removed = True
                        continue
                    dst.write(line if line.endswith("\n") else line + "\n")
            tmp_path.replace(self.path)
            # Update cache in-place instead of invalidating; cheaper.
            cache = self._summary_cache
            if cache is not None and removed:
                self._summary_cache = [r for r in cache if r.get("id") != run_id]
        return removed

    def remove_for_dataset(self, dataset_id: str) -> int:
        """Drop every row whose ``dataset_id`` matches. Returns count removed."""
        if not self.path.exists():
            return 0
        removed = 0
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with self._lock:
            with self.path.open() as src, tmp_path.open("w") as dst:
                for line in src:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        row = json.loads(stripped)
                    except json.JSONDecodeError:
                        dst.write(line if line.endswith("\n") else line + "\n")
                        continue
                    if row.get("dataset_id") == dataset_id:
                        removed += 1
                        continue
                    dst.write(line if line.endswith("\n") else line + "\n")
            tmp_path.replace(self.path)
            cache = self._summary_cache
            if cache is not None and removed:
                self._summary_cache = [
                    r for r in cache if r.get("dataset_id") != dataset_id
                ]
        return removed
