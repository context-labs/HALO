"""
Reusable batch processing: progress tracking, cost tracking, rate limiting,
worker orchestration, and graceful shutdown.

The main entry point is run_batch(), which ties everything together.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

__all__ = ["ProgressTracker", "RateLimiter", "GracefulShutdown", "run_batch"]

logger = logging.getLogger(__name__)

_ALPHA_DECAY = 0.1
_ALPHA_FLOOR = 0.002


class AdaptiveETAColumn(ProgressColumn):
    """ETA column using an adaptive EMA of inter-completion intervals.

    Alpha decays as completions accumulate: responsive early, stable late.
    Floor of 0.002 keeps the estimate from freezing on very long runs while
    smoothing out per-iteration noise from parallel worker timing.
    """

    def __init__(self) -> None:
        super().__init__()
        self._smoothed_spi: dict[int, float] = {}
        self._n_completed: dict[int, int] = {}
        self._last_time: dict[int, float] = {}
        self._lock = threading.Lock()

    def record_completion(self, task_id: int) -> None:
        """Record a task completion and update the EMA for ETA estimation."""
        now = time.monotonic()
        with self._lock:
            n = self._n_completed.get(task_id, 0)
            last = self._last_time.get(task_id)

            if last is not None:
                interval = now - last
                alpha = max(_ALPHA_FLOOR, 1.0 / (1.0 + n * _ALPHA_DECAY))
                prev = self._smoothed_spi.get(task_id, interval)
                self._smoothed_spi[task_id] = alpha * interval + (1.0 - alpha) * prev

            self._last_time[task_id] = now
            self._n_completed[task_id] = n + 1

    def render(self, task: Task) -> Text:
        with self._lock:
            spi = self._smoothed_spi.get(task.id)

        if spi is None or task.total is None:
            return Text("-:--:--", style="cyan")

        remaining = task.total - task.completed
        eta_seconds = max(0, spi * remaining)
        hours, remainder = divmod(int(eta_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return Text(f"{hours}:{minutes:02d}:{seconds:02d}", style="cyan")


@dataclass
class ProgressTracker:
    """Track progress and cost for a batch run.

    Cost is accumulated from caller-provided values — the tracker does not
    compute cost itself. This makes it compatible with any pricing model
    (per-token, per-request, pre-computed, or free).

    For free approaches, simply omit the cost argument — cost stays at zero
    and has_cost remains False, so callers can suppress cost display.
    """

    samples_processed: int = field(default=0, init=False)
    samples_failed: int = field(default=0, init=False)
    total_cost: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    @property
    def has_cost(self) -> bool:
        return self.total_cost > 0

    @property
    def total_samples(self) -> int:
        return self.samples_processed + self.samples_failed

    @property
    def avg_cost_per_sample(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.total_cost / self.total_samples

    def record_success(self, cost: float = 0.0) -> None:
        """Record a successful processing result, optionally adding its cost."""
        with self._lock:
            self.samples_processed += 1
            self.total_cost += cost

    def record_failure(self, cost: float = 0.0) -> None:
        """Record a failed processing result, optionally adding its cost."""
        with self._lock:
            self.samples_failed += 1
            self.total_cost += cost

    def estimate_total_cost(self, total_samples: int) -> float:
        """Extrapolate: if we've done N of M samples at $X, total is $X * M/N."""
        if self.total_samples == 0:
            return 0.0
        return total_samples * self.avg_cost_per_sample

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracker state to a plain dict for JSON checkpointing."""
        return {
            "samples_processed": self.samples_processed,
            "samples_failed": self.samples_failed,
            "total_cost": self.total_cost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProgressTracker:
        """Restore a ProgressTracker from a dict produced by ``to_dict()``."""
        tracker = cls()
        tracker.samples_processed = data.get("samples_processed", 0)
        tracker.samples_failed = data.get("samples_failed", 0)
        tracker.total_cost = data.get("total_cost", 0.0)
        return tracker


class RateLimiter:
    """Thread-safe fixed-interval rate limiter."""

    def __init__(self, max_rps: float) -> None:
        self.min_interval = 1.0 / max_rps
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        """Block until the rate limit interval has elapsed since the last call."""
        with self._lock:
            now = time.monotonic()
            wait_time = self._last + self.min_interval - now
            if wait_time > 0:
                self._last += self.min_interval
            else:
                self._last = now
        if wait_time > 0:
            time.sleep(wait_time)


class GracefulShutdown:
    """Context manager for graceful SIGINT/SIGTERM shutdown.

    Usage:
        with GracefulShutdown() as shutdown:
            while not shutdown.is_set():
                ...
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._original_handlers: dict[int, Any] = {}

    def install(self) -> GracefulShutdown:
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handler)
        return self

    def _handler(self, signum: int, frame: object) -> None:
        try:
            name = signal.Signals(signum).name
        except (ValueError, AttributeError):
            name = str(signum)
        logger.warning("Received %s, shutting down gracefully...", name)
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def restore(self) -> None:
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        self._original_handlers.clear()

    def __enter__(self) -> GracefulShutdown:
        return self.install()

    def __exit__(self, *exc: object) -> None:
        self.restore()


def run_batch(
    items: Iterable[Any],
    process_fn: Callable[[Any], Any],
    *,
    workers: int = 1,
    max_rps: float | None = None,
    total: int | None = None,
    desc: str = "",
    cost_fn: Callable[[Any], float] | None = None,
    error_fn: Callable[[Any], bool] | None = None,
    on_result: Callable[[Any, Any], None] | None = None,
    on_checkpoint: Callable[[list[tuple[Any, Any]]], None] | None = None,
    checkpoint_every: int = 20,
    shutdown: GracefulShutdown | None = None,
) -> ProgressTracker:
    """Run items through a processing function with progress, cost, and shutdown handling.

    Args:
        items: Iterable of items to process (any type).
        process_fn: Takes one item, returns one result (any type).
        workers: Concurrency level. 1 = sequential, N = parallel with backpressure.
        max_rps: Rate limit in requests/sec. None = no limit.
        total: Total item count for the progress bar. Inferred from len(items) if possible.
        desc: Label for the progress bar.
        cost_fn: Extract cost (float) from a result. If None, no cost tracking.
        error_fn: Return True if a result is a failure. If None, all results are successes.
        on_result: Called with (item, result) after each completion. Use for
            lightweight in-memory bookkeeping (e.g. updating a seen-set).
        on_checkpoint: Called with list[(item, result)] every checkpoint_every
            items and once on shutdown with any remaining buffered results.
            Use for batched disk I/O.
        checkpoint_every: How many results to buffer before calling on_checkpoint.
        shutdown: GracefulShutdown instance. If None, SIGINT is not handled.

    Returns:
        ProgressTracker with final counts and cost.
    """
    if total is None and hasattr(items, "__len__"):
        total = len(items)  # type: ignore[arg-type]

    tracker = ProgressTracker()
    limiter = RateLimiter(max_rps) if max_rps else None
    checkpoint_buffer: list[tuple[Any, Any]] = []

    eta_column = AdaptiveETAColumn()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None, complete_style="cyan", finished_style="green"),
        MofNCompleteColumn(),
        TextColumn("•", style="dim"),
        TimeElapsedColumn(),
        TextColumn("eta", style="dim"),
        eta_column,
        TextColumn("{task.fields[stats]}"),
    )
    task_id = progress.add_task(desc or "Processing", total=total, stats="")
    progress.start()

    def _update_stats() -> None:
        parts = [f"[green]✓ {tracker.samples_processed}[/]"]
        if tracker.samples_failed > 0:
            parts.append(f"[red]✗ {tracker.samples_failed}[/]")
        if tracker.has_cost and total and tracker.total_samples > 0:
            parts.append(f"[cyan]${tracker.total_cost:.4f}[/]")
            parts.append(f"[dim]est ${tracker.estimate_total_cost(total):.2f}[/]")
        progress.update(task_id, stats="  ".join(parts))

    def _flush_checkpoint() -> None:
        if on_checkpoint and checkpoint_buffer:
            on_checkpoint(checkpoint_buffer[:])
            checkpoint_buffer.clear()

    def _handle(item: object, result: object) -> None:
        cost = cost_fn(result) if cost_fn else 0.0
        is_error = error_fn(result) if error_fn else False

        if is_error:
            tracker.record_failure(cost=cost)
        else:
            tracker.record_success(cost=cost)

        if on_result:
            on_result(item, result)

        if on_checkpoint:
            checkpoint_buffer.append((item, result))
            if len(checkpoint_buffer) >= checkpoint_every:
                _flush_checkpoint()

        eta_column.record_completion(task_id)
        progress.update(task_id, advance=1)
        _update_stats()

    def _handle_error(item: object) -> None:
        tracker.record_failure()
        eta_column.record_completion(task_id)
        progress.update(task_id, advance=1)
        _update_stats()

    try:
        if workers <= 1:
            _run_sequential(items, process_fn, desc, limiter, shutdown, _handle, _handle_error)
        else:
            _run_parallel(
                items, process_fn, desc, workers, total, limiter, shutdown, _handle, _handle_error
            )
    finally:
        _flush_checkpoint()
        progress.stop()

    return tracker


def _run_sequential(
    items: Iterable[Any],
    process_fn: Callable[[Any], Any],
    desc: str,
    limiter: RateLimiter | None,
    shutdown: GracefulShutdown | None,
    handle: Callable[[Any, Any], None],
    handle_error: Callable[[Any], None],
) -> None:
    for item in items:
        if shutdown and shutdown.is_set():
            logger.warning("[%s] Shutdown requested", desc)
            break
        if limiter:
            limiter.acquire()
        try:
            result = process_fn(item)
            handle(item, result)
        except Exception:
            logger.exception("[%s] Unhandled error processing item", desc)
            handle_error(item)


def _run_parallel(
    items: Iterable[Any],
    process_fn: Callable[[Any], Any],
    desc: str,
    workers: int,
    total: int | None,
    limiter: RateLimiter | None,
    shutdown: GracefulShutdown | None,
    handle: Callable[[Any, Any], None],
    handle_error: Callable[[Any], None],
) -> None:
    batch_size = workers * 2
    item_iter = iter(items)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[Any, Any] = {}

        for _ in range(min(batch_size, total or batch_size)):
            try:
                item = next(item_iter)
                if limiter:
                    limiter.acquire()
                futures[executor.submit(process_fn, item)] = item
            except StopIteration:
                break

        while futures:
            if shutdown and shutdown.is_set():
                logger.warning("[%s] Shutdown requested", desc)
                executor.shutdown(wait=False, cancel_futures=True)
                break

            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED, timeout=1.0)

            for future in done:
                item = futures.pop(future)
                try:
                    result = future.result()
                    handle(item, result)
                except Exception:
                    logger.exception("[%s] Unhandled error processing item", desc)
                    handle_error(item)

                try:
                    next_item = next(item_iter)
                    if limiter:
                        limiter.acquire()
                    futures[executor.submit(process_fn, next_item)] = next_item
                except StopIteration:
                    pass
