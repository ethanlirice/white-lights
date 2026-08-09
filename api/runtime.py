"""Shared inference runtime: a bounded pool of warm models, and latency stats.

`ws_live` used to construct a `PoseEstimator` per WebSocket connection and hand
CPU-bound inference to asyncio's *default* executor. Three problems with that:

  * **Cold start per user.** The model loads lazily on first inference, so every
    new connection paid ~a second before its first frame was judged.
  * **Unbounded growth.** Nothing capped connections, models, or threads. The
    weights themselves are small (yolo11n-pose is a few MB — torch's ~2 GB is
    per *process*, not per model), but each instance still carries inference
    buffers and load time, and the default executor's thread count scales with
    CPU count regardless of how many of them torch can usefully run at once.
  * **No back-pressure.** Under load every client degraded together, silently,
    instead of the server saying it was full.

This module fixes all three with a fixed-size pool of pre-warmed estimators.
A pool rather than one shared instance is deliberate: ultralytics predictors
carry mutable per-call state and are not safe to drive from several threads at
once, so each worker borrows its own. Pool size therefore bounds *both* memory
and concurrency, and it is the single number to turn.

Everything degrades gracefully without the ``cv`` extra installed: warming is
best-effort, and the missing-runtime error still surfaces on the request path
exactly as it did before.
"""

from __future__ import annotations

import logging
import os
import queue
import statistics
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from whitelights.pose import PoseEstimator

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ[name]))
    except (KeyError, ValueError):
        return default


#: Concurrent inferences, and therefore models held in memory.
MAX_WORKERS = _env_int("WL_MAX_WORKERS", 2)
#: Live sockets accepted before new ones are shed with an explicit "busy".
MAX_CONNECTIONS = _env_int("WL_MAX_CONNECTIONS", 4)


class ServerBusy(RuntimeError):
    """Raised when the connection cap is reached. Shed, do not queue."""


@dataclass
class StageTimings:
    """Per-frame cost, split by stage so the bottleneck is visible."""

    decode_ms: float = 0.0
    inference_ms: float = 0.0
    judge_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.decode_ms + self.inference_ms + self.judge_ms


@dataclass
class LatencyStats:
    """Rolling per-stage latency over the most recent frames.

    A bounded window, so a long-running server reports *current* behaviour
    rather than an average smeared over everything since boot.
    """

    window: int = 512
    _decode: deque[float] = field(default_factory=deque)
    _inference: deque[float] = field(default_factory=deque)
    _judge: deque[float] = field(default_factory=deque)
    _total: deque[float] = field(default_factory=deque)
    frames: int = 0

    def __post_init__(self) -> None:
        for name in ("_decode", "_inference", "_judge", "_total"):
            setattr(self, name, deque(maxlen=self.window))

    def record(self, timings: StageTimings) -> None:
        self._decode.append(timings.decode_ms)
        self._inference.append(timings.inference_ms)
        self._judge.append(timings.judge_ms)
        self._total.append(timings.total_ms)
        self.frames += 1

    @staticmethod
    def _percentiles(samples: deque[float]) -> dict[str, float]:
        if not samples:
            return {}
        ordered = sorted(samples)

        def at(q: float) -> float:
            idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
            return round(ordered[idx], 2)

        return {
            "p50": round(statistics.median(ordered), 2),
            "p95": at(0.95),
            "p99": at(0.99),
            "max": round(ordered[-1], 2),
        }

    def snapshot(self) -> dict[str, Any]:
        total = self._percentiles(self._total)
        achievable_fps = round(1000.0 / total["p50"], 1) if total.get("p50") else None
        return {
            "frames_processed": self.frames,
            "window": len(self._total),
            "decode_ms": self._percentiles(self._decode),
            "inference_ms": self._percentiles(self._inference),
            "judge_ms": self._percentiles(self._judge),
            "total_ms": total,
            "achievable_fps_at_p50": achievable_fps,
        }


class InferenceRuntime:
    """Owns the model pool, the worker threads, and the connection budget."""

    def __init__(
        self, *, max_workers: int = MAX_WORKERS, max_connections: int = MAX_CONNECTIONS
    ) -> None:
        self.max_workers = max_workers
        self.max_connections = max_connections
        self.latency = LatencyStats()
        self._pool: queue.Queue[PoseEstimator] = queue.Queue()
        self._executor: ThreadPoolExecutor | None = None
        self._connections = 0
        self._warm = False

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Build the pool and pre-load weights. Safe without the ``cv`` extra."""
        from whitelights.pose import PoseEstimator

        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="wl-infer"
        )
        for _ in range(self.max_workers):
            self._pool.put(PoseEstimator())
        self._warm = self._warm_up()

    def _warm_up(self) -> bool:
        """Touch each model so the first real frame is not the one that pays.

        Returns whether warming succeeded; a missing pose runtime is expected in
        CI and dev, and is reported on the request path instead of at boot.
        """
        estimators = [self._pool.get() for _ in range(self._pool.qsize())]
        try:
            for estimator in estimators:
                _ = estimator.model  # forces the lazy ultralytics/torch import
        except ModuleNotFoundError as exc:
            logger.warning("pose runtime unavailable (%s); models will not be warmed", exc.name)
            return False
        except Exception:  # noqa: BLE001 - warming must never block startup
            logger.exception("model warm-up failed; continuing cold")
            return False
        else:
            logger.info("warmed %d pose model(s)", len(estimators))
            return True
        finally:
            for estimator in estimators:
                self._pool.put(estimator)

    def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    # -- connection budget ---------------------------------------------------

    @contextmanager
    def connection_slot(self):
        """Reserve one of the live-connection slots, or raise `ServerBusy`.

        Shedding beats queueing here: a client that waits in line for a
        real-time video socket gets a worse experience than one told plainly to
        try again.
        """
        if self._connections >= self.max_connections:
            raise ServerBusy(
                f"server is at capacity ({self.max_connections} live sessions); try again shortly"
            )
        self._connections += 1
        try:
            yield
        finally:
            self._connections -= 1

    # -- inference -----------------------------------------------------------

    @contextmanager
    def borrow(self):
        """Check out one estimator, guaranteeing it is used by one thread only."""
        estimator = self._pool.get()
        try:
            yield estimator
        finally:
            self._pool.put(estimator)

    @property
    def executor(self) -> ThreadPoolExecutor:
        if self._executor is None:  # tests may exercise the app without lifespan
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="wl-infer"
            )
            from whitelights.pose import PoseEstimator

            for _ in range(self.max_workers):
                self._pool.put(PoseEstimator())
        return self._executor

    def status(self) -> dict[str, Any]:
        return {
            "workers": self.max_workers,
            "max_connections": self.max_connections,
            "active_connections": self._connections,
            "models_warm": self._warm,
            "latency": self.latency.snapshot(),
        }


@contextmanager
def timed(timings: StageTimings, stage: str):
    """Accumulate wall time for one stage onto ``timings``."""
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - started) * 1000.0
        setattr(timings, stage, getattr(timings, stage) + elapsed)


def measure(fn: Callable[[], Any], timings: StageTimings, stage: str) -> Any:
    with timed(timings, stage):
        return fn()
