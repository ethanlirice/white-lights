"""Shared inference runtime: capacity, pooling, and latency reporting.

None of this needs the ``cv`` extra — the point of these tests is the resource
behaviour around the model, not the model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.main as main
from api.runtime import InferenceRuntime, LatencyStats, ServerBusy, StageTimings, timed

# ---------------------------------------------------------------------------
# Latency accounting
# ---------------------------------------------------------------------------


def test_stage_timings_total_is_the_sum_of_stages() -> None:
    timings = StageTimings(decode_ms=2.0, inference_ms=40.0, judge_ms=1.5)
    assert timings.total_ms == pytest.approx(43.5)


def test_timed_accumulates_onto_the_named_stage() -> None:
    timings = StageTimings()
    with timed(timings, "inference_ms"):
        sum(range(10_000))
    assert timings.inference_ms > 0.0
    assert timings.decode_ms == 0.0


def test_latency_reports_percentiles_and_achievable_fps() -> None:
    stats = LatencyStats()
    for ms in range(1, 101):  # 1..100 ms
        stats.record(StageTimings(inference_ms=float(ms)))

    snapshot = stats.snapshot()
    assert snapshot["frames_processed"] == 100
    assert snapshot["total_ms"]["p50"] == pytest.approx(50.5, abs=1.0)
    assert snapshot["total_ms"]["p95"] >= snapshot["total_ms"]["p50"]
    assert snapshot["total_ms"]["max"] == pytest.approx(100.0)
    # ~50 ms median -> ~20 fps
    assert snapshot["achievable_fps_at_p50"] == pytest.approx(20.0, abs=1.0)


def test_latency_window_is_bounded() -> None:
    """A long-running server must report current behaviour, not all of history."""
    stats = LatencyStats(window=16)
    for ms in range(200):
        stats.record(StageTimings(inference_ms=float(ms)))

    snapshot = stats.snapshot()
    assert snapshot["frames_processed"] == 200  # lifetime count is kept
    assert snapshot["window"] == 16  # but only the recent window is summarised
    assert snapshot["total_ms"]["max"] == pytest.approx(199.0)


def test_empty_latency_snapshot_is_safe() -> None:
    snapshot = LatencyStats().snapshot()
    assert snapshot["frames_processed"] == 0
    assert snapshot["achievable_fps_at_p50"] is None


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------


def test_connection_slots_are_released_on_exit() -> None:
    runtime = InferenceRuntime(max_workers=1, max_connections=1)
    with runtime.connection_slot():
        assert runtime.status()["active_connections"] == 1
    assert runtime.status()["active_connections"] == 0

    with runtime.connection_slot():  # reusable once freed
        pass


def test_exceeding_capacity_raises_rather_than_queueing() -> None:
    runtime = InferenceRuntime(max_workers=1, max_connections=1)
    with runtime.connection_slot():
        with pytest.raises(ServerBusy, match="capacity"):
            with runtime.connection_slot():
                pass


def test_slot_is_released_even_when_the_body_raises() -> None:
    runtime = InferenceRuntime(max_workers=1, max_connections=1)
    with pytest.raises(ValueError), runtime.connection_slot():
        raise ValueError("boom")
    assert runtime.status()["active_connections"] == 0


def test_pool_is_bounded_by_worker_count() -> None:
    """Models are pooled, not created per caller — the whole point of the change."""
    runtime = InferenceRuntime(max_workers=2, max_connections=8)
    runtime.start()
    try:
        with runtime.borrow() as a, runtime.borrow() as b:
            assert a is not b  # two workers, two distinct models
        # Returned to the pool, so the next borrow reuses one of them.
        with runtime.borrow() as c:
            assert c in (a, b)
    finally:
        runtime.stop()


def test_pool_does_not_double_on_restart() -> None:
    """start() -> stop() -> a later `.executor` access must not re-fill the pool.

    `stop()` tears the executor down but deliberately leaves the pool's
    estimators in place; the lazy `.executor` property exists so tests can
    exercise the app without running lifespan. Deriving "has the pool been
    filled?" from "is `_executor` None?" conflates those two facts — a
    restart-shaped sequence (real or, more likely, two tests sharing one
    module-level runtime) would double the pool every time it recurred,
    silently breaking the one invariant this module promises: pool size
    bounds memory, and it is the single number to turn.
    """
    runtime = InferenceRuntime(max_workers=2, max_connections=4)
    runtime.start()
    runtime.stop()
    assert runtime.executor is not None  # lazy path must not have crashed either
    assert runtime._pool.qsize() == 2

    # And the property alone, accessed repeatedly with no start() at all.
    lazy_only = InferenceRuntime(max_workers=3, max_connections=4)
    _ = lazy_only.executor
    _ = lazy_only.executor
    assert lazy_only._pool.qsize() == 3


def test_start_survives_a_missing_pose_runtime() -> None:
    """CI has no `cv` extra; boot must not depend on it."""
    runtime = InferenceRuntime(max_workers=1, max_connections=1)
    runtime.start()
    try:
        assert runtime.status()["models_warm"] is False
    finally:
        runtime.stop()


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------


def test_metrics_endpoint_reports_capacity_and_latency() -> None:
    with TestClient(main.app) as client:
        body = client.get("/metrics").json()

    assert body["workers"] >= 1
    assert body["max_connections"] >= 1
    assert body["active_connections"] == 0
    assert "latency" in body
    assert "inference_ms" in body["latency"]


def test_socket_beyond_capacity_is_shed_with_a_retry_hint() -> None:
    with TestClient(main.app) as client:
        original = main.runtime.max_connections
        main.runtime.max_connections = 1
        try:
            with client.websocket_connect("/ws/live"):
                with client.websocket_connect("/ws/live") as second:
                    message = second.receive_json()
            assert "error" in message
            assert message["retry"] is True
            assert "capacity" in message["error"]
        finally:
            main.runtime.max_connections = original
