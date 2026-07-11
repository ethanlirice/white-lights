"""Tests for the (lift, mode) -> tracker factory."""

from __future__ import annotations

from whitelights.bench import BenchTracker
from whitelights.deadlift import DeadliftTracker
from whitelights.judges import tracker_for
from whitelights.live import CompetitionTracker, OnlineRepTracker


def test_squat_training() -> None:
    assert isinstance(tracker_for("squat", "training"), OnlineRepTracker)


def test_squat_competition() -> None:
    assert isinstance(tracker_for("squat", "competition"), CompetitionTracker)


def test_bench_routes_to_bench_tracker() -> None:
    assert isinstance(tracker_for("bench", "competition"), BenchTracker)
    assert isinstance(tracker_for("bench", "training"), BenchTracker)  # only judge for now


def test_deadlift_routes_to_deadlift_tracker() -> None:
    assert isinstance(tracker_for("deadlift", "competition"), DeadliftTracker)


def test_defaults_and_unknown_fall_back_to_squat_training() -> None:
    assert isinstance(tracker_for(None, None), OnlineRepTracker)
    assert isinstance(tracker_for("CURL", "whatever"), OnlineRepTracker)
