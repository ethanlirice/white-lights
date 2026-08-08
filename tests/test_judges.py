"""Tests for the (lift, mode) -> tracker factory."""

from __future__ import annotations

from whitelights.bench import BenchTracker
from whitelights.deadlift import DeadliftTracker
from whitelights.judges import LIFTS, supports_training, tracker_for
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


def test_only_squat_advertises_a_training_judge() -> None:
    """The UI hides the training toggle off this; it must not over-promise."""
    assert supports_training("squat") is True
    assert supports_training("bench") is False
    assert supports_training("deadlift") is False
    assert supports_training(None) is True  # default lift is squat


def test_training_capable_lifts_actually_return_a_free_rep_judge() -> None:
    """Whatever claims training support must not hand back a single-attempt judge."""
    for lift in LIFTS:
        if supports_training(lift):
            assert isinstance(tracker_for(lift, "training"), OnlineRepTracker)
