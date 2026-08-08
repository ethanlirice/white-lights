"""Tests for the (lift, mode) -> tracker factory."""

from __future__ import annotations

import re
from pathlib import Path

from whitelights.bench import BenchTracker
from whitelights.deadlift import DeadliftTracker
from whitelights.freereps import FreeRepTracker
from whitelights.judges import LIFTS, supports_training, tracker_for
from whitelights.live import CompetitionTracker, OnlineRepTracker


def test_squat_training() -> None:
    assert isinstance(tracker_for("squat", "training"), OnlineRepTracker)


def test_squat_competition() -> None:
    assert isinstance(tracker_for("squat", "competition"), CompetitionTracker)


def test_bench_routes_by_mode() -> None:
    assert isinstance(tracker_for("bench", "competition"), BenchTracker)
    assert isinstance(tracker_for("bench", "training"), FreeRepTracker)


def test_deadlift_routes_by_mode() -> None:
    assert isinstance(tracker_for("deadlift", "competition"), DeadliftTracker)
    assert isinstance(tracker_for("deadlift", "training"), FreeRepTracker)


def test_ui_lift_modes_match_the_backend() -> None:
    """web/live.html decides which toggles to show; judges.py decides what the
    server will actually run. If they disagree the UI offers a mode the backend
    silently substitutes — which is exactly the bug that motivated this test.
    """
    html = (Path(__file__).resolve().parent.parent / "web" / "live.html").read_text()
    declared = dict(
        re.findall(r"label: '(\w+)',\s*\n\s*modes: \[([^\]]*)\]", html),
    )
    assert set(k.lower() for k in declared) == set(LIFTS), (
        f"web/live.html declares lifts {sorted(declared)}, backend has {sorted(LIFTS)}"
    )
    for label, modes in declared.items():
        offers_training = "'training'" in modes
        assert offers_training == supports_training(label.lower()), (
            f"{label}: UI offers training={offers_training}, "
            f"backend supports_training={supports_training(label.lower())}"
        )


def test_defaults_and_unknown_fall_back_to_squat_training() -> None:
    assert isinstance(tracker_for(None, None), OnlineRepTracker)
    assert isinstance(tracker_for("CURL", "whatever"), OnlineRepTracker)


def test_every_lift_advertises_a_training_judge() -> None:
    """The UI enables the training toggle off this; it must not over-promise."""
    for lift in LIFTS:
        assert supports_training(lift) is True
    assert supports_training(None) is True  # default lift is squat


def test_training_capable_lifts_actually_return_a_free_rep_judge() -> None:
    """Whatever claims training support must not hand back a single-attempt judge.

    This is the assertion that would have caught the original bug, where bench
    and deadlift trained you with a one-attempt referee judge.
    """
    single_attempt = (CompetitionTracker, BenchTracker, DeadliftTracker)
    for lift in LIFTS:
        if supports_training(lift):
            tracker = tracker_for(lift, "training")
            assert isinstance(tracker, OnlineRepTracker | FreeRepTracker)
            assert not isinstance(tracker, single_attempt)
