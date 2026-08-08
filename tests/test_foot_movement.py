"""FOOT_MOVEMENT in the live path, and its agreement with the batch path.

Until now this fault was only ever raised by `reps.py` (the batch pipeline), so
the live judge — the one the app actually runs — never called it despite the
README listing it. These tests cover the live trackers and pin the two paths
together: `FootDriftMonitor` (online) and `foot_displacement_ratio` (batch) share
an implementation and must report the same ratio for the same frames.
"""

from __future__ import annotations

import pytest
from conftest import ground_truth_depth, make_deadlift_3d, make_full_squat_3d, v_series

from whitelights.deadlift import DeadliftTracker
from whitelights.depth import DepthFrameResult
from whitelights.live import CompetitionTracker, OnlineRepTracker
from whitelights.posture import FootDriftMonitor, PostureConfig, foot_displacement_ratio
from whitelights.types import Fault, Pose3DSequence, Verdict

STILL = 25
MOVED = 0.3  # ankle drift well past the 0.15-of-thigh threshold
PLANTED = 0.0


def _ramp(a: float, b: float, n: int = 15) -> list[float]:
    step = (b - a) / (n - 1)
    return [a + step * i for i in range(n)]


def _attempt(bottom: float = 0.45) -> list[float]:
    return [1.0] * STILL + _ramp(1.0, bottom) + _ramp(bottom, 1.0) + [1.0] * STILL


def _drift_ankles(seq: Pose3DSequence, shift: float) -> Pose3DSequence:
    """Drift both ankles linearly in x across the sequence (post-hoc, any trace)."""
    n = len(seq.frames)
    for i, frame in enumerate(seq.frames):
        offset = shift * (i / (n - 1)) if n > 1 else 0.0
        for side in ("left", "right"):
            ankle = frame.get(f"{side}_ankle")
            if ankle is not None:
                ankle.x += offset
    return seq


def _drift_body(seq: Pose3DSequence, shift: float) -> Pose3DSequence:
    """Translate the whole lifter in x — a stance shuffle, not a distortion.

    Moving the ankles alone changes the hip-knee-ankle angle, which for the
    deadlift blocks lockout entirely and so never reaches a verdict. Translating
    every keypoint preserves all joint angles while still drifting the ankles
    away from their own centroid, which is what the detector measures.
    """
    n = len(seq.frames)
    for i, frame in enumerate(seq.frames):
        offset = shift * (i / (n - 1)) if n > 1 else 0.0
        for keypoint in frame.keypoints.values():
            keypoint.x += offset
    return seq


def _gated(seq):
    return [
        DepthFrameResult(frame_idx=f.frame_idx, time_s=f.time_s, gated=True) for f in seq.frames
    ]


def _faults(statuses):
    verdict = statuses[-1].last_verdict
    assert verdict is not None, "expected the attempt to be judged"
    return verdict


# ---------------------------------------------------------------------------
# The two execution paths must agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shift", [0.0, 0.05, 0.1, 0.3, 0.6])
def test_online_monitor_matches_batch_detector(shift: float) -> None:
    """Same frames in, same ratio out — the paths share one implementation."""
    seq = make_full_squat_3d(v_series(1.0, 0.45), foot_shift=shift)
    config = PostureConfig()

    monitor = FootDriftMonitor(config)
    for frame in seq.frames:
        monitor.observe(frame)

    assert monitor.ratio() == pytest.approx(foot_displacement_ratio(seq.frames, config))


def test_monitor_reports_unknown_without_ankles() -> None:
    """No ankle track means no call — never a false positive on absent signal."""
    seq = make_full_squat_3d(v_series(1.0, 0.45))
    for frame in seq.frames:
        for side in ("left", "right"):
            frame.keypoints.pop(f"{side}_ankle", None)

    monitor = FootDriftMonitor(PostureConfig())
    for frame in seq.frames:
        monitor.observe(frame)

    assert monitor.ratio() is None
    assert monitor.moved() is False


def test_reset_discards_the_previous_rep() -> None:
    """Drift in one rep must not follow the lifter into the next."""
    monitor = FootDriftMonitor(PostureConfig())
    for frame in make_full_squat_3d(v_series(1.0, 0.45), foot_shift=MOVED).frames:
        monitor.observe(frame)
    assert monitor.moved() is True

    monitor.reset()
    for frame in make_full_squat_3d(v_series(1.0, 0.45), foot_shift=PLANTED).frames:
        monitor.observe(frame)
    assert monitor.moved() is False


# ---------------------------------------------------------------------------
# Live trackers
# ---------------------------------------------------------------------------


def test_squat_competition_flags_foot_movement() -> None:
    poses = _drift_ankles(make_full_squat_3d(_attempt()), MOVED)
    depths = ground_truth_depth(poses)
    tracker = CompetitionTracker()
    statuses = [tracker.update(f, d) for f, d in zip(poses.frames, depths, strict=True)]

    verdict = _faults(statuses)
    assert Fault.FOOT_MOVEMENT in verdict.faults
    assert verdict.verdict == Verdict.NO_LIFT


def test_squat_competition_planted_feet_pass() -> None:
    poses = make_full_squat_3d(_attempt())
    depths = ground_truth_depth(poses)
    tracker = CompetitionTracker()
    statuses = [tracker.update(f, d) for f, d in zip(poses.frames, depths, strict=True)]

    verdict = _faults(statuses)
    assert Fault.FOOT_MOVEMENT not in verdict.faults
    assert verdict.verdict == Verdict.GOOD


def test_squat_training_flags_foot_movement() -> None:
    poses = _drift_ankles(make_full_squat_3d(v_series(1.0, 0.45)), MOVED)
    depths = ground_truth_depth(poses)
    tracker = OnlineRepTracker()
    statuses = [tracker.update(f, d) for f, d in zip(poses.frames, depths, strict=True)]

    verdict = _faults(statuses)
    assert Fault.FOOT_MOVEMENT in verdict.faults


def test_deadlift_flags_foot_movement() -> None:
    series = [0.15] * STILL + _ramp(0.15, 1.0, 20) + [1.0] * STILL
    poses = _drift_body(make_deadlift_3d(series), MOVED)
    tracker = DeadliftTracker()
    statuses = [tracker.update(f, d) for f, d in zip(poses.frames, _gated(poses), strict=True)]

    verdict = _faults(statuses)
    assert Fault.FOOT_MOVEMENT in verdict.faults
    assert verdict.verdict == Verdict.NO_LIFT
