"""Tests for the online (causal) rep tracker used by the live webcam judge.

Feeds the synthetic 3D squat traces + ground-truth depth one frame at a time,
mirroring how a camera stream arrives, and checks the tracker reaches the same
verdicts as the batch segmenter.
"""

from __future__ import annotations

from whitelights.live import LiveState, OnlineRepTracker, lift_frame_to_3d
from whitelights.types import Fault, FrameKeypoints, Keypoint2D, PoseSequence, Verdict


def _drive(tracker: OnlineRepTracker, poses, depth_results):
    statuses = []
    for frame, depth in zip(poses.frames, depth_results, strict=True):
        statuses.append(tracker.update(frame, depth))
    return statuses


def test_good_squat_completes_one_good_rep(good_squat_3d, make_depth) -> None:
    tracker = OnlineRepTracker()
    statuses = _drive(tracker, good_squat_3d, make_depth(good_squat_3d))
    assert statuses[-1].rep_count == 1
    verdict = statuses[-1].last_verdict
    assert verdict is not None
    assert verdict.verdict == Verdict.GOOD
    assert verdict.faults == []
    # Exactly one frame reports the rep as just-completed.
    assert sum(1 for s in statuses if s.rep_completed) == 1


def test_high_squat_completes_no_lift(high_squat_3d, make_depth) -> None:
    tracker = OnlineRepTracker()
    statuses = _drive(tracker, high_squat_3d, make_depth(high_squat_3d))
    assert statuses[-1].rep_count == 1
    verdict = statuses[-1].last_verdict
    assert verdict.verdict == Verdict.NO_LIFT
    assert Fault.INSUFFICIENT_DEPTH in verdict.faults


def test_double_bounce_flags_downward_movement(double_bounce_3d, make_depth) -> None:
    tracker = OnlineRepTracker()
    statuses = _drive(tracker, double_bounce_3d, make_depth(double_bounce_3d))
    assert statuses[-1].rep_count == 1
    verdict = statuses[-1].last_verdict
    assert Fault.DOWNWARD_MOVEMENT in verdict.faults
    assert verdict.verdict == Verdict.NO_LIFT


def test_below_parallel_light_tracks_depth(good_squat_3d, make_depth) -> None:
    tracker = OnlineRepTracker()
    statuses = _drive(tracker, good_squat_3d, make_depth(good_squat_3d))
    # The light goes green (below_parallel True) somewhere near the bottom...
    assert any(s.below_parallel is True for s in statuses)
    # ...and is red (False) while standing at the top.
    assert statuses[0].below_parallel is False


def test_standing_still_produces_no_reps(good_squat_3d, make_depth) -> None:
    tracker = OnlineRepTracker()
    # Feed only the first (standing) frame repeatedly: no descent, no rep.
    frame = good_squat_3d.frames[0]
    depth = make_depth(good_squat_3d)[0]
    for _ in range(30):
        status = tracker.update(frame, depth)
    assert status.rep_count == 0
    assert status.state == LiveState.STANDING


def test_lift_frame_to_3d_matches_convention() -> None:
    frame2d = FrameKeypoints(
        frame_idx=3,
        time_s=0.1,
        keypoints={"left_hip": Keypoint2D(name="left_hip", x=100.0, y=300.0, confidence=0.9)},
        detected=True,
        subject_confidence=0.9,
    )
    # Standalone lift (a 1-frame sequence path is also exercised).
    seq = PoseSequence(camera_id="cam0", fps=30.0, frames=[frame2d])
    assert seq.frames[0].get("left_hip").y == 300.0

    frame3d = lift_frame_to_3d(frame2d, fps=30.0)
    kp = frame3d.get("left_hip")
    assert kp.x == 100.0
    assert kp.z == -300.0  # image y -> world z, sign-flipped
    assert frame3d.frame_idx == 3
