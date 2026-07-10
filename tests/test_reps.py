"""Contract tests for `reps.segment_reps`.

Depth results are supplied as ground truth (via the `make_depth` fixture) so
these test the state machine in isolation from the depth module.

The DOWNWARD_MOVEMENT (v2.1) and command-timing (v2.2) tests are `strict` xfail:
they encode the target behaviour and will flip to failures — the cue to drop the
marker — once those features land.
"""

from __future__ import annotations

import pytest

from whitelights.depth import DepthFrameResult
from whitelights.reps import segment_reps
from whitelights.types import (
    Command,
    Fault,
    FrameKeypoints3D,
    Keypoint3D,
    Pose3DSequence,
    RefereeCommand,
    Verdict,
)


def test_good_squat_yields_one_good_rep(good_squat_3d, make_depth) -> None:
    verdicts = segment_reps(good_squat_3d, make_depth(good_squat_3d))
    assert len(verdicts) == 1
    assert verdicts[0].verdict == Verdict.GOOD
    assert verdicts[0].faults == []
    assert verdicts[0].depth_margin > 0


def test_high_squat_is_no_lift_for_depth(high_squat_3d, make_depth) -> None:
    verdicts = segment_reps(high_squat_3d, make_depth(high_squat_3d))
    assert len(verdicts) == 1
    assert verdicts[0].verdict == Verdict.NO_LIFT
    assert Fault.INSUFFICIENT_DEPTH in verdicts[0].faults


def test_no_movement_yields_no_reps() -> None:
    # Lifter stands still — no descent, so no rep should be segmented.
    frames = [
        FrameKeypoints3D(
            frame_idx=i,
            time_s=i / 30.0,
            keypoints={
                "left_hip": Keypoint3D(name="left_hip", x=0, y=0, z=1.0, confidence=0.9),
                "right_hip": Keypoint3D(name="right_hip", x=0, y=0, z=1.0, confidence=0.9),
            },
            confidence=0.9,
        )
        for i in range(20)
    ]
    poses = Pose3DSequence(fps=30.0, frames=frames, camera_ids=["cam0"])
    assert segment_reps(poses, []) == []


def test_all_gated_depth_is_uncertain(good_squat_3d) -> None:
    # Motion is clearly a rep, but every depth reading is gated -> UNCERTAIN.
    gated = [
        DepthFrameResult(frame_idx=f.frame_idx, time_s=f.time_s, gated=True)
        for f in good_squat_3d.frames
    ]
    verdicts = segment_reps(good_squat_3d, gated)
    assert len(verdicts) == 1
    assert verdicts[0].verdict == Verdict.UNCERTAIN
    assert verdicts[0].depth_margin is None


@pytest.mark.xfail(strict=True, reason="v2.1: downward-movement detection not implemented")
def test_double_bounce_flags_downward_movement(double_bounce_3d, make_depth) -> None:
    verdicts = segment_reps(double_bounce_3d, make_depth(double_bounce_3d))
    assert len(verdicts) == 1
    assert Fault.DOWNWARD_MOVEMENT in verdicts[0].faults
    assert verdicts[0].verdict == Verdict.NO_LIFT


@pytest.mark.xfail(strict=True, reason="v2.2: command-timing faults not implemented")
def test_descent_before_start_command_flags_early_descent(good_squat_3d, make_depth) -> None:
    # START issued late (after descent has begun) -> EARLY_DESCENT.
    commands = [
        RefereeCommand(command=Command.START, time_s=good_squat_3d.frames[-1].time_s),
        RefereeCommand(command=Command.RACK, time_s=good_squat_3d.frames[-1].time_s + 1),
    ]
    verdicts = segment_reps(good_squat_3d, make_depth(good_squat_3d), commands=commands)
    assert Fault.EARLY_DESCENT in verdicts[0].faults
