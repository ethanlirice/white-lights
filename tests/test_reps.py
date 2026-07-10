"""Contract tests for `reps.segment_reps`.

Depth results are supplied as ground truth (via the `make_depth` fixture) so
these test the state machine in isolation from the depth module. Covers depth
verdicts, downward-movement (v2.1), and EARLY_DESCENT command timing (v2.2).
"""

from __future__ import annotations

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


def test_double_bounce_flags_downward_movement(double_bounce_3d, make_depth) -> None:
    verdicts = segment_reps(double_bounce_3d, make_depth(double_bounce_3d))
    assert len(verdicts) == 1
    assert Fault.DOWNWARD_MOVEMENT in verdicts[0].faults
    assert verdicts[0].verdict == Verdict.NO_LIFT


def test_clean_squat_has_no_downward_movement(good_squat_3d, make_depth) -> None:
    # A monotonic ascent must not be mistaken for a bounce.
    verdicts = segment_reps(good_squat_3d, make_depth(good_squat_3d))
    assert Fault.DOWNWARD_MOVEMENT not in verdicts[0].faults
    assert verdicts[0].verdict == Verdict.GOOD


def test_descent_before_start_command_flags_early_descent(good_squat_3d, make_depth) -> None:
    # START issued late (after descent has begun) -> EARLY_DESCENT.
    commands = [
        RefereeCommand(command=Command.START, time_s=good_squat_3d.frames[-1].time_s),
        RefereeCommand(command=Command.RACK, time_s=good_squat_3d.frames[-1].time_s + 1),
    ]
    verdicts = segment_reps(good_squat_3d, make_depth(good_squat_3d), commands=commands)
    assert Fault.EARLY_DESCENT in verdicts[0].faults
    assert verdicts[0].verdict == Verdict.NO_LIFT


def test_start_before_descent_is_not_early(good_squat_3d, make_depth) -> None:
    # START issued at the very start, before any descent -> no fault.
    commands = [
        RefereeCommand(command=Command.START, time_s=0.0),
        RefereeCommand(command=Command.RACK, time_s=good_squat_3d.frames[-1].time_s + 1),
    ]
    verdicts = segment_reps(good_squat_3d, make_depth(good_squat_3d), commands=commands)
    assert Fault.EARLY_DESCENT not in verdicts[0].faults
    assert verdicts[0].verdict == Verdict.GOOD


def test_no_commands_means_no_command_faults(good_squat_3d, make_depth) -> None:
    verdicts = segment_reps(good_squat_3d, make_depth(good_squat_3d), commands=None)
    assert Fault.EARLY_DESCENT not in verdicts[0].faults


# --- postural faults (v2.3) --------------------------------------------------


def test_clean_full_squat_has_no_postural_faults(make_full_squat, make_depth) -> None:
    poses = make_full_squat()
    verdicts = segment_reps(poses, make_depth(poses))
    assert len(verdicts) == 1
    assert verdicts[0].verdict == Verdict.GOOD
    assert Fault.INCOMPLETE_LOCKOUT not in verdicts[0].faults
    assert Fault.FOOT_MOVEMENT not in verdicts[0].faults


def test_soft_knees_flag_incomplete_lockout(make_full_squat, make_depth) -> None:
    poses = make_full_squat(bend_offset=0.3)  # knees bent even at the top
    verdicts = segment_reps(poses, make_depth(poses))
    assert Fault.INCOMPLETE_LOCKOUT in verdicts[0].faults
    assert verdicts[0].verdict == Verdict.NO_LIFT


def test_foot_shift_flags_foot_movement(make_full_squat, make_depth) -> None:
    poses = make_full_squat(foot_shift=0.3)  # ankles drift during the rep
    verdicts = segment_reps(poses, make_depth(poses))
    assert Fault.FOOT_MOVEMENT in verdicts[0].faults
    assert verdicts[0].verdict == Verdict.NO_LIFT
