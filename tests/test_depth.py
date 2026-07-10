"""Contract tests for `depth.judge_depth_frame`."""

from __future__ import annotations

from whitelights.depth import DepthConfig, judge_depth_frame, judge_depth_sequence


def test_good_squat_bottom_is_below_parallel(good_squat_3d, bottom_of) -> None:
    result = judge_depth_frame(bottom_of(good_squat_3d))
    assert result.gated is False
    assert result.is_below_parallel is True
    assert result.depth_margin is not None and result.depth_margin > 0


def test_high_squat_bottom_is_not_below_parallel(high_squat_3d, bottom_of) -> None:
    result = judge_depth_frame(bottom_of(high_squat_3d))
    assert result.gated is False
    assert result.is_below_parallel is False
    assert result.depth_margin is not None and result.depth_margin < 0


def test_low_confidence_frame_is_gated(low_confidence_frame) -> None:
    result = judge_depth_frame(low_confidence_frame)
    assert result.gated is True
    assert result.is_below_parallel is None  # never guess on weak signal


def test_standing_frame_is_not_below_parallel(good_squat_3d) -> None:
    # First frame of the trace is the top of the squat (hips high).
    result = judge_depth_frame(good_squat_3d.frames[0])
    assert result.gated is False
    assert result.is_below_parallel is False
    assert result.depth_margin < 0


def test_missing_knees_are_gated() -> None:
    from whitelights.types import FrameKeypoints3D, Keypoint3D

    frame = FrameKeypoints3D(
        frame_idx=0,
        time_s=0.0,
        keypoints={"left_hip": Keypoint3D(name="left_hip", x=0, y=0, z=0.45, confidence=0.9)},
        confidence=0.9,
    )
    result = judge_depth_frame(frame)
    assert result.gated is True
    assert result.is_below_parallel is None


def test_hip_crease_offset_shifts_the_call(high_squat_3d, bottom_of) -> None:
    # Offset lowers the crease relative to the hip joint; a large one drops an
    # otherwise-high squat below the knee line.
    frame = bottom_of(high_squat_3d)  # hips 0.60, knees 0.50 -> margin -0.10
    default = judge_depth_frame(frame, DepthConfig(hip_crease_offset=0.0))
    lowered = judge_depth_frame(frame, DepthConfig(hip_crease_offset=0.15))
    assert default.is_below_parallel is False
    assert lowered.is_below_parallel is True


def test_judge_depth_sequence_covers_every_frame(good_squat_3d) -> None:
    results = judge_depth_sequence(good_squat_3d)
    assert len(results) == len(good_squat_3d.frames)
    # A good squat must have at least one confidently-below-parallel frame.
    assert any(r.is_below_parallel for r in results)
