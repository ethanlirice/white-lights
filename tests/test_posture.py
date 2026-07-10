"""Unit tests for the postural detectors in `whitelights.posture`."""

from __future__ import annotations

from whitelights.posture import (
    PostureConfig,
    foot_displacement_ratio,
    is_locked_out,
    knee_angle_deg,
)
from whitelights.types import FrameKeypoints3D, Keypoint3D


def _frame(idx: int = 0, **kps: tuple[float, float, float]) -> FrameKeypoints3D:
    """Build a 3D frame from name -> (x, y, z) with confidence 0.9."""
    keypoints = {
        name: Keypoint3D(name=name, x=x, y=y, z=z, confidence=0.9)
        for name, (x, y, z) in kps.items()
    }
    return FrameKeypoints3D(frame_idx=idx, time_s=idx / 30.0, keypoints=keypoints, confidence=0.9)


def _leg(hip: tuple, knee: tuple, ankle: tuple, side: str = "left") -> dict:
    return {f"{side}_hip": hip, f"{side}_knee": knee, f"{side}_ankle": ankle}


def test_knee_angle_straight_leg_is_180() -> None:
    frame = _frame(**_leg((0, 0, 1.0), (0, 0, 0.5), (0, 0, 0.0)))
    assert knee_angle_deg(frame, "left", 0.4) == 180.0


def test_knee_angle_bent_leg_is_acute() -> None:
    frame = _frame(**_leg((0.3, 0, 0.6), (0, 0, 0.5), (0, 0, 0.0)))
    angle = knee_angle_deg(frame, "left", 0.4)
    assert angle is not None
    assert 90 < angle < 160  # clearly bent


def test_knee_angle_none_when_ankle_missing() -> None:
    frame = _frame(
        left_hip=(0, 0, 1.0),
        left_knee=(0, 0, 0.5),
    )
    assert knee_angle_deg(frame, "left", 0.4) is None


def test_knee_angle_none_when_low_confidence() -> None:
    frame = FrameKeypoints3D(
        frame_idx=0,
        time_s=0.0,
        keypoints={
            "left_hip": Keypoint3D(name="left_hip", x=0, y=0, z=1.0, confidence=0.1),
            "left_knee": Keypoint3D(name="left_knee", x=0, y=0, z=0.5, confidence=0.9),
            "left_ankle": Keypoint3D(name="left_ankle", x=0, y=0, z=0.0, confidence=0.9),
        },
        confidence=0.5,
    )
    assert knee_angle_deg(frame, "left", 0.4) is None


def test_is_locked_out_true_when_both_legs_straight() -> None:
    frame = _frame(
        **_leg((-0.1, 0, 1.0), (-0.1, 0, 0.5), (-0.1, 0, 0.0), "left"),
        **_leg((0.1, 0, 1.0), (0.1, 0, 0.5), (0.1, 0, 0.0), "right"),
    )
    assert is_locked_out(frame, PostureConfig()) is True


def test_is_locked_out_false_when_one_leg_bent() -> None:
    frame = _frame(
        **_leg((-0.1, 0, 1.0), (-0.1, 0, 0.5), (-0.1, 0, 0.0), "left"),  # straight
        **_leg((0.4, 0, 0.6), (0.1, 0, 0.5), (0.1, 0, 0.0), "right"),  # bent
    )
    assert is_locked_out(frame, PostureConfig()) is False


def test_is_locked_out_none_when_unmeasurable() -> None:
    frame = _frame(left_hip=(0, 0, 1.0))  # no knees/ankles
    assert is_locked_out(frame, PostureConfig()) is None


def test_foot_displacement_none_without_ankles() -> None:
    frames = [_frame(i, left_hip=(0, 0, 1.0), left_knee=(0, 0, 0.5)) for i in range(5)]
    assert foot_displacement_ratio(frames, PostureConfig()) is None


def test_foot_displacement_small_when_still() -> None:
    frames = [_frame(i, **_leg((0, 0, 1.0), (0, 0, 0.5), (0, 0, 0.0))) for i in range(5)]
    ratio = foot_displacement_ratio(frames, PostureConfig())
    assert ratio == 0.0


def test_foot_displacement_detects_shift() -> None:
    # Ankle drifts 0 -> 0.2 in x; thigh reference is 0.5, so ratio ~0.2 > default 0.15.
    frames = [_frame(i, **_leg((0, 0, 1.0), (0, 0, 0.5), (0.05 * i, 0, 0.0))) for i in range(5)]
    ratio = foot_displacement_ratio(frames, PostureConfig())
    assert ratio is not None
    assert ratio > 0.15
