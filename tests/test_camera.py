"""Pinhole camera model, and the geometry facts the sensitivity analysis rests on.

`eval/geometry.py` concludes that camera yaw changes depth calls while pitch and
height barely do. That conclusion is only worth anything if the projection itself
is right, so these tests pin the model's behaviour directly — including the
property that *explains* the pitch result: rotating a camera about its own lens
cannot reorder two points vertically.
"""

from __future__ import annotations

import pytest
from conftest import make_squat_3d, v_series

from whitelights.camera import CameraPose, project_frame, project_sequence
from whitelights.types import FrameKeypoints3D, Keypoint3D


def _frame(points: dict[str, tuple[float, float, float]]) -> FrameKeypoints3D:
    return FrameKeypoints3D(
        frame_idx=0,
        time_s=0.0,
        keypoints={
            name: Keypoint3D(name=name, x=x, y=y, z=z, confidence=0.9)
            for name, (x, y, z) in points.items()
        },
        confidence=0.9,
    )


def test_point_at_camera_height_lands_on_the_horizon() -> None:
    """A point level with a level camera projects to the image centre row."""
    camera = CameraPose(distance=3.0, height=1.0, pitch_deg=0.0)
    u, v = camera.project_point((0.0, 0.0, 1.0))

    assert u == pytest.approx(camera.image_width / 2)
    assert v == pytest.approx(camera.image_height / 2)


def test_higher_in_the_world_is_higher_in_the_image() -> None:
    """Image y points down, so greater world z must give smaller v."""
    camera = CameraPose(distance=3.0, height=1.0)
    _, v_low = camera.project_point((0.0, 0.0, 0.5))
    _, v_high = camera.project_point((0.0, 0.0, 1.5))

    assert v_high < v_low


def test_closer_objects_project_larger() -> None:
    """Perspective: the same span subtends more pixels when it is nearer."""
    camera = CameraPose(distance=3.0, height=1.0)
    near = camera.project_point((0.0, -1.0, 1.5))[1] - camera.project_point((0.0, -1.0, 0.5))[1]
    far = camera.project_point((0.0, 1.0, 1.5))[1] - camera.project_point((0.0, 1.0, 0.5))[1]

    assert abs(near) > abs(far)


def test_points_behind_the_lens_are_dropped() -> None:
    camera = CameraPose(distance=3.0, height=1.0)
    assert camera.project_point((0.0, -10.0, 1.0)) is None

    frame = _frame({"visible": (0.0, 0.0, 1.0), "behind": (0.0, -10.0, 1.0)})
    projected = project_frame(frame, camera)
    assert "visible" in projected.keypoints
    assert "behind" not in projected.keypoints


@pytest.mark.parametrize("pitch", [-30.0, -10.0, 0.0, 10.0, 30.0])
def test_pitch_cannot_reorder_points_vertically(pitch: float) -> None:
    """The reason pitch does not move depth calls.

    Rotating about the lens is a homography: it moves every row, but monotonically.
    Two points keep their vertical order however the camera is tilted — so a
    hip-vs-knee row comparison is immune to pitch.
    """
    camera = CameraPose(distance=3.0, height=1.0, pitch_deg=pitch)
    heights = [0.2, 0.5, 0.9, 1.4, 1.8]
    rows = [camera.project_point((0.0, 0.0, z))[1] for z in heights]

    assert rows == sorted(rows, reverse=True), f"pitch {pitch}° reordered points vertically: {rows}"


def test_yaw_moves_points_at_different_depths_differently() -> None:
    """The reason yaw *does* move depth calls.

    Off-axis, two points separated in depth stop sharing a scale factor, which is
    the parallax the single-view lift cannot recover.
    """
    square = CameraPose(distance=3.0, height=1.0, yaw_deg=0.0)
    off_axis = CameraPose(distance=3.0, height=1.0, yaw_deg=40.0)

    # Two points at the same height, separated along the lifter's forward axis —
    # below the lens, since anything level with it lands on the horizon row where
    # no camera angle can separate the two.
    a, b = (0.0, 0.0, 0.5), (0.35, 0.0, 0.5)
    square_gap = square.project_point(b)[1] - square.project_point(a)[1]
    off_axis_gap = off_axis.project_point(b)[1] - off_axis.project_point(a)[1]

    assert square_gap == pytest.approx(0.0, abs=1e-6)
    assert abs(off_axis_gap) > 1.0  # pixels


def test_project_sequence_preserves_timebase() -> None:
    sequence = make_squat_3d(v_series(1.0, 0.45))
    view = project_sequence(sequence, CameraPose())

    assert view.fps == sequence.fps
    assert len(view.frames) == len(sequence.frames)
    assert [f.frame_idx for f in view.frames] == [f.frame_idx for f in sequence.frames]
    assert [f.time_s for f in view.frames] == [f.time_s for f in sequence.frames]
