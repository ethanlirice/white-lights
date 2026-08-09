"""Pinhole camera model: project world 3D onto an image plane.

This exists to answer a question the pipeline cannot currently answer about
itself. `fusion._lift_single_view` maps image ``y`` straight to world ``z``, so
`depth.judge_depth_frame` compares **image rows**, not heights. That is scale
invariant — the module says so, and it is true — but it is *not* invariant to
where the camera is. A hip and a knee at different distances from the lens
project with different scaling, and the squat is exactly the movement that
separates them: the knees travel forward as the lifter descends.

Projecting a known 3D trace through a virtual camera and running the result back
through the real pipeline turns "how much does camera placement matter?" into a
measurement. It needs no footage, because ground truth comes from the 3D trace
the projection started from.

Coordinates
-----------
World: ``x`` lateral (+right), ``y`` depth (+away from camera), ``z`` up — the
convention `types` already documents. Image: ``x`` right, ``y`` **down**, origin
top-left, matching `Keypoint2D`.

The camera sits at horizontal ``distance`` from the world origin, at ``height``,
and may be pitched and yawed. ``pitch_deg > 0`` tilts it **down** toward the
platform, which is how a camera on a tripod above hip height is actually aimed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .types import FrameKeypoints, FrameKeypoints3D, Keypoint2D, Pose3DSequence, PoseSequence


@dataclass(frozen=True)
class CameraPose:
    """Where the camera is and what it can see.

    ``focal_px`` together with ``image_width`` sets the field of view; the
    defaults describe a normal phone camera roughly 3.5 m from the platform at
    hip height, which is where a lifter would actually put one.
    """

    distance: float = 3.5  # horizontal distance from the subject, world units
    height: float = 1.0  # camera height above the floor
    pitch_deg: float = 0.0  # + tilts the lens down toward the platform
    yaw_deg: float = 0.0  # + swings the camera around the subject (off-axis)
    focal_px: float = 900.0
    image_width: int = 1280
    image_height: int = 720

    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(R, C)``: world->camera rotation, and camera position."""
        # Level camera: image x = world +x, image y = world -z (down), view = +y.
        theta = -math.radians(self.pitch_deg)  # negate so +pitch looks downward
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, math.sin(theta), -math.cos(theta)])
        z_axis = np.array([0.0, math.cos(theta), math.sin(theta)])

        psi = math.radians(self.yaw_deg)
        yaw = np.array(
            [
                [math.cos(psi), -math.sin(psi), 0.0],
                [math.sin(psi), math.cos(psi), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotation = np.stack([yaw @ x_axis, yaw @ y_axis, yaw @ z_axis])
        centre = yaw @ np.array([0.0, -self.distance, self.height])
        return rotation, centre

    def project_point(self, point: tuple[float, float, float]) -> tuple[float, float] | None:
        """Project one world point to pixels, or None if it is behind the lens."""
        rotation, centre = self.basis()
        cam = rotation @ (np.asarray(point, dtype=float) - centre)
        if cam[2] <= 1e-6:  # at or behind the image plane
            return None
        u = self.focal_px * cam[0] / cam[2] + self.image_width / 2.0
        v = self.focal_px * cam[1] / cam[2] + self.image_height / 2.0
        return float(u), float(v)


def project_frame(frame: FrameKeypoints3D, camera: CameraPose) -> FrameKeypoints:
    """Project one 3D frame to a 2D frame, as a real camera would see it."""
    keypoints: dict[str, Keypoint2D] = {}
    for name, kp in frame.keypoints.items():
        projected = camera.project_point((kp.x, kp.y, kp.z))
        if projected is None:
            continue
        u, v = projected
        keypoints[name] = Keypoint2D(name=name, x=u, y=v, confidence=kp.confidence)
    return FrameKeypoints(
        frame_idx=frame.frame_idx,
        time_s=frame.time_s,
        keypoints=keypoints,
        detected=bool(keypoints),
        subject_confidence=frame.confidence,
    )


def project_sequence(
    sequence: Pose3DSequence, camera: CameraPose, *, camera_id: str = "virtual"
) -> PoseSequence:
    """Project a whole 3D trace through ``camera`` into a single-camera 2D view.

    The result is the same shape `pose.py` produces from real video, so it can be
    fed straight back through fusion -> depth -> the trackers.
    """
    return PoseSequence(
        camera_id=camera_id,
        fps=sequence.fps,
        frames=[project_frame(f, camera) for f in sequence.frames],
        source=f"projected:{camera_id}",
    )
