"""Synthetic keypoint fixtures to develop the CV stubs against.

These build ground-truth 3D squat traces (and one noisy 2D trace) so the depth
and reps contracts can be exercised without any real video or model. The traces
are deliberately simple and known, so once you implement a stub you can assert
exact behaviour.

Vertical convention matches `types` 3D: +z is up. A hip below the knee (small
``z``) is legal depth.
"""

from __future__ import annotations

import numpy as np
import pytest

from whitelights.depth import DepthFrameResult
from whitelights.types import (
    FrameKeypoints,
    FrameKeypoints3D,
    Keypoint2D,
    Keypoint3D,
    Pose3DSequence,
    PoseSequence,
)

KNEE_Z = 0.50  # constant top-of-knee height for all synthetic traces
FPS = 30.0


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def v_series(top: float, bottom: float, n: int = 60) -> list[float]:
    """A simple descend-then-ascend hip-height trajectory."""
    half = n // 2
    down = np.linspace(top, bottom, half)
    up = np.linspace(bottom, top, n - half)
    return [float(z) for z in np.concatenate([down, up])]


def make_squat_3d(
    hip_z_series: list[float],
    *,
    fps: float = FPS,
    knee_z: float = KNEE_Z,
    confidence: float = 0.95,
) -> Pose3DSequence:
    """Build a `Pose3DSequence` where the hips follow ``hip_z_series``."""
    frames: list[FrameKeypoints3D] = []
    for i, hip_z in enumerate(hip_z_series):
        c = confidence
        kps = {
            "left_hip": Keypoint3D(name="left_hip", x=-0.1, y=0.0, z=hip_z, confidence=c),
            "right_hip": Keypoint3D(name="right_hip", x=0.1, y=0.0, z=hip_z, confidence=c),
            "left_knee": Keypoint3D(name="left_knee", x=-0.1, y=0.1, z=knee_z, confidence=c),
            "right_knee": Keypoint3D(name="right_knee", x=0.1, y=0.1, z=knee_z, confidence=c),
        }
        frames.append(
            FrameKeypoints3D(frame_idx=i, time_s=i / fps, keypoints=kps, confidence=confidence)
        )
    return Pose3DSequence(fps=fps, frames=frames, camera_ids=["cam0"])


def ground_truth_depth(
    sequence: Pose3DSequence, *, knee_z: float = KNEE_Z, min_confidence: float = 0.4
) -> list[DepthFrameResult]:
    """Derive *ground-truth* depth results from a known 3D trace.

    Lets the reps state-machine tests run in isolation from the depth stub.
    """
    results: list[DepthFrameResult] = []
    for f in sequence.frames:
        hip_z = min(f.get("left_hip").z, f.get("right_hip").z)  # lower of the two hips
        margin = knee_z - hip_z  # positive == below parallel
        gated = f.confidence < min_confidence
        results.append(
            DepthFrameResult(
                frame_idx=f.frame_idx,
                time_s=f.time_s,
                is_below_parallel=None if gated else margin > 0,
                depth_margin=None if gated else margin,
                confidence=f.confidence,
                gated=gated,
            )
        )
    return results


def bottom_frame(sequence: Pose3DSequence) -> FrameKeypoints3D:
    """The frame at the bottom of the squat (lowest hip)."""
    return min(sequence.frames, key=lambda f: min(f.get("left_hip").z, f.get("right_hip").z))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bottom_of():
    """Return the helper that finds the bottom (lowest-hip) frame of a trace."""
    return bottom_frame


@pytest.fixture
def make_depth():
    """Return the ground-truth depth-results builder (isolates reps from depth)."""
    return ground_truth_depth


@pytest.fixture
def good_squat_3d() -> Pose3DSequence:
    """Hip clearly breaks below the knee (bottom hip_z 0.45 < knee 0.50)."""
    return make_squat_3d(v_series(1.0, 0.45))


@pytest.fixture
def high_squat_3d() -> Pose3DSequence:
    """Hip never reaches depth (bottom hip_z 0.60 > knee 0.50)."""
    return make_squat_3d(v_series(1.0, 0.60))


@pytest.fixture
def double_bounce_3d() -> Pose3DSequence:
    """Reaches depth, ascends, then re-descends (downward-movement fault)."""
    series = (
        v_series(1.0, 0.45, n=30)[:15]  # descend to depth
        + [float(z) for z in np.linspace(0.45, 0.70, 10)]  # rise
        + [float(z) for z in np.linspace(0.70, 0.55, 5)]  # re-descend (bounce)
        + [float(z) for z in np.linspace(0.55, 1.0, 15)]  # finish
    )
    return make_squat_3d(series)


@pytest.fixture
def low_confidence_frame() -> FrameKeypoints3D:
    """A bottom-of-squat frame whose keypoints are too uncertain to judge."""
    seq = make_squat_3d([0.45], confidence=0.1)
    return seq.frames[0]


@pytest.fixture
def noisy_pose_2d() -> PoseSequence:
    """A 2D single-camera track with jitter and a short gap (for smoothing)."""
    rng = np.random.default_rng(0)
    frames: list[FrameKeypoints] = []
    for i in range(40):
        # a gap: frames 18-21 have no detection
        if 18 <= i <= 21:
            frames.append(FrameKeypoints(frame_idx=i, time_s=i / FPS, keypoints={}, detected=False))
            continue
        base_y = 300 + 50 * np.sin(i / 40 * np.pi)
        jitter = rng.normal(0, 3)
        kps = {
            "left_hip": Keypoint2D(
                name="left_hip", x=100 + jitter, y=base_y + jitter, confidence=0.9
            ),
        }
        frames.append(
            FrameKeypoints(
                frame_idx=i, time_s=i / FPS, keypoints=kps, detected=True, subject_confidence=0.9
            )
        )
    return PoseSequence(camera_id="cam0", fps=FPS, frames=frames)
