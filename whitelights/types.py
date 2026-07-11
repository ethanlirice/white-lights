"""Shared, typed data model for the White Lights pipeline.

Every stage of the pipeline speaks in terms of the models defined here, so the
contracts between `pose` -> `smoothing` -> `fusion` -> `depth` -> `reps` are
explicit and serialisable (they double as the API's response schema).

Coordinate conventions
-----------------------
2D (`Keypoint2D`): image pixel coordinates, origin top-left, +y points *down*.
3D (`Keypoint3D`): world/room coordinates after multi-view fusion, +z points
*up*. Depth reasoning happens in 3D; see `whitelights.depth`.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Keypoint schema
# ---------------------------------------------------------------------------

# COCO-17 layout, which is exactly what YOLO11-pose emits (in this order).
COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Keypoints that actually matter for a squat-depth judgment, for convenience.
DEPTH_KEYPOINTS: tuple[str, ...] = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
)


class Keypoint2D(BaseModel):
    """A single 2D keypoint detected in one camera's image plane."""

    name: str
    x: float
    y: float
    confidence: float = Field(ge=0.0, le=1.0)


class Keypoint3D(BaseModel):
    """A single 3D keypoint in fused world coordinates (+z up)."""

    name: str
    x: float
    y: float
    z: float
    confidence: float = Field(ge=0.0, le=1.0)


class FrameKeypoints(BaseModel):
    """All keypoints for the tracked subject in one frame of one camera."""

    frame_idx: int
    time_s: float
    keypoints: dict[str, Keypoint2D] = Field(default_factory=dict)
    detected: bool = True
    subject_confidence: float = 0.0

    def get(self, name: str) -> Keypoint2D | None:
        return self.keypoints.get(name)


class FrameKeypoints3D(BaseModel):
    """Fused 3D keypoints for the tracked subject in one time-synced frame."""

    frame_idx: int
    time_s: float
    keypoints: dict[str, Keypoint3D] = Field(default_factory=dict)
    confidence: float = 0.0

    def get(self, name: str) -> Keypoint3D | None:
        return self.keypoints.get(name)


class PoseSequence(BaseModel):
    """A full 2D pose track for one camera view."""

    camera_id: str
    fps: float
    frames: list[FrameKeypoints] = Field(default_factory=list)
    source: str | None = None

    @property
    def duration_s(self) -> float:
        return len(self.frames) / self.fps if self.fps else 0.0


class Pose3DSequence(BaseModel):
    """A fused 3D pose track built from one or more `PoseSequence` views."""

    fps: float
    frames: list[FrameKeypoints3D] = Field(default_factory=list)
    camera_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Referee commands (the "command sandwich" around a legal attempt)
# ---------------------------------------------------------------------------


class Command(StrEnum):
    """Chief-referee commands that bound a legal squat attempt."""

    START = "START"  # "Squat!" — descent may begin
    RACK = "RACK"  # "Rack!" — attempt is over, return the bar


class RefereeCommand(BaseModel):
    """A single command with the video timestamp at which it was issued.

    Sourced manually or (future) via audio detection. Optional throughout the
    pipeline: command-timing faults are only evaluated when these are provided.
    """

    command: Command
    time_s: float


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


class Verdict(StrEnum):
    GOOD = "GOOD"
    NO_LIFT = "NO_LIFT"
    UNCERTAIN = "UNCERTAIN"  # insufficient signal to make a call (confidence-gated)


class Fault(StrEnum):
    """Rule violations, ordered roughly by CV tractability.

    A rep may carry several at once, which is why verdicts hold a *list* of
    faults rather than a single reason. Detectors are implemented incrementally
    (see the roadmap in DESIGN.md); undetected fault types simply never appear.
    """

    INSUFFICIENT_DEPTH = "INSUFFICIENT_DEPTH"  # hip crease never broke below top of knee
    DOWNWARD_MOVEMENT = "DOWNWARD_MOVEMENT"  # re-descent / double-bounce on the ascent
    EARLY_DESCENT = "EARLY_DESCENT"  # began descending/lowering before the start command
    EARLY_RACK = "EARLY_RACK"  # racked before the "Rack!" command
    INCOMPLETE_LOCKOUT = "INCOMPLETE_LOCKOUT"  # knees/elbows not locked at start or finish
    FOOT_MOVEMENT = "FOOT_MOVEMENT"  # stepped or shifted feet before "Rack!"
    BAR_SUPPORTED_ON_THIGHS = "BAR_SUPPORTED_ON_THIGHS"
    # Bench press
    EARLY_PRESS = "EARLY_PRESS"  # pressed off the chest before the "Press!" command
    BAR_NOT_TO_CHEST = "BAR_NOT_TO_CHEST"  # bar not lowered to the chest
    # Deadlift
    EARLY_DOWN = "EARLY_DOWN"  # lowered the bar before the "Down!" command
    HITCHING = "HITCHING"  # ramped/hitched the bar up the thighs (not yet detected)


class RepVerdict(BaseModel):
    """One verdict per detected rep attempt."""

    rep_index: int
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    faults: list[Fault] = Field(default_factory=list)

    # Primary depth evidence: signed distance of hip crease relative to the
    # top-of-knee plane at the bottom of the rep, positive == legal depth
    # (hip crease below top of knee). Units are the fused 3D world unit.
    depth_margin: float | None = None

    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    notes: str | None = None


class JudgeResult(BaseModel):
    """Top-level pipeline output and API response body."""

    source: str
    fps: float
    frame_count: int
    camera_ids: list[str] = Field(default_factory=list)
    reps: list[RepVerdict] = Field(default_factory=list)
    processing_ms: float = 0.0
