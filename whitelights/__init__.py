"""White Lights — real-time computer-vision squat-depth judge for powerlifting."""

from __future__ import annotations

from .pipeline import PipelineConfig, judge_video
from .pose import PoseEstimator
from .types import (
    Command,
    Fault,
    JudgeResult,
    PoseSequence,
    RefereeCommand,
    RepVerdict,
    Verdict,
)

__version__ = "2.0.0.dev0"

__all__ = [
    "Command",
    "Fault",
    "JudgeResult",
    "PipelineConfig",
    "PoseEstimator",
    "PoseSequence",
    "RefereeCommand",
    "RepVerdict",
    "Verdict",
    "judge_video",
]
