"""End-to-end orchestration: video(s) -> per-rep verdicts.

Wires the stages together in order::

    pose (WORKING) -> smoothing -> fusion(3D) -> depth -> reps

Only `pose` is implemented today, so a real call runs pose and then raises
`NotImplementedError` at the first stub (smoothing). Callers (the API) map that
to a clear "core logic not implemented" response. Once each stub lands the
pipeline lights up stage by stage with no signature changes.
"""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel

from .depth import DepthConfig, judge_depth_sequence
from .fusion import reconstruct_3d
from .pose import DEFAULT_MODEL, PoseEstimator
from .reps import RepConfig, segment_reps
from .smoothing import SmoothingConfig, smooth_sequence
from .types import JudgeResult, PoseSequence, RefereeCommand


class PipelineConfig(BaseModel):
    model_path: str = DEFAULT_MODEL
    device: str | None = None
    conf: float = 0.25
    smoothing: SmoothingConfig = SmoothingConfig()
    depth: DepthConfig = DepthConfig()
    reps: RepConfig = RepConfig()


def judge_video(
    video_paths: str | Path | list[str | Path],
    commands: list[RefereeCommand] | None = None,
    config: PipelineConfig | None = None,
    estimator: PoseEstimator | None = None,
) -> JudgeResult:
    """Run the full pipeline over one or more synchronised camera views.

    Args:
        video_paths: a single video path, or a list of paths (one per camera).
        commands: optional referee commands.
        config: pipeline tunables; defaults applied when ``None``.
        estimator: inject a pre-built/mock `PoseEstimator` (used by tests);
            constructed from ``config`` when ``None``.

    Returns:
        A `JudgeResult` with one verdict per rep.

    Raises:
        NotImplementedError: from the first un-implemented core stage (today,
            smoothing). Expected until the CV logic is filled in.
    """
    config = config or PipelineConfig()
    paths = _as_path_list(video_paths)
    if not paths:
        raise ValueError("judge_video requires at least one video path")

    estimator = estimator or PoseEstimator(
        model_path=config.model_path, device=config.device, conf=config.conf
    )

    started = time.perf_counter()

    # Stage 1 — pose per camera (WORKING).
    views: list[PoseSequence] = [
        estimator.run_video(p, camera_id=f"cam{i}") for i, p in enumerate(paths)
    ]

    # Stage 2 — smoothing per camera (STUB -> raises).
    smoothed = [smooth_sequence(v, config.smoothing) for v in views]

    # Stage 3 — multi-view fusion to 3D (STUB).
    pose3d = reconstruct_3d(smoothed)

    # Stage 4 — per-frame depth judgment (STUB).
    depth_results = judge_depth_sequence(pose3d, config.depth)

    # Stage 5 — segment into reps and judge each (STUB).
    reps = segment_reps(pose3d, depth_results, commands, config.reps)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return JudgeResult(
        source=", ".join(str(p) for p in paths),
        fps=views[0].fps if views else 0.0,
        frame_count=len(views[0].frames) if views else 0,
        camera_ids=[v.camera_id for v in views],
        reps=reps,
        processing_ms=elapsed_ms,
    )


def _as_path_list(video_paths: str | Path | list[str | Path]) -> list[Path]:
    if isinstance(video_paths, (str, Path)):
        return [Path(video_paths)]
    return [Path(p) for p in video_paths]
