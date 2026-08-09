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
from collections.abc import Sequence
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
    video_paths: str | Path | Sequence[str | Path],
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

    # Stage 1 — pose per camera (WORKING). Everything downstream of this is
    # independent of video, which is what `judge_sequences` exposes.
    views: list[PoseSequence] = [
        estimator.run_video(p, camera_id=f"cam{i}") for i, p in enumerate(paths)
    ]
    return judge_sequences(
        views, commands=commands, config=config, source=", ".join(str(p) for p in paths)
    )


def judge_sequences(
    views: Sequence[PoseSequence],
    *,
    commands: list[RefereeCommand] | None = None,
    config: PipelineConfig | None = None,
    source: str = "sequences",
) -> JudgeResult:
    """Judge already-extracted pose tracks — the pipeline minus the model.

    Splitting this out is what lets validation run on saved keypoint traces (see
    `eval/traces.py`): pose estimation is slow, needs torch, and only changes
    when the model does, whereas everything here is fast, pure, and changes every
    time a threshold moves. Only the second half belongs in CI.
    """
    config = config or PipelineConfig()
    if not views:
        raise ValueError("judge_sequences requires at least one pose track")

    started = time.perf_counter()

    # Stage 2 — smoothing per camera.
    smoothed = [smooth_sequence(v, config.smoothing) for v in views]

    # Stage 3 — multi-view fusion to 3D (single-view lift today).
    pose3d = reconstruct_3d(smoothed)

    # Stage 4 — per-frame depth judgment.
    depth_results = judge_depth_sequence(pose3d, config.depth)

    # Stage 5 — segment into reps and judge each.
    reps = segment_reps(pose3d, depth_results, commands, config.reps)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return JudgeResult(
        source=source,
        fps=views[0].fps,
        frame_count=len(views[0].frames),
        camera_ids=[v.camera_id for v in views],
        reps=reps,
        processing_ms=elapsed_ms,
    )


def _as_path_list(video_paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(video_paths, (str, Path)):
        return [Path(video_paths)]
    return [Path(p) for p in video_paths]
