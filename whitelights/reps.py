"""Rep segmentation + per-rep verdict state machine.

Turns a continuous 3D pose track (plus per-frame depth results and optional
referee commands) into a list of discrete rep attempts, each with exactly one
:class:`~whitelights.types.RepVerdict`.

State machine (see DESIGN.md for the rationale)::

    WAITING_FOR_START -> DESCENDING -> BOTTOM -> ASCENDING -> LOCKED_OUT -> RACKED
                                          |
                                          +-- re-descent on ascent -> DOWNWARD_MOVEMENT fault

The driving signal is the subject's vertical hip trajectory over time (from the
pose track); `DepthFrameResult` supplies the pass/fail evidence at the bottom;
`RefereeCommand`s (optional) bound the attempt so command-timing faults can be
checked.

What this implementation does (v2.0 — depth only)
------------------------------------------------
  * Segments reps from the hip trajectory using a scale-invariant midpoint
    crossing with hysteresis, measured relative to the observed hip-travel range
    of the clip. This holds whether ``z`` is in pixels (single-camera fallback)
    or metric units (real triangulation).
  * Verdict per rep: GOOD when some confident frame in the rep was below
    parallel; NO_LIFT + INSUFFICIENT_DEPTH when confident frames were seen but
    none reached depth; UNCERTAIN when the rep had no confident depth reading
    (never forces a call on missing signal).

Deferred (interfaces already carry the needed inputs, so no signature changes):
  * v2.1 — DOWNWARD_MOVEMENT (re-descent on the ascent).
  * v2.2 — command-timing faults (EARLY_DESCENT / EARLY_RACK); ``commands`` is
    accepted now but not yet consulted.
  * v2.3+ — postural / foot faults.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from .depth import DepthFrameResult
from .types import Fault, Pose3DSequence, RefereeCommand, RepVerdict, Verdict

_HIP_KEYPOINTS = ("left_hip", "right_hip")


class RepState(StrEnum):
    WAITING_FOR_START = "WAITING_FOR_START"
    DESCENDING = "DESCENDING"
    BOTTOM = "BOTTOM"
    ASCENDING = "ASCENDING"
    LOCKED_OUT = "LOCKED_OUT"
    RACKED = "RACKED"


class RepConfig(BaseModel):
    """Tunables for segmentation / motion thresholds."""

    # Minimum vertical hip travel (world units) for a clip to contain a rep, to
    # reject fidgeting/setup noise. Absolute; comfortably below real squat travel
    # in both pixel and metric units.
    min_descent_travel: float = 0.1
    # Fraction of the hip-travel range, below the top, at which the lifter is
    # judged to be descending into a rep (enter) vs. back at lockout (exit).
    enter_fraction: float = 0.5
    exit_fraction: float = 0.2
    # Re-descent (world units) during ascent that trips DOWNWARD_MOVEMENT (v2.1).
    downward_movement_tolerance: float = 0.02


def segment_reps(
    poses: Pose3DSequence,
    depth_results: list[DepthFrameResult],
    commands: list[RefereeCommand] | None = None,
    config: RepConfig | None = None,
) -> list[RepVerdict]:
    """Segment a session into reps and emit one verdict each. See contract.

    Args:
        poses: fused 3D pose track for the whole session.
        depth_results: per-frame depth judgments aligned 1:1 with ``poses.frames``.
        commands: optional referee commands (not yet consulted — v2.2).
        config: segmentation tunables; defaults applied when ``None``.

    Returns:
        One `RepVerdict` per detected attempt, in time order.
    """
    config = config or RepConfig()
    hip_z = _hip_z_series(poses)
    segments = _segment_indices(hip_z, config)
    return [
        _verdict_for_segment(idx, start, end, poses, depth_results)
        for idx, (start, end) in enumerate(segments)
    ]


def _hip_z_series(poses: Pose3DSequence) -> list[float | None]:
    """Mean vertical hip position per frame (forward/back-filled over gaps)."""
    series: list[float | None] = []
    for f in poses.frames:
        zs = [kp.z for name in _HIP_KEYPOINTS if (kp := f.get(name)) is not None]
        series.append(sum(zs) / len(zs) if zs else None)

    # Forward-fill then back-fill so motion detection sees a continuous signal.
    last: float | None = None
    for i, v in enumerate(series):
        if v is None:
            series[i] = last
        else:
            last = v
    first_known = next((v for v in series if v is not None), None)
    return [first_known if v is None else v for v in series]


def _segment_indices(hip_z: list[float | None], config: RepConfig) -> list[tuple[int, int]]:
    """Find (start, end) index pairs for each rep via hysteresis on hip height."""
    values = [v for v in hip_z if v is not None]
    if len(values) < 3:
        return []
    top, bottom = max(values), min(values)
    travel = top - bottom
    if travel < config.min_descent_travel:
        return []

    enter = top - config.enter_fraction * travel  # descended into a rep
    exit_up = top - config.exit_fraction * travel  # returned to lockout

    segments: list[tuple[int, int]] = []
    descending = False
    start = 0
    last_top_idx = 0
    for i, v in enumerate(hip_z):
        if v is None:
            continue
        if not descending:
            if v >= exit_up:
                last_top_idx = i
            if v < enter:
                descending = True
                start = last_top_idx
        elif v > exit_up:
            segments.append((start, i))
            descending = False
            last_top_idx = i
    if descending:  # clip ended before returning to lockout
        segments.append((start, len(hip_z) - 1))
    return segments


def _verdict_for_segment(
    rep_index: int,
    start: int,
    end: int,
    poses: Pose3DSequence,
    depth_results: list[DepthFrameResult],
) -> RepVerdict:
    frames = poses.frames
    seg_depth = [depth_results[i] for i in range(start, end + 1) if i < len(depth_results)]
    confident = [d for d in seg_depth if not d.gated and d.depth_margin is not None]

    common = {
        "rep_index": rep_index,
        "start_frame": frames[start].frame_idx,
        "end_frame": frames[end].frame_idx,
        "start_time_s": frames[start].time_s,
        "end_time_s": frames[end].time_s,
    }

    if not confident:
        return RepVerdict(
            verdict=Verdict.UNCERTAIN,
            confidence=0.0,
            faults=[],
            depth_margin=None,
            notes="No confident depth reading in this rep.",
            **common,
        )

    deepest = max(confident, key=lambda d: d.depth_margin)
    reached_depth = any(d.is_below_parallel for d in confident)
    if reached_depth:
        verdict, faults = Verdict.GOOD, []
    else:
        verdict, faults = Verdict.NO_LIFT, [Fault.INSUFFICIENT_DEPTH]

    return RepVerdict(
        verdict=verdict,
        confidence=deepest.confidence,
        faults=faults,
        depth_margin=deepest.depth_margin,
        **common,
    )
