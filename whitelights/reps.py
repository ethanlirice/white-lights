"""Rep segmentation + per-rep verdict state machine — STUB.

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

Contract
--------
Input:  a :class:`~whitelights.types.Pose3DSequence`, the matching
        ``list[DepthFrameResult]`` (same length / frame indices), and an
        optional ``list[RefereeCommand]``.
Output: ``list[RepVerdict]`` — one per detected attempt, in time order.

Requirements the implementation must honour:
  * One verdict per rep; segment on the descend/bottom/ascend motion cycle.
  * Depth verdict: NO_LIFT + INSUFFICIENT_DEPTH unless some bottom frame was
    unambiguously below parallel; GOOD when it was and no other fault fired.
  * UNCERTAIN when the bottom of the rep was confidence-gated (no trustworthy
    depth reading) — never force a call on missing signal.
  * Fault flags accumulate: a rep may carry several faults at once.
  * Commands (when provided): descending before START -> EARLY_DESCENT; racking
    (leaving lockout) before RACK -> EARLY_RACK. When absent, skip these checks.

Roadmap: v2.0 implements DEPTH only; DOWNWARD_MOVEMENT, command-timing, and
postural faults land incrementally (DESIGN.md). Interfaces already carry the
inputs those need so no signature changes are required later.

TODO(ethan): implement the state machine.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from .depth import DepthFrameResult
from .types import Pose3DSequence, RefereeCommand, RepVerdict


class RepState(StrEnum):
    WAITING_FOR_START = "WAITING_FOR_START"
    DESCENDING = "DESCENDING"
    BOTTOM = "BOTTOM"
    ASCENDING = "ASCENDING"
    LOCKED_OUT = "LOCKED_OUT"
    RACKED = "RACKED"


class RepConfig(BaseModel):
    """Tunables for segmentation / motion thresholds."""

    # Minimum vertical hip travel (world units) to count as a real descent, to
    # reject fidgeting/setup noise.
    min_descent_travel: float = 0.1
    # Hip-velocity threshold (world units/s) below which motion is "still",
    # used to detect BOTTOM and LOCKED_OUT dwell.
    still_velocity: float = 0.05
    # Re-descent (world units) during ascent that trips DOWNWARD_MOVEMENT.
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
        commands: optional referee commands bounding the attempt.
        config: segmentation tunables; defaults applied when ``None``.

    Returns:
        One `RepVerdict` per detected attempt, in time order.
    """
    raise NotImplementedError("TODO(ethan): rep state machine not implemented")
