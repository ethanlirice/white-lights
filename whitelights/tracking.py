"""Shared machinery for the online (causal) lift judges.

Every lift judge is the same shape — track a primary signal against a body-scale
reference, decide when the lifter is *still*, hold a condition for N seconds
before issuing a command, accumulate evidence, then assemble a verdict — and the
three trackers had all of it copy-pasted: ``_held`` verbatim three times,
``_status`` and the fault-assembly block four times each, and half a dozen
near-identical signal/scale helpers.

That machinery lives here. What deliberately does *not* live here is the state
graph: the three lifts have genuinely different command topologies (the deadlift
has no return phase, the bench has a mid-lift PRESS command, the squat waits in
SET after its start command and the bench does not). Parameterising one graph
over those differences produces something harder to read than three explicit
state machines, so each lift keeps its own ``update``. The duplication that is
gone is the duplication that was accidental.

Signals are heights along the world +z axis; distances are expressed as
fractions of a body reference (thigh, arm, torso) so thresholds stay
scale-invariant. Every helper returns ``None`` rather than guessing when the
keypoints it needs are missing or below confidence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .posture import joint_angle_deg
from .types import Fault, FrameKeypoints3D, Verdict

SIDES = ("left", "right")


# ---------------------------------------------------------------------------
# Signals and body-scale references
# ---------------------------------------------------------------------------


def mean_height(
    frame: FrameKeypoints3D, names: tuple[str, ...], min_confidence: float
) -> float | None:
    """Mean world height (z) of ``names``, or None if none are confident enough.

    The primary signal for every lift: hips for the squat, wrists (as a bar
    proxy) for the bench and deadlift.
    """
    heights = [
        kp.z
        for name in names
        if (kp := frame.get(name)) is not None and kp.confidence >= min_confidence
    ]
    return sum(heights) / len(heights) if heights else None


def segment_length(
    frame: FrameKeypoints3D, chain: tuple[str, ...], min_confidence: float
) -> float | None:
    """Mean total length of a left/right keypoint chain, or None.

    ``chain`` names joints without a side prefix, e.g. ``("hip", "knee")`` for
    the thigh, ``("shoulder", "elbow", "wrist")`` for the arm, ``("shoulder",
    "hip")`` for the torso. Both sides are measured and averaged; a side missing
    any link is skipped entirely rather than partially counted.
    """
    lengths: list[float] = []
    for side in SIDES:
        points = [frame.get(f"{side}_{joint}") for joint in chain]
        if any(p is None for p in points):
            continue
        if min(p.confidence for p in points) < min_confidence:  # type: ignore[union-attr]
            continue
        total = sum(
            math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))  # type: ignore[union-attr]
            for a, b in zip(points, points[1:], strict=False)
        )
        lengths.append(total)
    return sum(lengths) / len(lengths) if lengths else None


def governing_angle(
    frame: FrameKeypoints3D, chain: tuple[str, str, str], min_confidence: float
) -> float | None:
    """The more-bent (minimum) of the left/right ``a-b-c`` joint angles, or None.

    The bent side governs: a lift is only locked out when *both* sides are.
    """
    a, b, c = chain
    angles = [
        angle
        for side in SIDES
        if (
            angle := joint_angle_deg(
                frame, f"{side}_{a}", f"{side}_{b}", f"{side}_{c}", min_confidence
            )
        )
        is not None
    ]
    return min(angles) if angles else None


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------


class MotionTracker:
    """Frame-to-frame velocity of the primary signal, and a stillness test.

    Velocity is per second and compared against a fraction of the body-scale
    reference, so "still" means the same thing whether the signal is in pixels
    or metres.
    """

    def __init__(self, still_velocity_fraction: float) -> None:
        self.still_velocity_fraction = still_velocity_fraction
        self.reset()

    def reset(self) -> None:
        self._prev: float | None = None
        self._prev_t: float | None = None
        self.velocity = 0.0

    def observe(self, value: float, time_s: float) -> float:
        """Record this frame's signal and return the current velocity."""
        dt = time_s - self._prev_t if self._prev_t is not None else None
        if dt and dt > 0 and self._prev is not None:
            self.velocity = (value - self._prev) / dt
        else:
            self.velocity = 0.0
        self._prev, self._prev_t = value, time_s
        return self.velocity

    def is_still(self, scale: float) -> bool:
        return abs(self.velocity) < self.still_velocity_fraction * scale


class HoldTimer:
    """True once a condition has held continuously for ``hold_s`` seconds.

    This is how every referee command is earned: the lifter must be still (and,
    where the rulebook says so, locked) for an unbroken interval — a single
    frame that breaks the condition restarts the clock.
    """

    def __init__(self) -> None:
        self._start: float | None = None

    def reset(self) -> None:
        self._start = None

    def held(self, time_s: float, condition: bool, hold_s: float) -> bool:
        if not condition:
            self._start = None
            return False
        if self._start is None:
            self._start = time_s
        return (time_s - self._start) >= hold_s


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def decide(faults: Sequence[Fault], *, uncertain: bool = False) -> Verdict:
    """Faults lose; genuine ambiguity is UNCERTAIN; otherwise good.

    Order matters, and it is the same order in all four judges: a hard fault is a
    hard fault even when some *other* signal was borderline, so faults are
    checked before uncertainty. Encoding it once means the four cannot drift into
    disagreeing about what a borderline-but-faulted attempt is.
    """
    if faults:
        return Verdict.NO_LIFT
    if uncertain:
        return Verdict.UNCERTAIN
    return Verdict.GOOD
