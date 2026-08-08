"""Wire-contract tests: what actually reaches the browser.

The other test modules assert on tracker internals (``status.last_verdict``).
That is not what the UI consumes: ``api.main.live_payload`` only serialises a
verdict on the frame where ``rep_completed`` is True, so a tracker can compute a
perfectly correct verdict that the client never sees.

These tests therefore assert on the serialised payload, for every lift x every
terminal scenario:

  * ``rep_completed`` fires (and, for single-attempt judges, exactly once);
  * the payload on that frame carries a populated ``verdict``;
  * the wire verdict matches what the tracker concluded.

This is the contract `web/live.html` depends on, and the one the tracker
refactor must preserve.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    ground_truth_depth,
    make_bench_3d,
    make_deadlift_3d,
    make_full_squat_3d,
    make_squat_3d,
    v_series,
)

from api.main import live_payload
from whitelights.bench import BenchTracker
from whitelights.deadlift import DeadliftTracker
from whitelights.depth import DepthFrameResult
from whitelights.live import CompetitionTracker, OnlineRepTracker
from whitelights.types import Fault, FrameKeypoints, Keypoint2D, Verdict

STILL = 25


def _ramp(a: float, b: float, n: int = 15) -> list[float]:
    return [float(z) for z in np.linspace(a, b, n)]


def _as_2d(frame3d) -> FrameKeypoints:
    """Project a synthetic 3D frame back to a plausible 2D frame for the payload."""
    return FrameKeypoints(
        frame_idx=frame3d.frame_idx,
        time_s=frame3d.time_s,
        keypoints={
            name: Keypoint2D(name=name, x=kp.x, y=-kp.z, confidence=kp.confidence)
            for name, kp in frame3d.keypoints.items()
        },
        detected=bool(frame3d.keypoints),
        subject_confidence=frame3d.confidence,
    )


def _gated(seq) -> list[DepthFrameResult]:
    return [
        DepthFrameResult(frame_idx=f.frame_idx, time_s=f.time_s, gated=True) for f in seq.frames
    ]


def _drive_to_payloads(tracker, poses, depths) -> list[dict]:
    """Run every frame through the tracker *and* the payload serialiser."""
    payloads = []
    for frame, depth in zip(poses.frames, depths, strict=True):
        status = tracker.update(frame, depth)
        payloads.append(live_payload(_as_2d(frame), status, width=640, height=480))
    return payloads


# ---------------------------------------------------------------------------
# Scenario table: (id, tracker, poses, depths, expected verdict, expected faults)
# ---------------------------------------------------------------------------


def _squat_attempt(bottom: float, *, setup: int = STILL) -> list[float]:
    return [1.0] * setup + _ramp(1.0, bottom) + _ramp(bottom, 1.0) + [1.0] * STILL


def _squat_case(series: list[float]):
    poses = make_full_squat_3d(series)
    return CompetitionTracker(), poses, ground_truth_depth(poses)


def _bench_case(series: list[float]):
    poses = make_bench_3d(series)
    return BenchTracker(), poses, _gated(poses)


def _deadlift_case(series: list[float]):
    poses = make_deadlift_3d(series)
    return DeadliftTracker(), poses, _gated(poses)


TOP, CHEST = 1.6, 1.05
FLOOR, DL_TOP = 0.15, 1.0


def _scenarios():
    yield (
        "squat-good",
        _squat_case(_squat_attempt(0.45)),
        Verdict.GOOD,
        None,
    )
    yield (
        "squat-high",
        _squat_case(_squat_attempt(0.60)),
        Verdict.NO_LIFT,
        Fault.INSUFFICIENT_DEPTH,
    )
    yield (
        "squat-early-descent",
        _squat_case(_squat_attempt(0.45, setup=3)),
        Verdict.NO_LIFT,
        Fault.EARLY_DESCENT,
    )
    yield (
        "squat-early-rack",
        _squat_case(
            [1.0] * STILL + _ramp(1.0, 0.45) + _ramp(0.45, 1.0) + [1.0] * 3 + _ramp(1.0, 0.5, 8)
        ),
        Verdict.NO_LIFT,
        Fault.EARLY_RACK,
    )
    yield (
        "bench-good",
        _bench_case(
            [TOP] * STILL + _ramp(TOP, CHEST) + [CHEST] * STILL + _ramp(CHEST, TOP) + [TOP] * STILL
        ),
        Verdict.GOOD,
        None,
    )
    yield (
        "bench-not-to-chest",
        _bench_case(
            [TOP] * STILL + _ramp(TOP, 1.35) + [1.35] * STILL + _ramp(1.35, TOP) + [TOP] * STILL
        ),
        Verdict.NO_LIFT,
        Fault.BAR_NOT_TO_CHEST,
    )
    yield (
        "deadlift-good",
        _deadlift_case(
            [FLOOR] * STILL + _ramp(FLOOR, DL_TOP, 20) + [DL_TOP] * STILL + _ramp(DL_TOP, FLOOR, 15)
        ),
        Verdict.GOOD,
        None,
    )
    yield (
        "deadlift-early-down",
        _deadlift_case(
            [FLOOR] * STILL + _ramp(FLOOR, DL_TOP, 20) + [DL_TOP] * 3 + _ramp(DL_TOP, FLOOR, 15)
        ),
        Verdict.NO_LIFT,
        Fault.EARLY_DOWN,
    )
    # The regression this module was written for: the lifter pulls partway, never
    # locks out, and sets the bar down. The tracker concludes NO_LIFT — and used
    # to finalise without emitting a command, so `rep_completed` never fired and
    # the browser was never told.
    yield (
        "deadlift-never-locked-out",
        _deadlift_case(
            [FLOOR] * STILL + _ramp(FLOOR, 0.7, 20) + [0.7] * 10 + _ramp(0.7, FLOOR, 15)
        ),
        Verdict.NO_LIFT,
        Fault.INCOMPLETE_LOCKOUT,
    )


SCENARIOS = list(_scenarios())


@pytest.mark.parametrize(
    ("case", "expected_verdict", "expected_fault"),
    [(c, v, f) for _, c, v, f in SCENARIOS],
    ids=[name for name, _, _, _ in SCENARIOS],
)
def test_verdict_reaches_the_wire(case, expected_verdict, expected_fault) -> None:
    """Every terminal outcome must be delivered to the client, not just computed."""
    tracker, poses, depths = case
    payloads = _drive_to_payloads(tracker, poses, depths)

    completed = [p for p in payloads if p["rep_completed"]]
    assert len(completed) == 1, (
        f"expected exactly one completed frame on the wire, got {len(completed)} — "
        "a verdict the client never receives is a verdict that did not happen"
    )

    wire = completed[0]["verdict"]
    assert wire is not None, "rep_completed fired but no verdict was serialised"
    assert wire["verdict"] == expected_verdict.value
    if expected_fault is not None:
        assert expected_fault.value in wire["faults"]


@pytest.mark.parametrize(
    "case",
    [c for _, c, _, _ in SCENARIOS],
    ids=[name for name, _, _, _ in SCENARIOS],
)
def test_payload_shape_is_stable(case) -> None:
    """Every frame carries the keys web/live.html reads, whatever the state."""
    tracker, poses, depths = case
    required = {
        "state",
        "checkpoint_met",
        "below_parallel",
        "lift_progress",
        "depth_progress",
        "rep_completed",
        "verdict",
        "note",
        "keypoints",
        "command",
    }
    for payload in _drive_to_payloads(tracker, poses, depths):
        assert required <= payload.keys()
        assert isinstance(payload["rep_completed"], bool)
        assert 0.0 <= payload["lift_progress"] <= 1.0


def test_training_mode_reports_every_rep() -> None:
    """Free-rep mode: each completed rep is delivered, not only the last."""
    three_reps = v_series(1.0, 0.45, n=40) * 3
    poses = make_squat_3d(three_reps)
    payloads = _drive_to_payloads(OnlineRepTracker(), poses, ground_truth_depth(poses))

    completed = [p for p in payloads if p["rep_completed"]]
    assert len(completed) >= 2, "expected repeated reps to each be reported"
    for p in completed:
        assert p["verdict"] is not None
        assert p["verdict"]["verdict"] in {v.value for v in Verdict}
