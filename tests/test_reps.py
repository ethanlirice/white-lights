"""Contract tests for `reps.segment_reps` (xfail until implemented).

Depth results are supplied as ground truth (via the `make_depth` fixture) so
these test the state machine in isolation from the depth stub.
"""

from __future__ import annotations

import pytest

from whitelights.reps import segment_reps
from whitelights.types import Command, Fault, RefereeCommand, Verdict

pytestmark = pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="TODO(ethan): reps stub"
)


def test_good_squat_yields_one_good_rep(good_squat_3d, make_depth) -> None:
    verdicts = segment_reps(good_squat_3d, make_depth(good_squat_3d))
    assert len(verdicts) == 1
    assert verdicts[0].verdict == Verdict.GOOD
    assert verdicts[0].faults == []


def test_high_squat_is_no_lift_for_depth(high_squat_3d, make_depth) -> None:
    verdicts = segment_reps(high_squat_3d, make_depth(high_squat_3d))
    assert len(verdicts) == 1
    assert verdicts[0].verdict == Verdict.NO_LIFT
    assert Fault.INSUFFICIENT_DEPTH in verdicts[0].faults


def test_double_bounce_flags_downward_movement(double_bounce_3d, make_depth) -> None:
    verdicts = segment_reps(double_bounce_3d, make_depth(double_bounce_3d))
    assert len(verdicts) == 1
    assert Fault.DOWNWARD_MOVEMENT in verdicts[0].faults
    assert verdicts[0].verdict == Verdict.NO_LIFT


def test_descent_before_start_command_flags_early_descent(good_squat_3d, make_depth) -> None:
    # START issued late (after descent has begun) -> EARLY_DESCENT.
    commands = [
        RefereeCommand(command=Command.START, time_s=good_squat_3d.frames[-1].time_s),
        RefereeCommand(command=Command.RACK, time_s=good_squat_3d.frames[-1].time_s + 1),
    ]
    verdicts = segment_reps(good_squat_3d, make_depth(good_squat_3d), commands=commands)
    assert Fault.EARLY_DESCENT in verdicts[0].faults
