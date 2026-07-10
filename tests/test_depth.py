"""Contract tests for `depth.judge_depth_frame` (xfail until implemented)."""

from __future__ import annotations

import pytest

from whitelights.depth import judge_depth_frame

pytestmark = pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="TODO(ethan): depth stub"
)


def test_good_squat_bottom_is_below_parallel(good_squat_3d, bottom_of) -> None:
    result = judge_depth_frame(bottom_of(good_squat_3d))
    assert result.gated is False
    assert result.is_below_parallel is True
    assert result.depth_margin is not None and result.depth_margin > 0


def test_high_squat_bottom_is_not_below_parallel(high_squat_3d, bottom_of) -> None:
    result = judge_depth_frame(bottom_of(high_squat_3d))
    assert result.gated is False
    assert result.is_below_parallel is False
    assert result.depth_margin is not None and result.depth_margin < 0


def test_low_confidence_frame_is_gated(low_confidence_frame) -> None:
    result = judge_depth_frame(low_confidence_frame)
    assert result.gated is True
    assert result.is_below_parallel is None  # never guess on weak signal
