"""Contract tests for `smoothing.smooth_sequence` (xfail until implemented).

`strict=True` means once you implement the stub and these pass, pytest reports
XPASS-as-failure — your cue to delete the xfail marker.
"""

from __future__ import annotations

import pytest

from whitelights.smoothing import smooth_sequence

pytestmark = pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="TODO(ethan): smoothing stub"
)


def test_preserves_time_base(noisy_pose_2d) -> None:
    out = smooth_sequence(noisy_pose_2d)
    assert len(out.frames) == len(noisy_pose_2d.frames)
    assert out.fps == noisy_pose_2d.fps
    assert out.camera_id == noisy_pose_2d.camera_id
    assert [f.time_s for f in out.frames] == [f.time_s for f in noisy_pose_2d.frames]


def test_fills_short_gap(noisy_pose_2d) -> None:
    out = smooth_sequence(noisy_pose_2d)
    # frames 18-21 were dropouts; a short gap should be bridged.
    for i in range(18, 22):
        assert out.frames[i].get("left_hip") is not None
