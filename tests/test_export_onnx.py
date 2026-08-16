"""Tests for the pure half of eval/export_onnx.py.

Everything that touches ultralytics/torch/onnxruntime needs the `cv` and
`onnx` extras and real model weights, so — same split as pose.py and
traces.py — only the pure helpers run in CI. The export + verification
functions themselves were exercised by hand (see the module docstring's
measured sizes and the commit that added this file) rather than in this
suite, which has no model to export.
"""

from __future__ import annotations

from eval.export_onnx import ExportVerification, format_size


def test_format_size_picks_the_right_unit() -> None:
    assert format_size(0) == "0 B"
    assert format_size(512) == "512 B"
    assert format_size(2048) == "2.0 KB"
    assert format_size(6_255_593) == "6.0 MB"
    assert format_size(11_853_518) == "11.3 MB"
    assert format_size(2 * 1024**3) == "2.0 GB"


def test_export_verification_passes_within_tolerance() -> None:
    result = ExportVerification(
        max_abs_diff=0.005737, mean_abs_diff=0.000037, output_shape=(1, 56, 8400), atol=0.01
    )
    assert result.passed
    assert "PASS" in str(result)


def test_export_verification_fails_outside_tolerance() -> None:
    result = ExportVerification(
        max_abs_diff=0.5, mean_abs_diff=0.01, output_shape=(1, 56, 8400), atol=0.01
    )
    assert not result.passed
    assert "FAIL" in str(result)


def test_export_verification_boundary_is_inclusive() -> None:
    """Exactly at the tolerance counts as passing, not failing."""
    result = ExportVerification(max_abs_diff=0.01, mean_abs_diff=0.0, output_shape=(1,), atol=0.01)
    assert result.passed
