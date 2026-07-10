"""Tests for the validation harness's pure scoring/reporting/reduction logic.

Pipeline execution (`_judge_clip`) is not exercised here — it needs the pose
runtime and real clips; these cover the deterministic accounting around it.
"""

from __future__ import annotations

import pytest

from eval.validate import (
    ClipResult,
    Report,
    format_confusion_matrix,
    load_labels,
    reduce_reps_to_call,
)
from whitelights.types import RepVerdict, Verdict


def _clip(
    true: str, pred: str | None, latency: float = 10.0, error: str | None = None
) -> ClipResult:
    return ClipResult(
        filename=f"{true}.mp4", true_call=true, predicted=pred, latency_ms=latency, error=error
    )


def _rep(rep_index: int, verdict: Verdict, start: int, end: int) -> RepVerdict:
    return RepVerdict(
        rep_index=rep_index,
        verdict=verdict,
        confidence=0.9,
        start_frame=start,
        end_frame=end,
        start_time_s=start / 30.0,
        end_time_s=end / 30.0,
    )


def test_agreement_over_judged_only() -> None:
    report = Report(
        results=[
            _clip("GOOD", "GOOD"),
            _clip("NO_LIFT", "NO_LIFT"),
            _clip("GOOD", "NO_LIFT"),
            _clip("NO_LIFT", None, latency=0.0, error="boom"),  # unjudged, excluded
        ]
    )
    assert report.agreement() == pytest.approx(2 / 3)
    assert len(report.judged) == 3


def test_agreement_zero_when_nothing_judged() -> None:
    report = Report(results=[_clip("GOOD", None, latency=0.0, error="x")])
    assert report.agreement() == 0.0


def test_per_class_recall_counts() -> None:
    report = Report(
        results=[
            _clip("GOOD", "GOOD"),
            _clip("GOOD", "NO_LIFT"),
            _clip("NO_LIFT", "NO_LIFT"),
        ]
    )
    per_class = report.per_class()
    assert per_class["GOOD"] == {"total": 2, "correct": 1}
    assert per_class["NO_LIFT"] == {"total": 1, "correct": 1}


def test_confusion_matrix_counts() -> None:
    report = Report(
        results=[
            _clip("GOOD", "GOOD"),
            _clip("GOOD", "NO_LIFT"),
            _clip("NO_LIFT", "NO_LIFT"),
        ]
    )
    matrix = report.confusion_matrix()
    assert matrix["GOOD"] == {"GOOD": 1, "NO_LIFT": 1}
    assert matrix["NO_LIFT"] == {"NO_LIFT": 1}
    # Renders without error and mentions the classes.
    rendered = format_confusion_matrix(matrix)
    assert "GOOD" in rendered and "NO_LIFT" in rendered


def test_latency_stats() -> None:
    report = Report(
        results=[_clip("GOOD", "GOOD", latency=lat) for lat in (10.0, 20.0, 30.0, 40.0)]
    )
    stats = report.latency_stats()
    assert stats["count"] == 4
    assert stats["median_ms"] == pytest.approx(25.0)
    assert stats["max_ms"] == 40.0


def test_reduce_reps_takes_longest_span() -> None:
    reps = [
        _rep(0, Verdict.NO_LIFT, start=0, end=2),  # short spurious segment
        _rep(1, Verdict.GOOD, start=5, end=55),  # the real attempt
    ]
    assert reduce_reps_to_call(reps) == "GOOD"


def test_reduce_reps_empty_is_none() -> None:
    assert reduce_reps_to_call([]) is None


def test_load_labels(tmp_path) -> None:
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("filename,true_call\nclip_0001.mp4,GOOD\nclip_0002.mp4,NO_LIFT\n")
    assert load_labels(csv_path) == [("clip_0001.mp4", "GOOD"), ("clip_0002.mp4", "NO_LIFT")]


def test_load_labels_rejects_bad_header(tmp_path) -> None:
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("file,call\nclip.mp4,GOOD\n")
    with pytest.raises(ValueError):
        load_labels(csv_path)
