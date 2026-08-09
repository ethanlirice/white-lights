"""Keypoint traces: round-trip fidelity and the torch-free validation path.

The whole point of a trace is that judging can be re-run without the model, so
these tests do exactly that — build a pose track, save it, load it, and drive the
real pipeline off the result, all without the ``cv`` extra installed.
"""

from __future__ import annotations

import json

import pytest
from conftest import make_squat_3d, v_series

from eval.traces import (
    TRACE_FORMAT_VERSION,
    load_trace,
    load_traces,
    save_trace,
    trace_path,
)
from eval.validate import run
from whitelights.pipeline import judge_sequences
from whitelights.types import FrameKeypoints, Keypoint2D, PoseSequence


def _pose_track(frames: int = 40, *, camera_id: str = "cam0") -> PoseSequence:
    """A 2D track shaped like real pose output, including a detection gap."""
    truth = make_squat_3d(v_series(1.0, 0.45, n=frames))
    out: list[FrameKeypoints] = []
    for i, frame3d in enumerate(truth.frames):
        if 18 <= i <= 20:  # dropout, as a real model produces
            out.append(FrameKeypoints(frame_idx=i, time_s=i / 30.0, detected=False))
            continue
        out.append(
            FrameKeypoints(
                frame_idx=i,
                time_s=i / 30.0,
                keypoints={
                    name: Keypoint2D(
                        name=name, x=kp.x * 100, y=-kp.z * 100, confidence=kp.confidence
                    )
                    for name, kp in frame3d.keypoints.items()
                },
                detected=True,
                subject_confidence=0.9,
            )
        )
    return PoseSequence(camera_id=camera_id, fps=30.0, frames=out, source="synthetic")


def test_trace_round_trips_exactly(tmp_path) -> None:
    original = _pose_track()
    path = trace_path(tmp_path, "clip_0001.mp4")
    save_trace(original, path, source_clip="clip_0001.mp4")

    restored = load_trace(path)
    assert restored == original


def test_trace_filename_is_derived_from_the_clip(tmp_path) -> None:
    assert trace_path(tmp_path, "clip_0001.mp4").name == "clip_0001.trace.json"
    assert trace_path(tmp_path, "clip_0001.mov").name == "clip_0001.trace.json"


def test_trace_is_small_enough_to_commit(tmp_path) -> None:
    """Traces live in the repo as fixtures, so size is a feature, not a detail."""
    path = trace_path(tmp_path, "clip.mp4")
    save_trace(_pose_track(frames=300), path, source_clip="clip.mp4")

    kb_per_second = path.stat().st_size / 1024 / (300 / 30.0)
    assert kb_per_second < 100, f"{kb_per_second:.0f} KB/s is too big to keep in git"


def test_stale_format_is_rejected_loudly(tmp_path) -> None:
    """A trace written under different assumptions validates silently — refuse it."""
    path = trace_path(tmp_path, "clip.mp4")
    save_trace(_pose_track(), path, source_clip="clip.mp4")
    payload = json.loads(path.read_text())
    payload["format_version"] = TRACE_FORMAT_VERSION + 1
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="re-extract"):
        load_trace(path)


def test_load_traces_keys_by_clip_stem(tmp_path) -> None:
    for name in ("a.mp4", "b.mp4"):
        save_trace(_pose_track(), trace_path(tmp_path, name), source_clip=name)

    traces = load_traces(tmp_path)
    assert sorted(traces) == ["a", "b"]


def test_judging_a_trace_needs_no_model(tmp_path) -> None:
    """The point of the split: everything after pose runs without torch."""
    path = trace_path(tmp_path, "clip.mp4")
    save_trace(_pose_track(), path, source_clip="clip.mp4")

    result = judge_sequences([load_trace(path)], source="clip.mp4")
    assert result.frame_count == 40
    assert result.source == "clip.mp4"
    assert result.camera_ids == ["cam0"]


def test_validate_runs_end_to_end_from_traces(tmp_path) -> None:
    """`eval.validate --traces-dir` is the CI-runnable form of the harness."""
    traces_dir = tmp_path / "traces"
    for name in ("clip_0001.mp4", "clip_0002.mp4"):
        save_trace(_pose_track(), trace_path(traces_dir, name), source_clip=name)

    labels = tmp_path / "labels.csv"
    labels.write_text("filename,true_call\nclip_0001.mp4,GOOD\nclip_0002.mp4,GOOD\n")

    report = run(clips_dir=None, labels_csv=labels, traces_dir=traces_dir)
    assert len(report.results) == 2
    # Every clip must be judged — an unjudged clip here means the trace path is
    # broken, which is different from the judge disagreeing with the label.
    assert all(r.error is None for r in report.results), [r.error for r in report.results]
    assert 0.0 <= report.agreement() <= 1.0
