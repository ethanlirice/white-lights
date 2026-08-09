"""Keypoint traces: run the model once, validate for ever after.

`eval/validate.py` judges video, which means every validation run needs the
``cv`` extra, a GPU-less minute per clip, and the clips themselves — so it can
only ever be a thing you run by hand, on your own machine, on footage you cannot
redistribute.

A **trace** is the pose model's output for one clip, saved as JSON: the same
`PoseSequence` the pipeline consumes, minus the video. Extracting traces once and
validating against those instead changes what validation *is*:

  * it runs **without torch**, so it works in CI on every push;
  * traces are kilobytes, so they can live in the repo as fixtures;
  * no footage is redistributed — a trace is coordinates, not a person;
  * the judging logic gets a regression gate, not a one-off script.

The split matters because the two halves fail differently. Pose extraction is
slow, needs the heavy stack, and changes only when the *model* changes. Judging
is fast, pure, and changes every time you touch a threshold. Only the second half
belongs in a loop you run constantly.

Workflow::

    # once, wherever the clips and the cv extra live
    python -m eval.traces extract --clips-dir data/clips --out data/traces

    # thereafter, anywhere — including CI
    python -m eval.validate --traces-dir data/traces --labels data/labels.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from whitelights.types import PoseSequence

TRACE_SUFFIX = ".trace.json"
#: Bumped when the on-disk shape changes, so stale traces are rejected loudly
#: rather than silently judged under different assumptions.
TRACE_FORMAT_VERSION = 1


def trace_path(directory: Path, clip_name: str) -> Path:
    """Where the trace for ``clip_name`` lives (extension-insensitive)."""
    return directory / (Path(clip_name).stem + TRACE_SUFFIX)


def save_trace(sequence: PoseSequence, path: Path, *, source_clip: str) -> None:
    """Write one pose track to disk as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": TRACE_FORMAT_VERSION,
        "source_clip": source_clip,
        "sequence": sequence.model_dump(mode="json"),
    }
    path.write_text(json.dumps(payload, separators=(",", ":")))


def load_trace(path: Path) -> PoseSequence:
    """Read a trace back into the same type the pipeline consumes.

    Raises `ValueError` on a version mismatch — a trace written under different
    assumptions is worse than no trace, because it validates silently.
    """
    payload = json.loads(path.read_text())
    version = payload.get("format_version")
    if version != TRACE_FORMAT_VERSION:
        raise ValueError(
            f"{path.name}: trace format v{version}, expected v{TRACE_FORMAT_VERSION} — "
            "re-extract with `python -m eval.traces extract`"
        )
    return PoseSequence.model_validate(payload["sequence"])


def load_traces(directory: Path) -> dict[str, PoseSequence]:
    """Load every trace in ``directory``, keyed by source clip stem."""
    return {
        p.name[: -len(TRACE_SUFFIX)]: load_trace(p)
        for p in sorted(directory.glob(f"*{TRACE_SUFFIX}"))
    }


def extract(clips_dir: Path, out_dir: Path, *, pattern: str = "*.mp4") -> list[Path]:
    """Run pose estimation over a directory of clips and save the traces.

    This is the only step that needs the ``cv`` extra. Clips that fail are
    reported and skipped rather than aborting a long run.
    """
    from whitelights.pose import PoseEstimator

    estimator = PoseEstimator()
    written: list[Path] = []
    clips = sorted(clips_dir.glob(pattern))
    if not clips:
        print(f"no clips matching {pattern!r} in {clips_dir}")
        return written

    for clip in clips:
        destination = trace_path(out_dir, clip.name)
        try:
            sequence = estimator.run_video(clip)
        except Exception as exc:  # noqa: BLE001 - one bad clip must not end the run
            print(f"  skip {clip.name}: {type(exc).__name__}: {exc}")
            continue
        save_trace(sequence, destination, source_clip=clip.name)
        written.append(destination)
        size_kb = destination.stat().st_size / 1024
        frames = len(sequence.frames)
        print(f"  {clip.name} -> {destination.name}  ({frames} frames, {size_kb:.0f} KB)")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and inspect keypoint traces")
    sub = parser.add_subparsers(dest="command", required=True)

    extract_cmd = sub.add_parser("extract", help="run the pose model over clips (needs [cv])")
    extract_cmd.add_argument("--clips-dir", type=Path, required=True)
    extract_cmd.add_argument("--out", type=Path, required=True)
    extract_cmd.add_argument("--pattern", default="*.mp4")

    list_cmd = sub.add_parser("list", help="summarise saved traces (no model needed)")
    list_cmd.add_argument("--traces-dir", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "extract":
        written = extract(args.clips_dir, args.out, pattern=args.pattern)
        print(f"\nwrote {len(written)} trace(s) to {args.out}")
        return

    traces = load_traces(args.traces_dir)
    print(f"\n{len(traces)} trace(s) in {args.traces_dir}")
    for name, sequence in traces.items():
        detected = sum(1 for f in sequence.frames if f.detected)
        print(
            f"  {name:<28} {len(sequence.frames):>5} frames  "
            f"{detected / max(1, len(sequence.frames)) * 100:>5.1f}% detected  "
            f"{sequence.fps:g} fps"
        )


if __name__ == "__main__":
    main()
