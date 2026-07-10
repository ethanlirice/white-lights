"""Validation harness skeleton — measure pipeline agreement against labels.

Takes a directory of labelled clips described by a CSV::

    filename,true_call
    clip_0001.mp4,GOOD
    clip_0002.mp4,NO_LIFT
    ...

runs the pipeline on each clip, and reports:
  * overall agreement %  (predicted verdict == true_call)
  * per-class breakdown   (precision/recall-style, confusion counts)
  * latency stats         (median / p95 processing time per clip)

STRUCTURE ONLY. The scoring/reporting scaffolding is real so you can wire in the
pipeline once the core logic exists; per-clip judging currently surfaces the
"not implemented" state. Run with::

    python -m eval.validate --clips-dir data/labelled --labels data/labels.csv

TODO(ethan): flesh out `_verdict_for_clip` policy (e.g. how a multi-rep clip
maps to a single call) and the confusion-matrix rendering.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from whitelights.pipeline import judge_video
from whitelights.types import Verdict


@dataclass
class ClipResult:
    filename: str
    true_call: str
    predicted: str | None  # None when the clip could not be judged
    latency_ms: float
    error: str | None = None


@dataclass
class Report:
    results: list[ClipResult] = field(default_factory=list)

    @property
    def judged(self) -> list[ClipResult]:
        return [r for r in self.results if r.predicted is not None]

    def agreement(self) -> float:
        judged = self.judged
        if not judged:
            return 0.0
        hits = sum(1 for r in judged if r.predicted == r.true_call)
        return hits / len(judged)

    def per_class(self) -> dict[str, dict[str, int]]:
        """Per true-class counts: total / correct / and predicted distribution."""
        out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in self.judged:
            bucket = out[r.true_call]
            bucket["total"] += 1
            if r.predicted == r.true_call:
                bucket["correct"] += 1
            bucket[f"pred_{r.predicted}"] += 1
        return {k: dict(v) for k, v in out.items()}

    def latency_stats(self) -> dict[str, float]:
        lat = [r.latency_ms for r in self.results if r.latency_ms > 0]
        if not lat:
            return {"count": 0}
        lat.sort()
        p95_idx = min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))
        return {
            "count": len(lat),
            "median_ms": statistics.median(lat),
            "p95_ms": lat[p95_idx],
            "max_ms": lat[-1],
        }


def load_labels(labels_csv: Path) -> list[tuple[str, str]]:
    """Read ``filename,true_call`` rows from the labels CSV."""
    with labels_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"filename", "true_call"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"labels CSV must have columns {required}; got {reader.fieldnames}")
        return [(row["filename"].strip(), row["true_call"].strip()) for row in reader]


def _verdict_for_clip(clips_dir: Path, filename: str) -> tuple[str | None, float, str | None]:
    """Judge one clip and reduce it to a single call.

    Returns ``(predicted_or_None, latency_ms, error_or_None)``.

    TODO(ethan): define how a multi-rep clip collapses to one call. For now we
    assume one rep per validation clip and take that rep's verdict.
    """
    path = clips_dir / filename
    started = time.perf_counter()
    try:
        result = judge_video(path)
    except NotImplementedError as exc:
        return None, 0.0, f"not_implemented: {exc}"
    except Exception as exc:  # noqa: BLE001 - harness should be robust to bad clips
        return None, (time.perf_counter() - started) * 1000.0, str(exc)
    latency_ms = (time.perf_counter() - started) * 1000.0
    if not result.reps:
        return None, latency_ms, "no reps detected"
    return result.reps[0].verdict.value, latency_ms, None


def run(clips_dir: Path, labels_csv: Path) -> Report:
    report = Report()
    for filename, true_call in load_labels(labels_csv):
        predicted, latency_ms, error = _verdict_for_clip(clips_dir, filename)
        report.results.append(
            ClipResult(
                filename=filename,
                true_call=true_call,
                predicted=predicted,
                latency_ms=latency_ms,
                error=error,
            )
        )
    return report


def print_report(report: Report) -> None:
    total = len(report.results)
    judged = report.judged
    print("\nWhite Lights — validation report")
    print("=" * 40)
    print(f"clips:         {total}")
    print(f"judged:        {len(judged)}")
    print(f"unjudged:      {total - len(judged)}")

    if judged:
        print(f"agreement:     {report.agreement() * 100:.1f}%  (over judged clips)")
        print("\nper-class:")
        for cls, counts in sorted(report.per_class().items()):
            n, correct = counts.get("total", 0), counts.get("correct", 0)
            rate = (correct / n * 100) if n else 0.0
            print(f"  {cls:<12} {correct}/{n}  ({rate:.1f}%)")

    lat = report.latency_stats()
    if lat.get("count"):
        print("\nlatency:")
        print(
            f"  median {lat['median_ms']:.0f} ms · "
            f"p95 {lat['p95_ms']:.0f} ms · max {lat['max_ms']:.0f} ms"
        )

    errors = [r for r in report.results if r.error]
    if errors:
        print(f"\n{len(errors)} clip(s) not judged. First few:")
        for r in errors[:5]:
            print(f"  {r.filename}: {r.error}")

    # Sanity: label vocabulary should be a subset of known verdicts.
    unknown = {r.true_call for r in report.results} - {v.value for v in Verdict}
    if unknown:
        print(f"\n⚠️  labels not in Verdict enum: {sorted(unknown)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="White Lights validation harness")
    parser.add_argument("--clips-dir", type=Path, required=True, help="Directory of clips")
    parser.add_argument("--labels", type=Path, required=True, help="labels CSV")
    args = parser.parse_args()

    report = run(args.clips_dir, args.labels)
    print_report(report)


if __name__ == "__main__":
    main()
