"""Validation harness — measure pipeline agreement against labelled clips.

Takes a directory of labelled clips described by a CSV::

    filename,true_call
    clip_0001.mp4,GOOD
    clip_0002.mp4,NO_LIFT
    ...

runs the pipeline on each clip, reduces it to a single call, and reports:
  * overall agreement %  (predicted call == true_call, over judged clips)
  * per-class breakdown   (recall per true class)
  * confusion matrix      (true x predicted counts)
  * latency stats         (median / p95 / max processing time per clip)

Run with::

    python -m eval.validate --clips-dir data/clips --labels data/labels.csv
    python -m eval.validate --clips-dir data/clips --labels data/labels.csv --json

Judging a clip requires the pose runtime (the ``cv`` extra); clips that raise
are recorded as unjudged with the reason rather than aborting the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from whitelights.pipeline import judge_video
from whitelights.types import RepVerdict, Verdict

# Fixed class order for reports / confusion matrix.
CLASSES: tuple[str, ...] = tuple(v.value for v in Verdict)


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
        """Per true-class totals and correct counts (recall numerator/denom)."""
        out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in self.judged:
            bucket = out[r.true_call]
            bucket["total"] += 1
            if r.predicted == r.true_call:
                bucket["correct"] += 1
        return {k: dict(v) for k, v in out.items()}

    def confusion_matrix(self) -> dict[str, dict[str, int]]:
        """counts[true][predicted]. Rows/cols beyond CLASSES are included too."""
        matrix: dict[str, dict[str, int]] = {}
        for r in self.judged:
            row = matrix.setdefault(r.true_call, {})
            assert r.predicted is not None  # judged clips have a prediction
            row[r.predicted] = row.get(r.predicted, 0) + 1
        return matrix

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

    def to_dict(self) -> dict:
        return {
            "clips": len(self.results),
            "judged": len(self.judged),
            "agreement": self.agreement(),
            "per_class": self.per_class(),
            "confusion_matrix": self.confusion_matrix(),
            "latency": self.latency_stats(),
            "results": [asdict(r) for r in self.results],
        }


def load_labels(labels_csv: Path) -> list[tuple[str, str]]:
    """Read ``filename,true_call`` rows from the labels CSV."""
    with labels_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"filename", "true_call"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"labels CSV must have columns {required}; got {reader.fieldnames}")
        return [
            (row["filename"].strip(), row["true_call"].strip())
            for row in reader
            if row.get("filename", "").strip()
        ]


def reduce_reps_to_call(reps: list[RepVerdict]) -> str | None:
    """Collapse a clip's reps into a single call to compare against the label.

    Validation clips are assumed to be a single competition attempt (one rep). We
    take the **primary** rep — the one spanning the most frames — as the attempt,
    which is robust to spurious short segments from tracking noise. Returns the
    primary rep's verdict, or ``None`` if no rep was detected.
    """
    if not reps:
        return None
    primary = max(reps, key=lambda r: r.end_frame - r.start_frame)
    return primary.verdict.value


def _judge_clip(clips_dir: Path, filename: str) -> tuple[str | None, float, str | None]:
    """Judge one clip and reduce it to a single call.

    Returns ``(predicted_or_None, latency_ms, error_or_None)``.
    """
    path = clips_dir / filename
    started = time.perf_counter()
    try:
        result = judge_video(path)
    except Exception as exc:  # noqa: BLE001 - harness must survive a bad clip
        return None, (time.perf_counter() - started) * 1000.0, f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - started) * 1000.0
    call = reduce_reps_to_call(result.reps)
    if call is None:
        return None, latency_ms, "no reps detected"
    return call, latency_ms, None


def run(clips_dir: Path, labels_csv: Path) -> Report:
    report = Report()
    for filename, true_call in load_labels(labels_csv):
        predicted, latency_ms, error = _judge_clip(clips_dir, filename)
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


def format_confusion_matrix(matrix: dict[str, dict[str, int]]) -> str:
    """Render true x predicted counts as a fixed-width text table."""
    # Include any labels/predictions outside CLASSES so nothing is hidden.
    labels = list(CLASSES)
    for true, row in matrix.items():
        if true not in labels:
            labels.append(true)
        for pred in row:
            if pred not in labels:
                labels.append(pred)

    width = max(max((len(x) for x in labels), default=0), 4)
    header = " " * (width + 2) + " ".join(f"{c:>{width}}" for c in labels)
    lines = [header]
    for true in labels:
        row = matrix.get(true, {})
        cells = " ".join(f"{row.get(pred, 0):>{width}}" for pred in labels)
        lines.append(f"{true:>{width}}  {cells}")
    return "\n".join(lines)


def print_report(report: Report) -> None:
    total = len(report.results)
    judged = report.judged
    print("\nWhite Lights — validation report")
    print("=" * 44)
    print(f"clips:         {total}")
    print(f"judged:        {len(judged)}")
    print(f"unjudged:      {total - len(judged)}")

    if judged:
        print(f"agreement:     {report.agreement() * 100:.1f}%  (over judged clips)")
        print("\nper-class recall:")
        for cls, counts in sorted(report.per_class().items()):
            n, correct = counts.get("total", 0), counts.get("correct", 0)
            rate = (correct / n * 100) if n else 0.0
            print(f"  {cls:<12} {correct}/{n}  ({rate:.1f}%)")

        print("\nconfusion matrix (rows = true, cols = predicted):")
        print(format_confusion_matrix(report.confusion_matrix()))

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

    unknown = {r.true_call for r in report.results} - set(CLASSES)
    if unknown:
        print(f"\n⚠️  labels not in Verdict enum: {sorted(unknown)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="White Lights validation harness")
    parser.add_argument("--clips-dir", type=Path, required=True, help="Directory of clips")
    parser.add_argument("--labels", type=Path, required=True, help="labels CSV")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON")
    args = parser.parse_args()

    report = run(args.clips_dir, args.labels)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
