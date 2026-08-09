"""Camera-geometry sensitivity: how much does camera placement change the call?

The depth judgment reduces to comparing the hip crease and the top of the knee.
With one camera the pipeline compares their **image rows**, which is only the
same question when both land at the same distance from the lens. This module
measures how far that assumption can be pushed: it builds a squat trace with a
realistic sagittal profile, projects it through a virtual camera, runs the result
back through the real pipeline (fusion -> depth), and compares against ground
truth taken from the 3D trace it started from.

No footage is required, which is the point — this is measurable today, before a
single clip has been labelled.

What it found
-------------
1. **The landmark, not the lens, was the dominant error.** `DepthConfig` ships
   ``hip_crease_offset = 0.0``, so the judge measures the hip *joint* rather than
   the *crease* the rulebook names — a constant bias toward "not deep enough"
   that missed every genuinely-below-parallel frame on a borderline rep. Worse,
   that parameter is documented in world units but is subtracted from ``z``,
   which the single-camera lift leaves in **pixels**, so a metric value there is
   off by orders of magnitude and cannot work. Hence
   ``hip_crease_thigh_fraction``: the same correction expressed as a fraction of
   thigh length, scale-invariant like every other threshold here.

2. **Camera yaw is what matters; height and pitch barely register.** Pitch is a
   pure rotation about the lens, which is a homography — it cannot reorder two
   points vertically, so it cannot flip a hip-vs-knee row comparison. Height
   changes perspective but leaves hip and knee in nearly the same depth plane in
   a square side-on view. Yaw does not: swinging off-axis rotates the knees'
   forward travel *into* the depth axis, creating exactly the parallax a
   single view cannot recover.

3. **Tolerance scales with how deep the rep actually was** — see
   ``operating_envelope``. A clearly deep squat survives large camera error; a
   borderline one demands a square camera; a rep within ~1 cm of parallel is
   below the method's resolution at any placement.

Run it::

    python -m eval.geometry                 # envelope + height/pitch grid
    python -m eval.geometry --json          # machine-readable
    python -m eval.geometry --calibrate     # fit hip_crease_thigh_fraction
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

from whitelights.camera import CameraPose, project_sequence
from whitelights.depth import DepthConfig, judge_depth_frame
from whitelights.fusion import reconstruct_3d
from whitelights.types import FrameKeypoints3D, Keypoint3D, Pose3DSequence

FPS = 30.0

# Anatomy of the reference lifter, in metres. Segment lengths are ordinary adult
# proportions; what matters for this analysis is the *ratio* between them and the
# forward knee travel, both of which are realistic.
THIGH = 0.42
SHANK = 0.43
HIP_WIDTH = 0.18
#: How far the knee travels forward (toward the camera) between standing and the
#: bottom of a squat. This is the term that makes camera pose matter at all: set
#: it to zero and hip and knee stay coplanar, and every camera agrees.
KNEE_FORWARD_TRAVEL = 0.16
#: True vertical offset from the COCO hip *joint* keypoint down to the anatomical
#: hip *crease*. depth.DepthConfig.hip_crease_offset defaults to 0.0, i.e. the
#: pipeline currently judges the wrong landmark; --calibrate recovers this value.
TRUE_CREASE_OFFSET = 0.06


@dataclass
class SweepResult:
    """Agreement between the projected pipeline and 3D ground truth."""

    height: float
    pitch_deg: float
    yaw_deg: float
    frames: int
    disagreements: int
    verdict_flipped: bool

    @property
    def flip_rate(self) -> float:
        return self.disagreements / self.frames if self.frames else 0.0


def squat_trace(
    *, bottom_margin: float, frames: int = 60, crease_offset: float = TRUE_CREASE_OFFSET
) -> Pose3DSequence:
    """A sagittally realistic squat: knees travel forward as the hips descend.

    ``bottom_margin`` is the true depth at the bottom, in metres of hip crease
    relative to the top of the knee — positive means legal depth. The returned
    keypoints are COCO hip *joints*; the crease sits ``crease_offset`` below them,
    which is the discrepancy `depth.py` has a TODO about.
    """
    knee_z = SHANK
    # Place the hip crease at the requested margin below the knee at the bottom,
    # then work back up to the hip joint keypoint the pose model would report.
    bottom_hip_joint = knee_z - bottom_margin + crease_offset
    top_hip_joint = knee_z + THIGH * 0.95

    hip_heights = np.concatenate(
        [
            np.linspace(top_hip_joint, bottom_hip_joint, frames // 2),
            np.linspace(bottom_hip_joint, top_hip_joint, frames - frames // 2),
        ]
    )

    out: list[FrameKeypoints3D] = []
    for i, hip_z in enumerate(hip_heights):
        # Descent fraction drives forward knee travel: deepest squat, most travel.
        descent = (top_hip_joint - hip_z) / max(1e-9, top_hip_joint - bottom_hip_joint)
        knee_x = KNEE_FORWARD_TRAVEL * max(0.0, descent)  # knees track over the toes
        keypoints: dict[str, Keypoint3D] = {}
        # Sagittal (side-on) staging — how squat depth is actually judged. The
        # lifter faces +x, so forward knee travel runs across the image, and the
        # near and far limbs sit at different DEPTHS (y = +-half hip width). That
        # depth difference is the parallax the single-view lift cannot see, and
        # the reason camera placement can change the call at all.
        for side, sy in (("left", -HIP_WIDTH / 2), ("right", HIP_WIDTH / 2)):
            keypoints[f"{side}_hip"] = Keypoint3D(
                name=f"{side}_hip", x=0.0, y=sy, z=float(hip_z), confidence=0.95
            )
            keypoints[f"{side}_knee"] = Keypoint3D(
                name=f"{side}_knee", x=knee_x, y=sy, z=knee_z, confidence=0.95
            )
            keypoints[f"{side}_ankle"] = Keypoint3D(
                name=f"{side}_ankle", x=0.0, y=sy, z=0.0, confidence=0.95
            )
        out.append(
            FrameKeypoints3D(frame_idx=i, time_s=i / FPS, keypoints=keypoints, confidence=0.95)
        )
    return Pose3DSequence(fps=FPS, frames=out, camera_ids=["truth"])


def ground_truth_calls(
    sequence: Pose3DSequence, *, crease_offset: float = TRUE_CREASE_OFFSET
) -> list[bool]:
    """The correct below-parallel call per frame, straight from the 3D trace."""
    calls = []
    for frame in sequence.frames:
        hip_crease = max(kp.z for n in ("left_hip", "right_hip") if (kp := frame.get(n)))
        knee_top = max(kp.z for n in ("left_knee", "right_knee") if (kp := frame.get(n)))
        calls.append((knee_top - (hip_crease - crease_offset)) > 0)
    return calls


def pipeline_calls(
    sequence: Pose3DSequence, camera: CameraPose, config: DepthConfig | None = None
) -> list[bool | None]:
    """What the real pipeline says after seeing only this camera's 2D view."""
    view = project_sequence(sequence, camera)
    lifted = reconstruct_3d([view])
    return [judge_depth_frame(f, config).is_below_parallel for f in lifted.frames]


def evaluate(sequence: Pose3DSequence, camera: CameraPose, config: DepthConfig) -> SweepResult:
    """Compare pipeline calls against ground truth for one camera placement."""
    truth = ground_truth_calls(sequence)
    predicted = pipeline_calls(sequence, camera, config)

    judged = [(t, p) for t, p in zip(truth, predicted, strict=True) if p is not None]
    disagreements = sum(1 for t, p in judged if t != p)
    # Attempt-level: did the rep *reach* depth according to each?
    verdict_flipped = any(truth) != any(p for _, p in judged)
    return SweepResult(
        height=camera.height,
        pitch_deg=camera.pitch_deg,
        yaw_deg=camera.yaw_deg,
        frames=len(judged),
        disagreements=disagreements,
        verdict_flipped=verdict_flipped,
    )


def sweep(
    *,
    bottom_margin: float = 0.02,
    heights: tuple[float, ...] = (0.4, 0.7, 1.0, 1.3, 1.6, 1.9),
    pitches: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0),
    config: DepthConfig | None = None,
) -> list[SweepResult]:
    """Sweep camera height x pitch on a borderline-depth squat.

    A *borderline* rep is the honest test: 2 cm below parallel is a call a human
    referee would have to think about, and it is where geometric bias actually
    changes the verdict rather than being absorbed by slack.
    """
    trace = squat_trace(bottom_margin=bottom_margin)
    # Calibrate the landmark first. Without this the constant hip-joint-vs-crease
    # bias swamps everything and the sweep just measures that instead of geometry.
    if config is None:
        fraction, _ = calibrate_crease_fraction()
        config = DepthConfig(hip_crease_thigh_fraction=fraction)
    return [
        evaluate(trace, CameraPose(height=h, pitch_deg=p), config) for h in heights for p in pitches
    ]


#: The crease offset as a fraction of thigh length — the scale-invariant form,
#: and the only one that means anything while `z` is pixels.
TRUE_CREASE_FRACTION = TRUE_CREASE_OFFSET / THIGH


def calibrate_crease_fraction(
    *, margins: tuple[float, ...] = (-0.04, -0.01, 0.01, 0.04), candidates: int = 41
) -> tuple[float, float]:
    """Fit ``hip_crease_thigh_fraction`` by minimising disagreement with truth.

    `DepthConfig` ships ``0.0``, so the judge currently measures the hip *joint*
    rather than the *crease* the rulebook names — every call carries a constant
    bias toward "not deep enough". Fitting over reps on both sides of parallel
    stops the fit from simply learning one answer.

    Returns ``(best_fraction, disagreement_rate)``.
    """
    grid = np.linspace(0.0, 0.30, candidates)
    camera = CameraPose(height=1.0, pitch_deg=0.0)
    traces = [(squat_trace(bottom_margin=m), m) for m in margins]

    best = (0.0, 1.0)
    for fraction in grid:
        config = DepthConfig(hip_crease_thigh_fraction=float(fraction))
        wrong = total = 0
        for trace, _ in traces:
            truth = ground_truth_calls(trace)
            predicted = pipeline_calls(trace, camera, config)
            for t, p in zip(truth, predicted, strict=True):
                if p is None:
                    continue
                total += 1
                wrong += t != p
        rate = wrong / total if total else 1.0
        if rate < best[1]:
            best = (float(fraction), rate)
    return best


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass
class Envelope:
    """How far off-axis the camera may sit before the verdict changes."""

    margin_cm: float
    max_yaw_deg: int  # -1 when the call is wrong even square-on
    pitch_robust: bool
    height_robust: bool


def operating_envelope(
    margins: tuple[float, ...] = (0.005, 0.01, 0.02, 0.04, 0.08),
    config: DepthConfig | None = None,
) -> list[Envelope]:
    """Find the camera-placement tolerance at each true depth.

    Yaw is swept to failure; pitch and height are checked across their whole
    plausible range and reported as a yes/no, because they turn out to be
    second-order (see the module docstring and README).
    """
    if config is None:
        fraction, _ = calibrate_crease_fraction()
        config = DepthConfig(hip_crease_thigh_fraction=fraction)

    out: list[Envelope] = []
    for margin in margins:
        trace = squat_trace(bottom_margin=margin)
        truth = any(ground_truth_calls(trace))

        def reached(camera: CameraPose, trace: Pose3DSequence = trace) -> bool:
            calls = pipeline_calls(trace, camera, config)
            return any(bool(c) for c in calls if c is not None)

        max_yaw = -1
        for yaw in range(0, 91):
            if reached(CameraPose(height=1.0, yaw_deg=float(yaw))) == truth:
                max_yaw = yaw
            else:
                break
        out.append(
            Envelope(
                margin_cm=margin * 100,
                max_yaw_deg=max_yaw,
                pitch_robust=all(
                    reached(CameraPose(height=1.0, pitch_deg=float(p))) == truth
                    for p in range(-40, 41, 5)
                ),
                height_robust=all(
                    reached(CameraPose(height=h / 10)) == truth for h in range(3, 23)
                ),
            )
        )
    return out


def format_envelope(rows: list[Envelope]) -> str:
    lines = [
        "",
        "Operating envelope — how far the camera may stray before the call changes",
        "",
        "  true depth    max yaw off-axis    pitch +-40deg    height 0.3-2.2m",
        "  " + "-" * 68,
    ]
    for r in rows:
        yaw = f"{r.max_yaw_deg:>3}째" if r.max_yaw_deg >= 0 else " n/a"
        yaw = yaw.replace("째", "°")
        note = "" if r.max_yaw_deg >= 0 else "   (wrong even square-on)"
        lines.append(
            f"  {r.margin_cm:>5.1f} cm      {yaw:>12}    "
            f"{'ok' if r.pitch_robust else 'FAILS':>13}    "
            f"{'ok' if r.height_robust else 'FAILS':>15}{note}"
        )
    lines += [
        "",
        "  Read: the shallower the rep, the squarer the camera must be. A clearly",
        "  deep squat survives large camera error; a borderline one does not.",
    ]
    return "\n".join(lines)


def format_sweep(results: list[SweepResult]) -> str:
    heights = sorted({r.height for r in results})
    pitches = sorted({r.pitch_deg for r in results})
    lookup = {(r.height, r.pitch_deg): r for r in results}

    lines = ["", "Frame-level disagreement with 3D ground truth (%)", ""]
    lines.append("  height \\ pitch " + "".join(f"{p:>9.0f}°" for p in pitches))
    for h in heights:
        row = "".join(f"{lookup[(h, p)].flip_rate * 100:>9.0f} " for p in pitches)
        lines.append(f"  {h:>13.1f}m {row}")

    flipped = [r for r in results if r.verdict_flipped]
    lines += [
        "",
        f"placements tested      : {len(results)}",
        f"verdict flipped        : {len(flipped)} / {len(results)}",
    ]
    clean = [r for r in results if r.flip_rate == 0.0]
    if clean:
        hs = sorted({r.height for r in clean})
        ps = sorted({r.pitch_deg for r in clean})
        lines.append(f"exact-agreement region : height {hs[0]:.1f}–{hs[-1]:.1f} m, ")
        lines.append(f"                         pitch {ps[0]:+.0f}° to {ps[-1]:+.0f}°")
    else:
        lines.append("exact-agreement region : none — every placement disagreed somewhere")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera-geometry sensitivity sweep")
    parser.add_argument("--json", action="store_true", help="emit results as JSON")
    parser.add_argument(
        "--calibrate", action="store_true", help="fit depth.hip_crease_offset and exit"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.02,
        help="true depth at the bottom, metres below parallel (default: borderline 0.02)",
    )
    args = parser.parse_args()

    if args.calibrate:
        fraction, rate = calibrate_crease_fraction()
        print("\nhip_crease_thigh_fraction calibration")
        print("=" * 52)
        print("  shipped default : 0.000  (judges the hip JOINT, not the crease)")
        print(f"  best fit        : {fraction:.3f}  of thigh length")
        print(f"  true value      : {TRUE_CREASE_FRACTION:.3f}")
        print(f"  residual error  : {rate * 100:.1f}% of frames")
        return

    fraction, residual = calibrate_crease_fraction()
    config = DepthConfig(hip_crease_thigh_fraction=fraction)
    rows = operating_envelope(config=config)
    results = sweep(bottom_margin=args.margin, config=config)

    if args.json:
        print(
            json.dumps(
                {
                    "hip_crease_thigh_fraction": fraction,
                    "calibration_residual": residual,
                    "envelope": [asdict(r) for r in rows],
                    "height_pitch_sweep": [asdict(r) | {"flip_rate": r.flip_rate} for r in results],
                },
                indent=2,
            )
        )
        return

    print("\nWhite Lights — camera-geometry sensitivity")
    print("=" * 72)
    print(f"  landmark calibrated first : hip_crease_thigh_fraction={fraction:.3f}")
    print(f"  residual after calibration: {residual * 100:.1f}%")
    print(format_envelope(rows))
    print(f"\n  Height x pitch grid at {args.margin * 100:+.0f} cm true depth:")
    print(format_sweep(results))
    print()


if __name__ == "__main__":
    main()
