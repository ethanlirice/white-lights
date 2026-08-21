"""Terminal webcam demo — OpenCV window with the live overlay.

The browser UI (`web/live.html` over `/ws/live`) is the real front end; this is
the no-server path for checking the judge against a camera directly. It lives
apart from `live.py` so that module stays pure judging logic: drawing code, the
capture loop and argparse have no business sitting next to the state machines.

Run it::

    pip install -e ".[cv]"
    python -m whitelights.cli                            # squat, training
    python -m whitelights.cli --lift bench --mode competition
    python -m whitelights.cli --camera 1                  # pick a different camera (see below)

macOS note: if the feed opens on your iPhone, that is Continuity Camera grabbing
index 0. Try ``--camera 1`` / ``--camera 2`` for the built-in FaceTime camera,
or turn Continuity Camera off on the phone. Press ESC or q to quit.
"""

from __future__ import annotations

import argparse

from .judges import LIFTS, MODES, tracker_for
from .live import LiveJudge
from .pose import DEFAULT_MODEL, PoseEstimator
from .types import FrameKeypoints, LiveStatus, Verdict

_SIDES = ("left", "right")

_SKELETON = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def _light_color(below: bool | None) -> tuple[int, int, int]:
    if below is True:
        return (0, 200, 0)
    if below is False:
        return (0, 0, 220)
    return (128, 128, 128)


def _draw_overlay(
    img, frame2d: FrameKeypoints, status: LiveStatus, conf: float, *, judges_depth: bool
) -> None:
    import cv2

    h, w = img.shape[:2]

    # Knee line (depth target): only meaningful for a lift that judges depth —
    # bench/deadlift have no rule about the knee line at all, so drawing it
    # unconditionally would show a landmark the judge underneath isn't using.
    if judges_depth:
        knees = [frame2d.get(f"{s}_knee") for s in _SIDES]
        knee_ys = [int(k.y) for k in knees if k and k.confidence >= conf]
        if knee_ys:
            ky = min(knee_ys)
            cv2.line(img, (0, ky), (w, ky), (0, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(
                img,
                "knee line",
                (w - 130, ky - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 220),
                1,
                cv2.LINE_AA,
            )

    # Skeleton (thick) + joints.
    for a, b in _SKELETON:
        ka, kb = frame2d.get(a), frame2d.get(b)
        if ka and kb and ka.confidence >= conf and kb.confidence >= conf:
            cv2.line(
                img, (int(ka.x), int(ka.y)), (int(kb.x), int(kb.y)), (255, 255, 255), 3, cv2.LINE_AA
            )
    for kp in frame2d.keypoints.values():
        if kp.confidence >= conf:
            cv2.circle(img, (int(kp.x), int(kp.y)), 6, (0, 200, 255), -1, cv2.LINE_AA)

    # Top banner: light + state + note. `checkpoint` — not `below_parallel` — is
    # the generic "key checkpoint met" field (squat: below parallel; bench: bar
    # on chest; deadlift: locked out); `below_parallel` is a squat-specific
    # holdover that bench/deadlift's own LiveStatus always sets to None, so
    # keying the light off it would leave it permanently grey for those lifts.
    cv2.rectangle(img, (0, 0), (w, 96), (30, 30, 30), -1)
    cv2.circle(img, (46, 48), 26, _light_color(status.checkpoint), -1, cv2.LINE_AA)
    cv2.putText(
        img, status.state, (88, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        img, status.note, (88, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA
    )
    cv2.putText(
        img,
        f"reps: {status.rep_count}",
        (w - 200, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Progress bar (how far into a rep, and whether the checkpoint is met).
    if status.descent_fraction is not None:
        bx, by, bw, bh = 88, 108, 260, 18
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (80, 80, 80), 1)
        fill = int(min(1.0, status.descent_fraction) * bw)
        cv2.rectangle(img, (bx, by), (bx + fill, by + bh), _light_color(status.checkpoint), -1)

    # Last verdict.
    if status.last_verdict is not None:
        v = status.last_verdict
        label = v.verdict.value + (
            "  (" + ", ".join(f.value for f in v.faults) + ")" if v.faults else ""
        )
        color = _light_color(True if v.verdict == Verdict.GOOD else None)
        cv2.putText(
            img,
            f"last rep: {label}",
            (16, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="White Lights — live webcam judge")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (try 1/2 for built-in)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO11-pose weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--lift", choices=LIFTS, default="squat")
    parser.add_argument("--mode", choices=MODES, default="training")
    args = parser.parse_args()

    import cv2

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera} (try --camera 1 or 2)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Same factory the WebSocket handler uses (see api/main.py's `_handle_control`)
    # — the CLI picking its own tracker independently is exactly the kind of
    # second implementation that drifts from the real one.
    tracker = tracker_for(args.lift, args.mode)
    estimator = PoseEstimator(model_path=args.model, conf=args.conf)
    judge = LiveJudge(estimator, fps=fps, tracker=tracker)
    print(f"White Lights live — {args.lift} / {args.mode} — press ESC or q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame2d, _depth, status = judge.process_frame(frame)
            _draw_overlay(frame, frame2d, status, args.conf, judges_depth=tracker.judges_depth)
            if status.rep_completed and status.last_verdict is not None:
                v = status.last_verdict
                print(f"rep {v.rep_index}: {v.verdict.value} {[f.value for f in v.faults]}")
            cv2.imshow("White Lights — live", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
