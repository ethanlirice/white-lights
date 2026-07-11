# Architecture

White Lights judges powerlifting attempts from a single camera in real time. This
document covers the data flow, the module map, and the engineering decisions worth
knowing. For the *why* behind individual choices see [DESIGN.md](DESIGN.md).

## Two execution paths, one rulebook

The federation rules are implemented once and run in two modes:

| | **Batch** | **Online (live)** |
|---|---|---|
| Entry | `POST /judge` (video file) | `WS /ws/live` (webcam stream) |
| Processing | whole clip at once | one frame at a time, **causal** |
| Segmentation | global min/max over the clip | a running state machine |
| Orchestrator | `whitelights/pipeline.py` | `whitelights/live.py` trackers |

Both consume the same typed data model (`whitelights/types.py`) and the same
per-frame primitives (depth, posture). The online path is what powers the live
web app; the batch path is used for offline validation.

## Live data flow

```
Browser                          Server (async FastAPI)
───────                          ─────────────────────
webcam frame
  ↓ downscale → 480px JPEG
  ↓ WebSocket send  ───────────►  receive_bytes
                                   ↓ run_in_executor (threadpool)
                                   │   cv2.imdecode
                                   │   PoseEstimator (YOLO11-pose)  → 2D keypoints
                                   │   StreamingKeypointSmoother    → One-Euro filter
                                   │   fusion: single-view 3D lift
                                   │   depth / posture per frame
                                   │   tracker.update(frame, depth) → LiveStatus
                                   ↓ live_payload(...)  → JSON
  canvas overlay  ◄─────────────  send_json
  + reasoning panel
```

**Backpressure:** the client sends the next frame only after the previous
response arrives (ping-pong), so there's exactly one frame in flight — no
unbounded queue, and the stream self-throttles to whatever the server can keep up
with.

**Why a threadpool:** YOLO inference is synchronous and CPU-bound. Running it
directly in the async handler would block the event loop for the whole socket;
`loop.run_in_executor(...)` moves it to a worker thread so the connection stays
responsive.

## The online judge: a state machine per lift × mode

Each lift+mode selects a tracker via `whitelights/judges.py::tracker_for(lift, mode)`:

| lift | mode | tracker | commands |
|---|---|---|---|
| squat | training | `OnlineRepTracker` | — (free reps) |
| squat | competition | `CompetitionTracker` | SQUAT → RACK |
| bench | competition | `BenchTracker` | START → PRESS → RACK |
| deadlift | competition | `DeadliftTracker` | DOWN |

Every tracker exposes the same interface — `update(frame, depth) -> LiveStatus`
and `reset()` — so `LiveJudge` and the WebSocket handler are lift-agnostic. The
client swaps trackers by sending `{cmd:"start", lift, mode}`.

Each tracker is a small causal state machine over the lifter's primary signal
(hip height for squat, bar/wrist height for bench & deadlift) plus per-frame
depth/lockout evidence. They share a common toolbox:

- **Stillness** — hip/bar velocity ≈ 0, used to detect a stable setup/lockout.
- **Joint-angle lockout** — `posture.py` computes hip-knee-ankle (knees) and
  shoulder-elbow-wrist (elbows) angles; a lift is "locked" when the governing
  joint is near-straight, graded against a **calibrated per-lifter** reference.
- **Downward-movement** — re-descent past a running peak (scale-invariant).
- **Command sandwich** — issue the start command on a still, locked setup; judge;
  issue the end command on a still, locked finish. Moving before/after a command
  is a fault (early-descent / early-rack / …).

Output is one `RepVerdict` per rep/attempt: `GOOD | NO_LIFT | UNCERTAIN`, a
confidence, the list of `Fault`s, and the depth margin.

## Module map (`whitelights/`)

| module | responsibility |
|---|---|
| `types.py` | shared Pydantic data model: keypoints, sequences, `Verdict`, `Fault`, `RepVerdict` |
| `pose.py` | YOLO11-pose wrapper → typed per-frame 2D keypoints (lazy torch import) |
| `smoothing.py` | batch gap-fill / confidence gating; `filters.py` = online One-Euro |
| `fusion.py` | 2D → 3D (single-view lift now; multi-camera triangulation later) |
| `depth.py` | per-frame "hip crease below knee?" with confidence gating |
| `posture.py` | joint angles, knee/elbow lockout, foot-movement |
| `reps.py` | **batch** rep segmentation + verdict state machine |
| `live.py` | **online** squat trackers + `LiveJudge` glue + OpenCV demo |
| `bench.py`, `deadlift.py` | online bench/deadlift judges |
| `judges.py` | `(lift, mode) → tracker` factory |
| `pipeline.py` | batch orchestrator (pose → smoothing → fusion → depth → reps) |

## Engineering notes

- **Scale invariance.** Single-camera "3D" has a faked depth axis, so absolute
  units are unreliable. Thresholds are expressed as fractions of a body reference
  (thigh / torso / arm length) and key calls are *sign* tests, so they hold in
  pixel or metric space.
- **Uncertainty as a feature.** Confidence-gating and a first-class `UNCERTAIN`
  verdict mean borderline calls are surfaced, not guessed — important because a
  pose estimator's "locked out" is noisy and rarely a clean 180°.
- **Deterministic tests for stochastic input.** The 99-test suite drives the
  trackers with hand-built synthetic keypoint traces (known good / high /
  double-bounce / early-command …), so judging logic is tested exactly without a
  camera or model.
- **Honest deferrals.** Multi-camera triangulation, hitching detection, and
  bar-on-thighs need signals the current single-camera pose doesn't expose; they
  are documented stubs rather than faked.
