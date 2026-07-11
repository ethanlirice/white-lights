# White Lights

Real-time computer-vision powerlifting judge. Open the live web app, point your
webcam at the platform, and White Lights calls each lift — **GOOD**, **NO_LIFT**,
or **UNCERTAIN** — against the federation rules, in real time, with the specific
fault(s) flagged.

> **▶︎ Live UI demo (in your browser, nothing to install):**
> **https://ethanlirice.github.io/white-lights/**
>
> ⚠️ **This hosted link is a UI demo only — it does _not_ judge real movement.**
> GitHub Pages can't run the pose model, so the page falls back to a built-in
> **simulator** that plays canned data to show off the interface. The actual
> judging (real webcam → YOLO pose → live verdicts) only runs when you start the
> **backend locally** — see [Run](#run) below.

**Two modes:**

- 🏋️ **Training** — free reps: pick a weight, start a set, get a live
  GOOD / NO_LIFT call on every rep, and log your set history (stored in the
  browser, exportable to JSON).
- 🏆 **Competition** — the computer plays referee: it waits for a still,
  locked-out setup, issues the **SQUAT** / **RACK** commands itself, and judges
  the single attempt on the full rulebook — depth, downward movement,
  early-descent / early-rack, and lockout — with a "three white lights" reveal.

Under the hood: real-time pose → depth / lockout / motion analysis → a
referee-command state machine, streamed over a browser ↔ FastAPI WebSocket.
Squat is fully implemented; **bench press and deadlift are in progress**. When
a call is genuinely borderline it returns **UNCERTAIN** ("too close to call")
rather than forcing a guess. See [DESIGN.md](DESIGN.md).

## What it judges

A legal squat is a "command sandwich": the lifter sets up erect and locked,
receives the **"Squat!"** command, descends until the **hip crease is below the
top of the knee**, ascends without any downward movement, locks out, and returns
the bar on the **"Rack!"** command. White Lights targets these failure modes,
implemented incrementally: insufficient depth (primary), downward movement on the
ascent, command-timing violations, and postural/foot faults.

## Architecture

```
                 ┌──────────────┐   one video per camera view
   video(s) ───► │  pose.py     │   YOLO11-pose → per-frame 2D keypoints
                 │  (done)      │   + confidences, per camera
                 └──────┬───────┘
                        │  PoseSequence  (2D, per camera)
                 ┌──────▼───────┐
                 │ smoothing.py │   gap-fill + confidence gate       [done*]
                 └──────┬───────┘
                        │  PoseSequence  (2D, cleaned)
                 ┌──────▼───────┐
                 │  fusion.py   │   1 view → 3D lift                 [done†]
                 └──────┬───────┘
                        │  Pose3DSequence  (world coords, +z up)
                 ┌──────▼───────┐
                 │  depth.py    │   per-frame below-parallel + gating [done]
                 └──────┬───────┘
                        │  list[DepthFrameResult]
                 ┌──────▼───────┐
                 │   reps.py    │   segment → one verdict/rep        [done‡]
                 └──────┬───────┘
                        │  list[RepVerdict]
                 ┌──────▼───────┐
                 │ api/main.py  │   FastAPI: POST /judge, GET /
                 └──────────────┘

  Orchestrated by whitelights/pipeline.py. Shared types in whitelights/types.py.

  * smoothing: gap-fill done; genuine jitter reduction (One-Euro/Kalman) deferred.
  † fusion: single-camera lift done; multi-view triangulation raises (v2.3+).
  ‡ reps: depth verdict done; downward-movement (v2.1) + command-timing (v2.2) deferred.
```

- **`whitelights/`** — core package (`pose`, `smoothing`, `fusion`, `depth`,
  `reps`, `pipeline`, `types`).
- **`api/`** — FastAPI app (`POST /judge`, `GET /`).
- **`web/`** — single static HTML+JS upload page.
- **`tests/`** — pytest suite with synthetic keypoint fixtures.
- **`eval/`** — validation harness (`validate.py`).

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate

# Full local install (pose model + API + dev tools):
pip install -e ".[cv,api,dev]"

# Lighter installs also work:
#   pip install -e ".[api,dev]"   # everything except the pose model (torch/opencv)
#   pip install -e ".[dev]"       # types + tests only
```

Dependencies are split into extras so tests and CI stay fast: `cv`
(ultralytics + opencv, pulls torch), `api` (fastapi + uvicorn), `dev` (pytest +
ruff). The YOLO11-pose weights (`yolo11n-pose.pt`) auto-download on first run.

## Run

**This is the real, working judge** — the hosted GitHub Pages link is a
simulated UI demo only. Run the backend locally, then open `/live` in your
browser for actual webcam judging (needs the `cv` extra so the pose model runs).

```bash
# API + web UI  →  http://127.0.0.1:8000
uvicorn api.main:app
#   /       upload a clip for batch judging
#   /live   live webcam judge — the real thing (real-time; needs the cv extra)
# (avoid `--reload` here: it watches the whole .venv and thrashes on torch's files)

# Live webcam judge in a terminal window (OpenCV), instead of the browser:
python -m whitelights.live --camera 1     # try 1/2 to pick the built-in camera

# Tests (deferred-feature contracts xfail by design)
pytest

# Lint
ruff check .

# Validation harness (once labelled clips exist)
python -m eval.validate --clips-dir data/labelled --labels data/labels.csv
```

A **single-camera** upload now runs the full v2.0 pipeline and returns per-rep
verdicts as JSON (requires the `cv` extra so the pose model can run). A
**multi-camera** upload returns **HTTP 501** — real triangulation is not built
yet (v2.3+); without the `cv` extra installed, `/judge` returns **HTTP 503** with
an install hint.

## Metrics

v1 was validated at **91% agreement on 5,000+ reps under competition
conditions**. v2 is a rebuild and **revalidation is in progress** — no v2
performance numbers are claimed yet. The `eval/` harness is how v2 will be
measured (agreement %, per-class breakdown, latency) once labelled clips and the
core logic are in place.
