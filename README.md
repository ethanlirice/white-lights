# White Lights

Real-time computer-vision squat-depth judge for powerlifting. Point cameras at a
lifter, and White Lights segments the video into rep attempts and calls each one
— **GOOD**, **NO_LIFT**, or **UNCERTAIN** — against the federation depth rule
(hip crease below the top of the knee), with the specific fault(s) flagged.

> **Status:** v2 is a ground-up rebuild. The scaffolding, pose estimation, HTTP
> surface, and test/eval harnesses are in place; the core CV judging logic
> (smoothing, 3D fusion, depth, rep segmentation) is being reimplemented behind
> stable interfaces. See [DESIGN.md](DESIGN.md).

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
                 │  (WORKING)   │   + confidences, per camera
                 └──────┬───────┘
                        │  PoseSequence  (2D, per camera)
                 ┌──────▼───────┐
                 │ smoothing.py │   de-jitter + gap-fill each track   [stub]
                 └──────┬───────┘
                        │  PoseSequence  (2D, cleaned)
                 ┌──────▼───────┐
                 │  fusion.py   │   multi-view triangulation → 3D     [stub]
                 └──────┬───────┘
                        │  Pose3DSequence  (world coords, +z up)
                 ┌──────▼───────┐
                 │  depth.py    │   per-frame below-parallel + gating [stub]
                 └──────┬───────┘
                        │  list[DepthFrameResult]
                 ┌──────▼───────┐
                 │   reps.py    │   state machine → one verdict/rep   [stub]
                 └──────┬───────┘
                        │  list[RepVerdict]
                 ┌──────▼───────┐
                 │ api/main.py  │   FastAPI: POST /judge, GET /
                 └──────────────┘

  Orchestrated by whitelights/pipeline.py. Shared types in whitelights/types.py.
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

```bash
# API + web UI  →  http://127.0.0.1:8000
uvicorn api.main:app --reload

# Tests (pose helpers pass; stub contracts xfail by design)
pytest

# Lint
ruff check .

# Validation harness (once labelled clips exist)
python -m eval.validate --clips-dir data/labelled --labels data/labels.csv
```

Until the CV core is implemented, `POST /judge` runs pose estimation and then
returns **HTTP 501** with a clear "core logic not implemented" message — the
pipeline is wired end to end.

## Metrics

v1 was validated at **91% agreement on 5,000+ reps under competition
conditions**. v2 is a rebuild and **revalidation is in progress** — no v2
performance numbers are claimed yet. The `eval/` harness is how v2 will be
measured (agreement %, per-class breakdown, latency) once labelled clips and the
core logic are in place.
