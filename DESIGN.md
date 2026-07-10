# Design notes

Rationale for each decision in the v2 scaffold, for review. The brief was
scaffolding only: `pose.py` is fully implemented; `smoothing`, `fusion`,
`depth`, and `reps` are stubs with firm contracts and failing/xfail tests.

## Layout & packaging

- **`whitelights/` core, `api/` + `web/` at the edges, `eval/` separate.** The
  judging logic must be usable without FastAPI (from the eval harness, a
  notebook, or a future real-time capture loop), so HTTP is a thin adapter over
  `pipeline.judge_video`, not the other way around.
- **`pyproject.toml` with setuptools.** Standard, no plugin needed for a pure-
  Python package.
- **Dependencies split into extras (`cv`, `api`, `dev`).** The brief lists
  ultralytics/opencv/fastapi/etc. as deps; keeping them all as hard core deps
  would drag torch into every `pip install` and every CI run (slow, occasionally
  flaky). Core deps are just `pydantic` + `numpy`; the heavy stacks are opt-in.
  Crucially, `pose.py` imports ultralytics/opencv **lazily**, so its pure
  helpers unit-test without the `cv` extra, and CI runs lint+tests with no torch.

## The shared type model (`types.py`)

- **Pydantic everywhere.** You asked for typed dataclasses/Pydantic models; one
  Pydantic vocabulary means the same objects flow through the pipeline *and*
  serialise as the API response with no mapping layer.
- **2D vs 3D split (`Keypoint2D`/`PoseSequence` vs `Keypoint3D`/`Pose3DSequence`).**
  Forced by the multi-camera choice (below). Pose output is per-camera 2D;
  everything from fusion onward is 3D world coordinates. Encoding this in the
  types keeps a camera view from being silently treated as 3D.
- **Coordinate conventions pinned in docstrings** (2D: +y down, image pixels;
  3D: +z up, world units). Depth logic is unforgiving about sign, so this is
  stated once, centrally.

## Verdict representation — three-state + fault flags

- **`Verdict` = GOOD / NO_LIFT / UNCERTAIN**, plus `confidence` and a **list of
  `Fault`s**, plus `depth_margin` as the primary evidence.
- A rep can violate several rules at once (e.g. high *and* double-bounced), so a
  single "reason" enum would lose information — hence a fault *list*.
- **UNCERTAIN earns its place specifically because of the multi-camera choice:**
  triangulation/occlusion can leave a frame with no trustworthy depth reading,
  and confidence-gating needs a channel to say "not enough signal" rather than
  forcing a wrong call. `depth.py` returns `is_below_parallel=None` when gated;
  `reps.py` turns a gated bottom into UNCERTAIN.

## Camera model — multi-camera / 3D, with a fusion seam

- You chose multi-camera/3D, so the interfaces carry a camera axis: `pose.py`
  tags each `PoseSequence` with a `camera_id`, and `api`/`pipeline` accept a
  **list** of videos (one per view).
- This makes a **2D→3D fusion step unavoidable**, so I added `whitelights/fusion.py`
  (`reconstruct_3d`) as its own stub rather than smuggling triangulation into
  depth. `depth.py` then judges pure 3D geometry and stays camera-agnostic. This
  is the one module beyond the four you named — flagged here for your call.
- **Single-camera pragmatism:** the fusion contract documents a degenerate
  fallback (one view lifted to z=0) so the depth-only v2.0 milestone can proceed
  before a real multi-cam rig exists. The default model stays `yolo11n-pose`.

## `pose.py` (implemented)

- **Lazy model loading** — constructing a `PoseEstimator` is cheap and import-
  safe; weights load on first inference.
- **Pure helpers separated from the ultralytics glue** (`select_subject_index`,
  `frame_from_person`, `result_to_frame`). This is what makes the module
  testable without torch and keeps the ultralytics-specific parsing in one
  duck-typed spot.
- **Subject selection is explicit** (`"largest"` box by default, `"confidence"`
  alternative). A meet platform can have spotters/loaders in frame; picking the
  lifter deterministically matters, so it's a named strategy, not a hidden
  `[0]`.
- **Contiguous time base** — undetected frames become `empty_frame(...)` rather
  than gaps in the list, so smoothing/segmentation can assume one entry per
  frame index.

## The stubs — contracts, not logic

Each stub carries a docstring contract (inputs/outputs/invariants), typed
signatures, a `*Config` model for tunables, and `raise NotImplementedError`
with `TODO(ethan)`. Design choices worth noting:

- **`smoothing` is a filter, not a resampler** — same length/time base in and
  out — so downstream indexing stays valid. Contract asks for confidence-aware,
  causal-friendly, bounded-gap behaviour.
- **`depth` is per-frame and gates rather than guesses.** Segmentation lives in
  `reps`, keeping "is this frame below parallel?" separate from "where are the
  reps and what's the verdict?".
- **`reps` is a command-aware state machine.** States
  `WAITING_FOR_START → DESCENDING → BOTTOM → ASCENDING → LOCKED_OUT → RACKED`
  map directly onto the command sandwich. It takes the 3D poses (motion signal),
  the per-frame depth results (pass/fail evidence), **and optional referee
  commands** — the commands are in the signature now (ignored until v2.2) so
  adding command-timing faults later needs no interface change.

## Pipeline & API

- **`pipeline.judge_video` orchestrates; the API is a thin adapter.** Pose runs
  for real, then the first stub raises `NotImplementedError`, which the API maps
  to **HTTP 501** — semantically exact ("Not Implemented") and an honest
  end-to-end wiring rather than a fake success.
- **`estimator` is injectable** into the pipeline so tests exercise the wiring
  with a fake estimator (no weights, no torch).
- **No DB, no auth, synchronous, temp-file cleanup.** Per the brief: minimal.
  Uploads are written to a temp dir and deleted after judging.

## Tests

- **Pose helpers have real passing tests** (fake ultralytics `Results`).
- **Stub contracts are `strict` xfail** (`raises=NotImplementedError`). They
  encode the *expected* behaviour on synthetic fixtures; today they xfail on the
  `NotImplementedError`, and when you implement a stub correctly the strict xfail
  flips to a failure (XPASS) — a built-in reminder to remove the marker.
- **Synthetic fixtures** (`conftest.py`) build known good / high / double-bounce
  3D traces and a noisy 2D track with a gap, plus a ground-truth depth builder
  so the `reps` tests isolate the state machine from the depth stub.

## Roadmap (incremental feature order)

1. **v2.0 — depth only.** smoothing + fusion (single-cam fallback ok) + depth +
   a depth-only rep segmenter. Matches the "main basis": below-parallel calls.
2. **v2.1 — downward movement.** Re-descent detection on the ascent; reuses the
   same hip trajectory.
3. **v2.2 — command timing.** Wire referee `START`/`RACK` timestamps (manual,
   then audio) into EARLY_DESCENT / EARLY_RACK. Interface already present.
4. **v2.3+ — postural faults.** Lockout, foot movement, bar-on-thighs as added
   fault flags, plus real multi-camera calibration/triangulation in `fusion`.
```
