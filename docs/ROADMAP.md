# Roadmap

White Lights v2 is a ground-up rebuild of a lost v1 (which judged at 91%
agreement on 5,000+ reps under competition conditions). Progress has been
deliberately incremental — each stage shipped with tests behind stable
interfaces.

## Done

- **v2.0 — depth-only pipeline.** pose → smoothing → single-view 3D fusion →
  depth → rep segmentation, end to end, for single-camera clips.
- **v2.1 — downward movement.** Re-descent / double-bounce detection.
- **v2.2 — command timing.** Referee `START`/`RACK` wired in; EARLY_DESCENT.
- **v2.3 — postural faults.** Knee/elbow lockout, foot movement.
- **Live web app.** Browser ↔ WebSocket ↔ FastAPI real-time judge with a One-Euro
  online rep tracker; robust against phantom reps on noisy webcam pose.
- **Competition mode.** The computer plays referee — auto-issued commands,
  three-white-lights reveal, early-descent / early-rack judging.
- **Multi-lift.** Bench and deadlift judges (referee-command state machines) + a
  lift selector in the UI. Generic `checkpoint` / `command` / `progress` contract.
- **UI redesign.** Multi-page app (live / landing / history / stats), per-lift
  checkpoint light + command tracker, in-browser history.
- **Tooling.** 176-test suite including a wire-contract layer, CI (ruff + mypy +
  pytest), Pages demo, Dockerfile.
- **Free-reps judges for bench & deadlift.** One `FreeRepTracker` serves both,
  working in travel-from-rest so the two directions share a code path.
- **Shared tracker machinery.** `tracking.py` holds the signal/scale/stillness/
  hold-timer/verdict logic all five judges were duplicating.
- **Camera-geometry envelope.** `eval/geometry.py` projects known 3D traces
  through a virtual camera to measure placement sensitivity — no footage needed.
  Found the landmark (hip joint vs crease) mattered more than the lens, and that
  yaw is the only camera axis that moves the call.
- **Production real-time path.** Bounded pool of pre-warmed models, explicit load
  shedding, per-stage latency at `/metrics`.
- **Keypoint traces.** `eval/traces.py` saves pose output as JSON so validation
  runs torch-free in CI, and fixtures can be committed without redistributing
  footage.
- **Frontend logic is tested, not just typed carefully.** `web/lib/*.mjs` holds
  the pure logic three pages used to carry inline — pose synthesis, history
  aggregation, chart math — imported as native ES modules (no bundler) and
  covered by 45 Vitest cases, wired into CI as its own job. Writing the first
  of these immediately found a real bug no amount of careful reading had:
  the demo squat's checkpoint light and its depth-overlay lines were computed
  independently and disagreed with each other for the entire back half of
  every rep.

## Next

- **Threshold tuning + validation.** The numbers in every tracker are placeholders.
  Collect labelled clips, extract traces, and run `eval/validate.py` to measure v2
  against the v1 91% baseline, then tune per federation profile.
- **Ship `hip_crease_thigh_fraction`.** The geometry harness fits ~0.29 of thigh
  against a model whose true value is 0.14; the gap is real (the thigh
  foreshortens badly at the bottom of a squat, making it a shaky reference there).
  It defaults to 0.0 until labelled clips settle it.

## Later / deferred (need new input signals)

- **Real multi-camera triangulation** in `fusion.py` — the biggest accuracy
  unlock; would replace the single-view "3D" lift and tighten borderline lockout
  calls toward referee-grade.
- **Hitching / ramping** detection (deadlift) — repeated knee re-bends.
- **Bar-on-thighs**, buttocks-off-bench, grip — need a bar/contact signal.
- **ONNX export** to drop the torch dependency (~2 GB → ~200 MB, faster CPU
  inference, fits any free host).
