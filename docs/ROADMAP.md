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
- **Tooling.** 99-test suite, CI (ruff + pytest), Pages demo, Dockerfile.

## Next

- **Free-reps judges for bench & deadlift.** They currently reuse their
  competition judge in both modes; training mode needs a free-rep counter.
- **Threshold tuning + validation.** The numbers in every tracker are placeholders.
  Collect labelled clips and run `eval/validate.py` to measure v2 vs the v1 91%
  baseline, then tune per federation profile.
- **Deploy the real backend** (Hugging Face Spaces — see [DEPLOY.md](DEPLOY.md)).

## Later / deferred (need new input signals)

- **Real multi-camera triangulation** in `fusion.py` — the biggest accuracy
  unlock; would replace the single-view "3D" lift and tighten borderline lockout
  calls toward referee-grade.
- **Hitching / ramping** detection (deadlift) — repeated knee re-bends.
- **Bar-on-thighs**, buttocks-off-bench, grip — need a bar/contact signal.
- **ONNX export** to drop the torch dependency (~2 GB → ~200 MB, faster CPU
  inference, fits any free host).
