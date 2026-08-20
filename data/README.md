# Validation data

Layout for running the validation harness (`eval/validate.py`):

```
data/
├── labels.csv          # your labels (git-ignored; copy from the example)
├── labels.example.csv  # committed template
├── clips/               # the video clips (git-ignored)
│   ├── clip_0001.mp4
│   ├── clip_0002.mp4
│   └── ...
└── traces/              # extracted keypoint traces (git-ignored) — see below
    ├── clip_0001.trace.json
    └── ...
```

## labels.csv

One row per clip. `filename` is relative to the clips directory; `true_call` is
the ground-truth referee call — one of `GOOD`, `NO_LIFT`, `UNCERTAIN` (the
`Verdict` values). See `labels.example.csv`.

```csv
filename,true_call
clip_0001.mp4,GOOD
clip_0002.mp4,NO_LIFT
```

## Running

Two ways to judge the clips, at different points in the pipeline — pick
`--traces-dir` unless you have a specific reason not to.

**Preferred: extract traces once, then validate torch-free.** Pose
estimation is the only stage that needs the heavy `cv` extra (torch +
ultralytics); everything after it — smoothing, fusion, depth, rep
segmentation — is pure Python. Splitting them means validation itself runs
without torch installed, which is what lets `eval/validate.py --traces-dir`
run in CI (see `eval/traces.py`) and lets a trace be committed as a fixture
without redistributing footage.

```bash
pip install -e ".[cv]"        # only needed for this extraction step
python -m eval.traces extract --clips-dir data/clips --out data/traces

python -m eval.validate --traces-dir data/traces --labels data/labels.csv
```

**Direct: judge the clips themselves, every run.** Simpler for a one-off
check, but re-runs pose estimation on every clip every time and always needs
the `cv` extra installed.

```bash
pip install -e ".[cv]"
python -m eval.validate --clips-dir data/clips --labels data/labels.csv
```

Both accept `--json` for a machine-readable report. Either way, each clip is
judged, reduced to a single call (the longest-spanning rep — see
`reduce_reps_to_call`), and compared to its label. The report gives overall
agreement, per-class recall, a confusion matrix, and latency stats.

## Notes

- Clips, traces, and your real `labels.csv` are all git-ignored (see the repo
  `.gitignore`); only the example label file and this README are committed.
  Keep the actual video data out of git — a trace is coordinates, not a
  person, so it's the one of the three that's safe to commit as a fixture if
  you want one in CI.
- Clips are assumed to be a single competition attempt (one rep). Multi-rep
  clips collapse to the primary (longest) rep.
