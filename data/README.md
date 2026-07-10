# Validation data

Layout for running the validation harness (`eval/validate.py`):

```
data/
├── labels.csv          # your labels (git-ignored; copy from the example)
├── labels.example.csv  # committed template
└── clips/              # the video clips (git-ignored)
    ├── clip_0001.mp4
    ├── clip_0002.mp4
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

Requires the pose runtime (`pip install -e ".[cv]"`).

```bash
python -m eval.validate --clips-dir data/clips --labels data/labels.csv
python -m eval.validate --clips-dir data/clips --labels data/labels.csv --json
```

Each clip is judged, reduced to a single call (the longest-spanning rep — see
`reduce_reps_to_call`), and compared to its label. The report gives overall
agreement, per-class recall, a confusion matrix, and latency stats.

## Notes

- Clips and your real `labels.csv` are git-ignored (see the repo `.gitignore`);
  only the example label file and this README are committed. Keep the actual
  video data out of git.
- Clips are assumed to be a single competition attempt (one rep). Multi-rep
  clips collapse to the primary (longest) rep.
