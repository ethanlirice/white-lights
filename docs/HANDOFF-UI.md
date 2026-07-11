# HANDOFF — Multi-lift UI + new pages

## web/live.html (modified in place)

### New element IDs / hooks
- `#liftSquatBtn` / `#liftBenchBtn` / `#liftDeadliftBtn` — lift selector buttons inside the existing `#modeSwitch` pill (each has `data-lift="squat|bench|deadlift"`). All existing IDs are untouched.
- `#checkpointLabel` — the caption under `#state` in the (former "depth") module. Set per lift: "Depth light" / "Chest touch · lockout" / "Lockout light".
- `#cmdTracker` — command-sequence rail inside the checkpoint module. Rendered from `LIFTS[lift].commands`; visible only in competition mode. Chips get `.done` / `.current` classes as commands are issued.
- `LIFTS` config object (top of the script's LIFT SELECT section) — single place to change labels, command lists, intro copy, and per-lift "good" notes.

### Generic data contract (what the backend feeds the hooks)
The status payload over `/ws/live` is unchanged and backward compatible; two new optional fields generalize it:
- `checkpoint_met: true | false | null` — drives the checkpoint light (`#light`: white / red / grey). Falls back to `below_parallel` if absent, so the current squat backend works unmodified.
- `lift_progress: 0..1` — drives the progress bar (`#fill`). Falls back to `depth_progress`.
- `command: string` — unchanged, but now any of `SQUAT | RACK | START | PRESS | DOWN`. Intermediate commands auto-dismiss the banner after 1s; the **last** command in the lift's sequence holds until the verdict. Duplicate consecutive `command` values are ignored (safe to resend).
- `state: string` — displayed verbatim; `DESCENDING/ASCENDING/LOWERING/PRESSING/PULLING` pulse the lamp.
- `rep_completed` / `verdict {verdict, faults[]}` / `note` / `keypoints` — unchanged.

### Control messages (browser → server)
`{cmd:'start', mode:'training'|'competition', lift:'squat'|'bench'|'deadlift'}` — `lift` was added to every start message (mode switch, lift switch, New Attempt).

### History schema (localStorage key `wl.history`)
Entries gained a `lift` field: `{type:'training', lift, time, weight, unit, reps:[{verdict,time}]}` and `{type:'competition', lift, time, verdict, faults[]}`. Entries without `lift` are treated as squat everywhere.

### Bug fixed
`startAttempt()` called `tracker.reset()` on an undefined `tracker` (threw before `sendControl` ran). Now guarded: `if (typeof tracker !== 'undefined' && ...)`.

### Demo simulator
Now plays all six lift × mode combos: per-lift free-rep cycles in training, and a scripted attempt (commands → checkpoint → verdict) when an attempt is armed in competition.

## New pages (need FastAPI routes)
- `web/landing.html` — marketing/intro page. Static.
- `web/history.html` — full log view; reads/writes the same `wl.history` key (export/import/clear included). Filters by lift and type.
- `web/stats.html` — inline-SVG charts (good-rate trend, top weight per lift, reps per day) from `wl.history`. Has an in-memory "sample data" preview that never writes to storage.

All pages link to each other with relative hrefs (`live.html`, `history.html`, `stats.html`, `landing.html`, `index.html`) — adjust to your route names if they differ (e.g. `/live`). Theme preference is shared via localStorage key `wl.theme`.
