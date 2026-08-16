# White Lights — `/ws/live` handoff (historical)

**Status: implemented.** This was the pre-implementation spec for the
`/ws/live` WebSocket route — written when the endpoint didn't exist yet, kept
for the design reasoning ("why binary frames, why JSON per frame, why run the
model off-thread"), not as a current reference for the wire format. The wire
format itself has grown well past what's described below: `checkpoint_met`,
`lift_progress`, `command`, and `geometry` were all added since (see
`docs/HANDOFF-UI.md` for the first round of that, and `whitelights/depth.py` /
`api/main.py:depth_geometry` for the most recent).

**For the current wire contract, read the code that can't drift from it:**

- `api/main.py:live_payload` — builds the exact JSON sent per frame.
- `tests/test_wire_contract.py` — asserts on that JSON directly, for every
  lift × terminal scenario. This is the actual spec; it fails the moment the
  payload and this suite disagree, which a hand-written doc cannot promise.

## Why it looked like this

**Client -> server:** binary WebSocket messages, one JPEG frame (~480px wide)
per message, sent roughly every 150ms — still true today.

**Server -> client:** one JSON text message per processed frame, decode
off the asyncio event loop (`run_in_executor` / the pool in `api/runtime.py`)
since `judge.process_frame` calls into the pose model synchronously and would
otherwise block every other connection — still true today, and is now a
bounded pool rather than one model per connection (see
`docs/ARCHITECTURE.md`'s "Scaling beyond one process").

## Scope reminder (as originally written)

Only `api/main.py` (new route) and possibly a small helper module were meant
to change. That constraint applied to *this* task; it does not describe a
rule for the codebase generally — `web/live.html`, `whitelights/`, `tests/`,
and `eval/` have all changed substantially since, tracked in
`docs/ROADMAP.md`.
