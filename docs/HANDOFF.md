# White Lights — `/ws/live` handoff

`web/live.html` is a finished, self-contained frontend for the live webcam judge.
Do not change its structure, element IDs, or JS — it already expects a WebSocket
endpoint at `/ws/live` and works standalone (via a built-in offline simulator)
until that endpoint exists.

## What to build

A FastAPI WebSocket route, `GET /ws/live` (upgrades to WS), that wraps the
existing `whitelights.live.LiveJudge` / `OnlineRepTracker` for one connected
client at a time.

## Wire protocol

**Client -> server:** binary WebSocket messages, each one JPEG frame (~480px
wide), sent roughly every 150ms.

**Server -> client:** one JSON text message per processed frame:

```json
{
  "state": "STANDING | DESCENDING | ASCENDING",
  "below_parallel": true,
  "depth_progress": 0.62,
  "rep_completed": false,
  "verdict": null,
  "note": "Hip crease 3cm above the knee line.",
  "keypoints": [
    {"name": "left_hip", "x": 412.0, "y": 611.0, "confidence": 0.93}
  ]
}
```

- `below_parallel`: `true` / `false` / `null` (null = gated / not enough signal
  yet — maps to `DepthFrameResult.gated`).
- `depth_progress`: 0-1 float for the progress bar. Not modeled explicitly by
  `LiveStatus` today — derive it (e.g. normalized hip travel toward the
  standing->bottom range, or reuse `depth_margin` scaled by thigh length) or
  send 1.0 the instant `below_parallel` flips true and interpolate on the way
  down.
- `rep_completed` + `verdict`: set only on the frame where
  `LiveStatus.rep_completed` is `True`; `verdict` is `LiveStatus.last_verdict`
  serialized (`RepVerdict.model_dump()` works as-is — `faults` values are the
  `Fault` enum strings already, e.g. `"INSUFFICIENT_DEPTH"`).
- `note`: free text, one line, "what it's thinking" — synthesize something
  readable from the depth margin / state (no field for this in `LiveStatus`
  today; add it in the WS handler, not in `live.py`'s core logic).
- `keypoints`: array of `{name, x, y, confidence}` in the **same pixel space as
  the JPEG the client just sent** (2D, not 3D) — this is `frame2d` from
  `LiveJudge.process_frame`, not the fused 3D frame. Omit or send `null` when
  no person is detected.

## Suggested handler shape

```python
@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    judge = LiveJudge()  # one per connection
    try:
        while True:
            jpeg_bytes = await ws.receive_bytes()
            frame = decode_jpeg_to_bgr(jpeg_bytes)  # cv2.imdecode
            frame2d, depth, status = judge.process_frame(frame)
            await ws.send_json(to_wire_message(frame2d, depth, status))
    except WebSocketDisconnect:
        pass
```

Run `judge.process_frame` in a thread pool executor (`run_in_executor`) — it
calls into the YOLO model synchronously and will block the event loop
otherwise.

## Scope reminder

Only `api/main.py` (new route) and possibly a small helper module need
changes. Do not touch `web/live.html`, `whitelights/`, `tests/`, or `eval/`.
