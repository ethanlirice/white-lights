"""FastAPI surface for White Lights.

Routes
------
GET  /         -> serves the landing page (web/landing.html), the app's front door
GET  /live     -> serves the live webcam judge UI (web/live.html)
GET  /history  -> serves the training-history page (web/history.html)
GET  /stats    -> serves the stats page (web/stats.html)
GET  /upload   -> serves the batch upload UI (web/upload.html)
GET  /lib/*    -> static ES modules the pages import (web/lib/*.mjs) — pure
                 logic (e.g. the offline simulator's synthetic poses) that
                 has real unit tests (web/lib/*.test.mjs) precisely because
                 it lives outside the inline, untested page scripts.
GET  /metrics  -> per-stage live latency, capacity and pool state as JSON
POST /judge    -> accepts one or more video uploads, runs the batch pipeline,
                 returns per-rep verdicts as JSON.
WS   /ws/live  -> streams JPEG frames in, returns per-frame keypoints + the live
                 tracker's reasoning as JSON (one response per frame).

Deliberately minimal: no database, no auth. Inference runs on a shared, bounded
pool of pre-warmed models (see `api.runtime`) rather than one model per
connection, and sockets beyond the capacity limit are shed with an explicit
"busy" rather than quietly degrading everyone. Run with::

    uvicorn api.main:app

Capacity is tunable without code changes::

    WL_MAX_WORKERS=4 WL_MAX_CONNECTIONS=8 uvicorn api.main:app
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from api.runtime import InferenceRuntime, ServerBusy, StageTimings, timed
from whitelights.pipeline import judge_video
from whitelights.types import JudgeResult, RefereeCommand

if TYPE_CHECKING:
    from whitelights.depth import DepthFrameResult
    from whitelights.live import LiveJudge
    from whitelights.pose import PoseEstimator
    from whitelights.types import FrameKeypoints, LiveStatus

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

#: Shared across every connection: a bounded pool of warm models plus the
#: latency window reported at /metrics. See api/runtime.py for why it is a pool
#: rather than a single instance.
runtime = InferenceRuntime()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm the models at boot so the first lifter is not the one who waits."""
    runtime.start()
    try:
        yield
    finally:
        runtime.stop()


app = FastAPI(
    title="White Lights",
    version="2.0.0.dev0",
    description="Real-time computer-vision powerlifting judge — squat, bench, and deadlift.",
    lifespan=lifespan,
)

# The pages import their pure logic (e.g. web/live.html <- web/lib/poses.mjs)
# as native ES modules — a browser feature, not a dependency — so this needs
# nothing beyond serving the files; there is no bundler in this app's path.
app.mount("/lib", StaticFiles(directory=WEB_DIR / "lib"), name="lib")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the landing page — the app's front door."""
    return FileResponse(WEB_DIR / "landing.html")


@app.get("/live", include_in_schema=False)
def live_page() -> FileResponse:
    """Serve the live webcam judge UI."""
    return FileResponse(WEB_DIR / "live.html")


@app.get("/upload", include_in_schema=False)
def upload_page() -> FileResponse:
    """Serve the batch upload UI."""
    return FileResponse(WEB_DIR / "upload.html")


# The multi-page UI (landing / live / history / stats / upload) links between
# pages with relative `*.html` hrefs; serve those, plus clean
# `/landing|/history|/stats|/upload`.
_PAGES = frozenset({"upload", "live", "landing", "history", "stats"})


@app.get("/{page}.html", include_in_schema=False)
def page_html(page: str) -> FileResponse:
    if page not in _PAGES:
        raise HTTPException(status_code=404, detail="page not found")
    return FileResponse(WEB_DIR / f"{page}.html")


@app.get("/landing", include_in_schema=False)
def landing_page() -> FileResponse:
    return FileResponse(WEB_DIR / "landing.html")


@app.get("/history", include_in_schema=False)
def history_page() -> FileResponse:
    return FileResponse(WEB_DIR / "history.html")


@app.get("/stats", include_in_schema=False)
def stats_page() -> FileResponse:
    return FileResponse(WEB_DIR / "stats.html")


@app.get("/metrics")
def metrics() -> dict:
    """Live-path health: per-stage frame latency, capacity, and pool state.

    Split by stage on purpose — a total frame time does not tell you whether to
    optimise the decoder, the model, or the judge, and for this pipeline the
    answer is almost always the model.
    """
    return runtime.status()


def depth_geometry(depth: DepthFrameResult, height: int, *, judges_depth: bool) -> dict | None:
    """The hip-crease/knee-top comparison as image rows, for the canvas overlay.

    The judge works in world ``z`` (+z up); the single-view lift sets ``z = -y``
    in pixels, so inverting it recovers the image row. Sending the rows the judge
    actually used — rather than letting the client average two hip keypoints and
    hope — is what keeps the overlay honest: which hip, which knee, and how far
    the crease offset drops the landmark are decisions `depth.py` owns.

    Returns ``None`` when there is nothing truthful to draw: a gated frame, or a
    lift whose judge does not apply the depth rule at all.
    """
    if not judges_depth or depth.gated:
        return None
    if depth.hip_crease_z is None or depth.knee_top_z is None:
        return None
    h = height or 1
    return {
        "hip_row": -depth.hip_crease_z / h,
        "knee_row": -depth.knee_top_z / h,
        "margin": depth.depth_margin,
        "below": depth.is_below_parallel,
        "confidence": depth.confidence,
    }


def live_payload(
    frame2d: FrameKeypoints,
    depth: DepthFrameResult,
    status: LiveStatus,
    width: int,
    height: int,
    *,
    judges_depth: bool = True,
) -> dict:
    """Build the per-frame JSON the browser renders (see web/live.html +
    tests/test_wire_contract.py, which asserts on this exact payload —
    docs/HANDOFF.md is the historical pre-implementation spec, not this).

    Keypoints are a list of ``{name, x, y, confidence}`` normalised to [0, 1]
    against the processed frame size so the client can scale them to any canvas.
    ``geometry`` is normalised the same way. ``verdict`` is only populated on the
    frame a rep completes.
    """
    w = width or 1
    h = height or 1
    keypoints = [
        {"name": name, "x": kp.x / w, "y": kp.y / h, "confidence": kp.confidence}
        for name, kp in frame2d.keypoints.items()
    ]
    verdict = None
    if status.rep_completed and status.last_verdict is not None:
        verdict = status.last_verdict.model_dump(mode="json")
    progress = max(0.0, min(1.0, status.descent_fraction or 0.0))
    return {
        "state": str(status.state),
        # Generic per-lift "key checkpoint met" (squat: below parallel; bench: bar
        # on chest; deadlift: locked) drives the checkpoint light. `below_parallel`
        # / `depth_progress` are legacy aliases the UI falls back to.
        "checkpoint_met": status.checkpoint,
        "below_parallel": status.checkpoint,
        "lift_progress": progress,
        "depth_progress": progress,
        "rep_completed": status.rep_completed,
        "verdict": verdict,
        "note": status.note,
        "keypoints": keypoints or None,
        "command": status.command,  # e.g. SQUAT / START / PRESS / RACK / DOWN, else None
        # The measurement behind the call, for the overlay to draw. None when the
        # frame is gated or the lift has no depth rule — see `depth_geometry`.
        "geometry": depth_geometry(depth, height, judges_depth=judges_depth),
    }


def _process_frame_bytes(
    judge: LiveJudge, estimator: PoseEstimator, data: bytes
) -> tuple[dict, StageTimings]:
    """Decode a JPEG frame and run one live-judging step (runs off-thread).

    Timed by stage — decode, model, judge — because "the frame took 90 ms" is
    not actionable but "the model took 84 ms of it" is.
    """
    import cv2
    import numpy as np

    timings = StageTimings()
    with timed(timings, "decode_ms"):
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "could not decode frame"}, timings

    height, width = img.shape[:2]
    judge.estimator = estimator  # this thread's model, borrowed from the pool
    with timed(timings, "inference_ms"):
        frame2d, depth, status = judge.process_frame(img)
    with timed(timings, "judge_ms"):
        payload = live_payload(
            frame2d, depth, status, width, height, judges_depth=judge.tracker.judges_depth
        )
    return payload, timings


def _run_frame(judge: LiveJudge, data: bytes) -> tuple[dict, StageTimings]:
    """Borrow a model from the pool for the duration of one frame."""
    with runtime.borrow() as estimator:
        return _process_frame_bytes(judge, estimator, data)


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    """Receive JPEG frames, return per-frame keypoints + tracker reasoning.

    Capacity is shed, not queued: a client made to wait in line for a real-time
    video socket has a worse experience than one told plainly to come back.
    """
    await ws.accept()
    try:
        with runtime.connection_slot():
            await _stream_frames(ws)
    except ServerBusy as exc:
        await ws.send_json({"error": str(exc), "retry": True})
        await ws.close(code=1013)  # "try again later"
    except WebSocketDisconnect:
        return


async def _stream_frames(ws: WebSocket) -> None:
    from whitelights.live import LiveJudge

    # The estimator is swapped in per frame from the shared pool; this one is a
    # cheap placeholder so LiveJudge has something to hold.
    from whitelights.pose import PoseEstimator

    judge = LiveJudge(PoseEstimator())
    loop = asyncio.get_running_loop()
    while True:
        message = await ws.receive()
        if message["type"] == "websocket.disconnect":
            return
        # Text frames are control messages (e.g. {"cmd": "reset"} to start a set).
        text = message.get("text")
        if text is not None:
            _handle_control(judge, text)
            continue
        data = message.get("bytes")
        if not data:
            continue
        try:
            payload, timings = await loop.run_in_executor(runtime.executor, _run_frame, judge, data)
        except ModuleNotFoundError as exc:
            await ws.send_json(
                {"error": f"pose runtime not installed ({exc.name}); pip install -e '.[cv]'"}
            )
            await ws.close()
            return
        except Exception as exc:  # noqa: BLE001 - keep the socket alive on a bad frame
            await ws.send_json({"error": str(exc)})
            continue
        runtime.latency.record(timings)
        await ws.send_json(payload)


def _handle_control(judge: LiveJudge, text: str) -> None:
    """Apply a client control message: reset a set, or switch mode (training vs
    competition) at the start of a set / attempt."""
    try:
        cmd = json.loads(text)
    except json.JSONDecodeError:
        return
    if cmd.get("cmd") not in ("reset", "start"):
        return
    lift, mode = cmd.get("lift"), cmd.get("mode")
    if lift or mode:
        from whitelights.judges import tracker_for

        judge.set_tracker(tracker_for(lift, mode))
    else:
        judge.reset()


@app.post("/judge", response_model=JudgeResult)
def judge(
    files: list[UploadFile] = File(..., description="One video per camera view."),
    commands: str | None = Form(
        default=None,
        description="Optional JSON array of referee commands, e.g. "
        '[{"command": "START", "time_s": 2.5}, {"command": "RACK", "time_s": 6.0}]',
    ),
) -> JudgeResult:
    """Run the judging pipeline on the uploaded video(s)."""
    parsed_commands = _parse_commands(commands)

    tmp_dir = Path(tempfile.mkdtemp(prefix="whitelights_"))
    try:
        saved = _save_uploads(files, tmp_dir)
        try:
            return judge_video(saved, commands=parsed_commands)
        except NotImplementedError as exc:
            # Expected until the CV core is implemented — surface it clearly.
            raise HTTPException(
                status_code=501,
                detail={
                    "error": "core_logic_not_implemented",
                    "message": (
                        "White Lights v2 core judging logic is not implemented yet: "
                        f"{exc}. Pose estimation runs, but smoothing/fusion/depth/"
                        "reps are stubs."
                    ),
                },
            ) from exc
        except ModuleNotFoundError as exc:
            # The pose runtime (ultralytics/opencv) isn't installed. Give an
            # actionable message rather than a raw 500.
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "pose_runtime_unavailable",
                    "message": (
                        f"Pose runtime not installed ({exc.name}). "
                        'Install it with: pip install -e ".[cv]"'
                    ),
                },
            ) from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _parse_commands(commands: str | None) -> list[RefereeCommand] | None:
    if not commands:
        return None
    try:
        raw = json.loads(commands)
        return [RefereeCommand.model_validate(item) for item in raw]
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid `commands` payload: {exc}") from exc


def _save_uploads(files: list[UploadFile], dest: Path) -> list[Path]:
    if not files:
        raise HTTPException(status_code=422, detail="No video file uploaded.")
    saved: list[Path] = []
    for i, upload in enumerate(files):
        name = Path(upload.filename or f"view{i}.mp4").name
        target = dest / f"{i:02d}_{name}"
        with target.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        saved.append(target)
    return saved


@app.exception_handler(ValueError)
def _value_error_handler(_request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
