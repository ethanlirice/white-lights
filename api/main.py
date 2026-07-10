"""FastAPI surface for White Lights.

Routes
------
GET  /        -> serves the minimal upload UI (web/index.html)
POST /judge   -> accepts one or more video uploads, runs the batch pipeline,
                 returns per-rep verdicts as JSON.
GET  /live     -> serves the live webcam judge UI (web/live.html)
WS   /ws/live  -> streams JPEG frames in, returns per-frame keypoints + the live
                 tracker's reasoning as JSON (one response per frame).

Deliberately minimal: no database, no auth. Run with::

    uvicorn api.main:app --reload
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from whitelights.pipeline import judge_video
from whitelights.types import JudgeResult, RefereeCommand

if TYPE_CHECKING:
    from whitelights.live import LiveJudge, LiveStatus
    from whitelights.types import FrameKeypoints

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(
    title="White Lights",
    version="2.0.0.dev0",
    description="Real-time computer-vision squat-depth judge for powerlifting.",
)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the static upload page."""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/live", include_in_schema=False)
def live_page() -> FileResponse:
    """Serve the live webcam judge UI."""
    return FileResponse(WEB_DIR / "live.html")


def live_payload(frame2d: FrameKeypoints, status: LiveStatus, width: int, height: int) -> dict:
    """Build the per-frame JSON the browser renders (see web/live.html + HANDOFF.md).

    Keypoints are a list of ``{name, x, y, confidence}`` normalised to [0, 1]
    against the processed frame size so the client can scale them to any canvas.
    ``verdict`` is only populated on the frame a rep completes.
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
    return {
        "state": str(status.state),
        "below_parallel": status.below_parallel,
        "depth_progress": max(0.0, min(1.0, status.descent_fraction or 0.0)),
        "rep_completed": status.rep_completed,
        "verdict": verdict,
        "note": status.note,
        "keypoints": keypoints or None,
        "command": status.command,  # "SQUAT"/"RACK" in competition mode, else None
    }


def _process_frame_bytes(judge: LiveJudge, data: bytes) -> dict:
    """Decode a JPEG frame and run one live-judging step (runs off-thread)."""
    import cv2
    import numpy as np

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "could not decode frame"}
    height, width = img.shape[:2]
    frame2d, _depth, status = judge.process_frame(img)
    return live_payload(frame2d, status, width, height)


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    """Receive JPEG frames, return per-frame keypoints + tracker reasoning."""
    await ws.accept()
    from whitelights.live import LiveJudge
    from whitelights.pose import PoseEstimator

    judge = LiveJudge(PoseEstimator())
    loop = asyncio.get_event_loop()
    try:
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
                payload = await loop.run_in_executor(None, _process_frame_bytes, judge, data)
            except ModuleNotFoundError as exc:
                await ws.send_json(
                    {"error": f"pose runtime not installed ({exc.name}); pip install -e '.[cv]'"}
                )
                await ws.close()
                return
            except Exception as exc:  # noqa: BLE001 - keep the socket alive on a bad frame
                await ws.send_json({"error": str(exc)})
                continue
            await ws.send_json(payload)
    except WebSocketDisconnect:
        return


def _handle_control(judge: LiveJudge, text: str) -> None:
    """Apply a client control message: reset a set, or switch mode (training vs
    competition) at the start of a set / attempt."""
    try:
        cmd = json.loads(text)
    except json.JSONDecodeError:
        return
    if cmd.get("cmd") not in ("reset", "start"):
        return
    from whitelights.live import CompetitionTracker, OnlineRepTracker

    mode = cmd.get("mode")
    if mode == "competition":
        judge.set_tracker(CompetitionTracker())
    elif mode == "training":
        judge.set_tracker(OnlineRepTracker())
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
