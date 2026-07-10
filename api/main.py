"""FastAPI surface for White Lights.

Routes
------
GET  /        -> serves the minimal upload UI (web/index.html)
POST /judge   -> accepts one or more video uploads, runs the pipeline, returns
                 per-rep verdicts as JSON. While the core CV logic is stubbed it
                 responds 501 with a clear "core logic not implemented" message.

Deliberately minimal: no database, no auth, synchronous processing. Run with::

    uvicorn api.main:app --reload
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from whitelights.pipeline import judge_video
from whitelights.types import JudgeResult, RefereeCommand

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
