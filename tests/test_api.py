"""API surface tests using FastAPI's TestClient (no model / torch required).

The pipeline is monkeypatched so we test the HTTP contract — routing, upload
handling, and error mapping — independent of the CV core.
"""

from __future__ import annotations

import io

from fastapi.testclient import TestClient

import api.main as main
from api.main import app

client = TestClient(app)


def test_index_serves_upload_page() -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "White Lights" in res.text


def test_live_page_served() -> None:
    res = client.get("/live")
    assert res.status_code == 200
    assert "live" in res.text.lower()
    assert "/ws/live" in res.text  # client wires up the websocket


def test_live_payload_normalises_keypoints_and_verdict() -> None:
    from whitelights.live import LiveState, LiveStatus
    from whitelights.types import FrameKeypoints, Keypoint2D, RepVerdict, Verdict

    frame = FrameKeypoints(
        frame_idx=0,
        time_s=0.0,
        keypoints={"left_hip": Keypoint2D(name="left_hip", x=240.0, y=120.0, confidence=0.9)},
        detected=True,
        subject_confidence=0.9,
    )
    status = LiveStatus(
        state=LiveState.DESCENDING,
        note="descending…",
        below_parallel=False,
        depth_margin=-3.0,
        hip_z=-120.0,
        standing_ref=-40.0,
        descent_fraction=0.5,
        rep_count=1,
        last_verdict=RepVerdict(
            rep_index=0,
            verdict=Verdict.NO_LIFT,
            confidence=0.8,
            faults=[],
            start_frame=0,
            end_frame=10,
            start_time_s=0.0,
            end_time_s=0.3,
        ),
        rep_completed=True,
    )
    payload = main.live_payload(frame, status, width=480, height=240)

    kp = payload["keypoints"]["left_hip"]
    assert kp["x"] == 0.5 and kp["y"] == 0.5  # normalised to [0, 1]
    assert payload["state"] == "DESCENDING"
    assert payload["rep_count"] == 1
    assert payload["last_verdict"]["verdict"] == "NO_LIFT"
    assert payload["last_verdict"]["duration_s"] == 0.3  # end 0.3 - start 0.0


def test_ws_live_reports_missing_pose_runtime() -> None:
    # No `cv` extra in the test env, so the socket should surface a clear error
    # (this still exercises accept -> receive -> process -> error path).
    with client.websocket_connect("/ws/live") as ws:
        ws.send_bytes(b"not-a-real-jpeg")
        msg = ws.receive_json()
        assert "error" in msg


def test_ws_live_accepts_reset_control() -> None:
    # A control message (start-of-set reset) is handled without needing cv, and
    # the socket stays open afterwards (a following bad frame still errors).
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text('{"cmd": "reset"}')
        ws.send_bytes(b"x")
        msg = ws.receive_json()
        assert "error" in msg


def test_judge_maps_not_implemented_to_501(monkeypatch) -> None:
    def _raise(*_args, **_kwargs):
        raise NotImplementedError("smoothing not implemented")

    monkeypatch.setattr(main, "judge_video", _raise)
    res = client.post(
        "/judge",
        files={"files": ("squat.mp4", io.BytesIO(b"not a real video"), "video/mp4")},
    )
    assert res.status_code == 501
    detail = res.json()["detail"]
    assert detail["error"] == "core_logic_not_implemented"
    assert "not implemented" in detail["message"].lower()


def test_judge_rejects_bad_commands_json(monkeypatch) -> None:
    # Should fail at command parsing before ever touching the pipeline.
    monkeypatch.setattr(main, "judge_video", lambda *a, **k: None)
    res = client.post(
        "/judge",
        files={"files": ("squat.mp4", io.BytesIO(b"x"), "video/mp4")},
        data={"commands": "{not json}"},
    )
    assert res.status_code == 422


def test_judge_accepts_valid_commands(monkeypatch) -> None:
    captured = {}

    def _capture(paths, commands=None, *a, **k):
        captured["commands"] = commands
        raise NotImplementedError("stub")

    monkeypatch.setattr(main, "judge_video", _capture)
    res = client.post(
        "/judge",
        files={"files": ("squat.mp4", io.BytesIO(b"x"), "video/mp4")},
        data={"commands": '[{"command": "START", "time_s": 2.5}]'},
    )
    assert res.status_code == 501
    assert captured["commands"][0].command.value == "START"
    assert captured["commands"][0].time_s == 2.5
