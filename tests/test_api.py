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
