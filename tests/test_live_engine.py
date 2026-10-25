"""Live engine end to end with a stand-in estimator and a generated video (no camera, no model download)."""

import os
import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


@pytest.fixture()
def video(tmp_path):
    path = str(tmp_path / "clip.avi")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 360))
    for i in range(90):
        frame = np.full((360, 640, 3), 40 + i % 50, np.uint8)
        out.write(frame)
    out.release()
    return path


def test_live_stream(video, monkeypatch):
    monkeypatch.setenv("LIVE_ESTIMATOR", "synthetic")
    from fastapi.testclient import TestClient

    import main

    with TestClient(main.app) as c:
        st = c.post("/api/live/start", json={"source": video, "model": "wholebody", "view": "side"}).json()
        assert st["state"] in ("loading", "running")
        frames = []
        with c.websocket_connect("/ws/live") as ws:
            deadline = time.time() + 10
            while time.time() < deadline and len(frames) < 20:
                msg = ws.receive_json()
                if msg["type"] == "frame":
                    frames.append(msg)
        assert len(frames) >= 20
        last = frames[-1]
        assert last["meta"]["view"] == "side" and last["meta"]["facing"] == "left"
        assert "left.hip_flexion" in last["values"] and "left.elbow_flexion" in last["values"]
        assert "left.index_mcp" in last["values"]  # wholebody hand
        hips = [f["values"]["left.hip_flexion"] for f in frames]
        assert max(hips) - min(hips) > 1  # the synthetic knee lift is moving

        with c.stream("GET", "/live/video.mjpg") as r:
            assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
            chunk = b""
            for part in r.iter_bytes():
                chunk += part
                if b"\xff\xd9" in chunk:  # end of first JPEG
                    break
            assert b"--frame" in chunk and b"\xff\xd8" in chunk

        status = c.get("/api/live/status").json()
        assert status["state"] == "running" and status["fps"] > 5
        assert c.post("/api/live/stop").json()["state"] == "idle"


def test_bad_source_reports_error(monkeypatch):
    monkeypatch.setenv("LIVE_ESTIMATOR", "synthetic")
    from fastapi.testclient import TestClient

    import main

    with TestClient(main.app) as c:
        c.post("/api/live/start", json={"source": "/no/such/video.mp4"})
        for _ in range(50):
            st = c.get("/api/live/status").json()
            if st["state"] == "error":
                break
            time.sleep(0.05)
        assert st["state"] == "error" and "Could not open" in st["error"]
        c.post("/api/live/stop")
