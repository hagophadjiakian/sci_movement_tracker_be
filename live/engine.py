"""Live camera + RTMPose engine.

Two threads:
- capture: reads the camera (or a video file at its real speed) and keeps only the newest frame,
  so a slow model never builds up delay;
- inference: runs RTMPose on the newest frame, follows one person, fixes left/right swaps,
  computes 2D angles, draws the skeleton, and publishes the result.

Results go out two ways: an MJPEG video stream with the skeleton drawn in (shown in an <img>),
and a WebSocket message per processed frame with all angle values.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from live.angles2d import ViewSelector, compute_2d
from live.points import KEYPOINT_COUNT, Points, pick_person, to_points
from live.side_tracker import SideTracker

logger = logging.getLogger("neuromotion.live")

COLORS = {"l": (248, 189, 56), "r": (60, 146, 251), "m": (240, 232, 226)}  # BGR: sky, orange, light
BODY_LINES = [
    ("m", "l_shoulder", "r_shoulder"), ("m", "l_hip", "r_hip"),
    ("m", "nose", "l_eye"), ("m", "nose", "r_eye"), ("m", "l_eye", "l_ear"), ("m", "r_eye", "r_ear"),
]
for _s in ("l", "r"):
    BODY_LINES += [(_s, f"{_s}_{a}", f"{_s}_{b}") for a, b in (
        ("shoulder", "elbow"), ("elbow", "wrist"), ("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"),
        ("ankle", "heel"), ("heel", "bigtoe"), ("ankle", "bigtoe"), ("bigtoe", "smalltoe"))]
HAND_LINES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def make_estimator(model: str, mode: str, device: str, det_frequency: int):
    if os.environ.get("LIVE_ESTIMATOR") == "synthetic":
        from live.synthetic import SyntheticEstimator

        return SyntheticEstimator(model)
    from rtmlib import Body, BodyWithFeet, PoseTracker, Wholebody

    solution = {"body": Body, "body_feet": BodyWithFeet, "wholebody": Wholebody}[model]
    return PoseTracker(solution, det_frequency=det_frequency, tracking=False, mode=mode,
                       backend="onnxruntime", device=device)


def open_capture(source: int | str):
    import cv2

    if isinstance(source, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # opens much faster on Windows
    else:
        cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def list_cameras(max_index: int = 5, skip: set[int] | None = None) -> list[dict]:
    import cv2

    found = []
    for i in range(max_index + 1):
        if skip and i in skip:
            found.append({"index": i, "in_use": True})
            continue
        cap = open_capture(i)
        ok = cap.isOpened() and cap.read()[0]
        if ok:
            found.append({"index": i, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})
        cap.release()
    return found


@dataclass
class LiveSettings:
    source: int | str = 0
    model: str = "wholebody"  # body | body_feet | wholebody
    mode: str = "balanced"  # lightweight | balanced | performance
    device: str = "auto"  # auto | cpu | cuda
    view: str = "auto"  # auto | side | front
    det_frequency: int = 7


@dataclass
class LiveStatus:
    state: str = "idle"  # idle | loading | running | error
    error: str | None = None
    settings: dict = field(default_factory=dict)
    device: str | None = None
    fps: float = 0.0
    frame_size: tuple[int, int] | None = None
    people: int = 0
    side_swaps: int = 0


class LiveEngine:
    def __init__(self):
        self.status = LiveStatus()
        self.settings = LiveSettings()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._jpeg: bytes | None = None
        self.jpeg_id = 0
        self.latest: dict | None = None
        self._subscribers: set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = set()
        self._new_frame = threading.Condition()
        self._t0 = time.monotonic()

    # ---- public API -------------------------------------------------------
    def start(self, settings: LiveSettings) -> LiveStatus:
        self.stop()
        self.settings = settings
        self._stop.clear()
        self.status = LiveStatus(state="loading", settings=settings.__dict__.copy())
        self._t0 = time.monotonic()
        self._publish_status()
        t = threading.Thread(target=self._run, name="live-engine", daemon=True)
        self._threads = [t]
        t.start()
        return self.status

    def stop(self) -> LiveStatus:
        self._stop.set()
        with self._new_frame:
            self._new_frame.notify_all()
        for t in self._threads:
            t.join(timeout=5)
        self._threads = []
        if self.status.state in ("running", "loading"):
            self.status.state = "idle"
        self._jpeg = None
        self._publish_status()
        return self.status

    def jpeg(self) -> tuple[int, bytes | None]:
        return self.jpeg_id, self._jpeg

    def subscribe(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        self._subscribers.add((loop, q))
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers = {s for s in self._subscribers if s[1] is not q}

    def camera_in_use(self) -> set[int]:
        s = self.settings.source
        return {s} if self.status.state in ("loading", "running") and isinstance(s, int) else set()

    # ---- internals ------------------------------------------------------------
    def _emit(self, msg: dict):
        for loop, q in list(self._subscribers):
            def put(q=q, msg=msg):
                if q.full():  # slow client: drop the oldest message rather than lag behind
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(msg)
            try:
                loop.call_soon_threadsafe(put)
            except RuntimeError:
                self.unsubscribe(q)

    def _publish_status(self):
        st = self.status.__dict__.copy()
        self._emit({"type": "status", **st})

    def _fail(self, msg: str):
        logger.error("Live engine: %s", msg)
        self.status.state = "error"
        self.status.error = msg
        self._publish_status()

    def _capture_loop(self, cap, is_file: bool):
        import cv2

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        period = 1.0 / fps if is_file and fps > 0 else 0.0
        next_t = time.monotonic()
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the video
                    continue
                self._fail("Camera stopped delivering frames.")
                break
            with self._new_frame:
                self._frame = frame
                self._frame_id += 1
                self._new_frame.notify_all()
            if period:
                next_t += period
                time.sleep(max(0.0, next_t - time.monotonic()))
        cap.release()

    def _run(self):
        import cv2

        s = self.settings
        try:
            device = resolve_device(s.device)
            self.status.device = device
            estimator = make_estimator(s.model, s.mode, device, s.det_frequency)
        except Exception as exc:
            self._fail(f"Could not load the pose model ({type(exc).__name__}: {exc}). "
                       "Is rtmlib installed (pip install -r requirements-live.txt)?")
            return
        if self._stop.is_set():
            return
        source: int | str = s.source
        is_file = isinstance(source, str) and not source.isdigit()
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        cap = open_capture(source)
        if not cap.isOpened():
            self._fail(f"Could not open camera/video '{s.source}'. Is another app using the camera?")
            return
        cap_thread = threading.Thread(target=self._capture_loop, args=(cap, is_file), name="live-capture", daemon=True)
        self._threads.append(cap_thread)
        cap_thread.start()

        tracker = SideTracker()
        views = ViewSelector(s.view)  # type: ignore[arg-type]
        last_center: np.ndarray | None = None
        last_id = 0
        times: list[float] = []
        self.status.state = "running"
        self._publish_status()
        while not self._stop.is_set():
            with self._new_frame:
                while self._frame_id == last_id and not self._stop.is_set():
                    self._new_frame.wait(timeout=1.0)
                if self._stop.is_set():
                    break
                frame = self._frame
                last_id = self._frame_id
            if frame is None:
                continue
            h, w = frame.shape[:2]
            try:
                kpts, scores = estimator(frame)
            except Exception as exc:
                self._fail(f"Pose model failed ({type(exc).__name__}: {exc})")
                break
            kpts = np.asarray(kpts) if len(kpts) else np.zeros((0, KEYPOINT_COUNT[s.model], 2))
            scores = np.asarray(scores) if len(scores) else np.zeros((0, KEYPOINT_COUNT[s.model]))
            idx = pick_person(kpts, scores, last_center)
            values: dict[str, float] = {}
            meta: dict[str, Any] = {"engine": "rtmpose", "dims": "2d", "frame": None,
                                    "hand": {"left": None, "right": None}}
            pts: Points | None = None
            if idx is not None:
                pts = to_points(kpts[idx], scores[idx], s.model, w, h)
                tracker.fix(pts)
                values, meta = compute_2d(pts, views.update(pts))
                good = [p for p in pts.named.values() if p is not None]
                last_center = np.mean(good, axis=0) if good else None
            else:
                tracker.reset()
                last_center = None

            now = time.monotonic()
            times = [x for x in times if now - x < 1.0] + [now]
            self.status.fps = round(len(times) / max(1e-3, now - times[0]), 1) if len(times) > 1 else 0.0
            self.status.frame_size = (w, h)
            self.status.people = int(len(kpts))
            self.status.side_swaps = tracker.swaps

            img = frame.copy()
            if pts is not None:
                self._draw(img, pts)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                self._jpeg = buf.tobytes()
                self.jpeg_id += 1
            msg = {"type": "frame", "t": round(now - self._t0, 3), "fps": self.status.fps,
                   "people": self.status.people, "values": values, "meta": meta}
            self.latest = msg
            self._emit(msg)
        if self.status.state == "running":
            self.status.state = "idle"
            self._publish_status()

    @staticmethod
    def _draw(img: np.ndarray, p: Points):
        import cv2

        th = max(2, img.shape[0] // 240)

        def pt(a):
            return int(a[0]), int(a[1])

        for side, a, b in BODY_LINES:
            pa, pb = p.get(a), p.get(b)
            if pa is not None and pb is not None:
                cv2.line(img, pt(pa), pt(pb), COLORS[side], th, cv2.LINE_AA)
        for s in ("l", "r"):
            hand = p.hands.get(s)
            if hand is not None:
                for a, b in HAND_LINES:
                    cv2.line(img, pt(hand[a]), pt(hand[b]), COLORS[s], max(1, th // 2), cv2.LINE_AA)
            wr = p.get(f"{s}_wrist")
            if wr is not None:
                cv2.putText(img, "L" if s == "l" else "R", (int(wr[0]) + 8, int(wr[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[s], 2, cv2.LINE_AA)


engine = LiveEngine()
