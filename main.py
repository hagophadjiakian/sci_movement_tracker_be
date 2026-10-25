import logging
import os

import asyncio
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from schemas.movement import MovementRequest, MuscleAnalysisResult
from schemas.session import SessionInfo, SessionUpload
from services import session_store
from services.ai_service import analyze_sci_movement

load_dotenv()
logger = logging.getLogger("neuromotion")

app = FastAPI(
    title="NeuroMotion SCI Backend",
    version="1.3.0",
    description="Estimates likely muscle contributions from joint angles using Gemini.",
)

origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.on_event("startup")
async def _startup():
    session_store.load_existing()


@app.get("/")
async def root():
    return {"status": "online", "service": "NeuroMotion SCI AI Engine"}


@app.post("/api/analyze-movement", response_model=MuscleAnalysisResult)
async def analyze_movement_endpoint(req: MovementRequest):
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")
    try:
        return await analyze_sci_movement(req)
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=502, detail=f"Analysis failed ({type(exc).__name__}). See server log.")


@app.post("/api/sessions", response_model=SessionInfo, status_code=202)
async def upload_session(up: SessionUpload):
    """Store a recording and analyse it in the background. Returns at once with status 'queued'."""
    return session_store.submit(up)


@app.get("/api/sessions", response_model=list[SessionInfo])
async def list_sessions(include_results: bool = False):
    items = session_store.list_sessions()
    if include_results:
        return items
    return [s.model_copy(update={"result": None}) for s in items]


@app.get("/api/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    info = session_store.get(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return info


# ---- live camera engine (RTMPose, runs on this computer) ---------------------------------


class LiveStartRequest(BaseModel):
    source: int | str = Field(default=0, description="Camera index (0, 1, ...) or a video file path")
    model: Literal["body", "body_feet", "wholebody"] = "wholebody"
    mode: Literal["lightweight", "balanced", "performance"] = "balanced"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    view: Literal["auto", "side", "front"] = "auto"
    det_frequency: int = Field(default=7, ge=1, le=60)


def _live():
    from live.engine import engine

    return engine


@app.get("/api/live/status")
async def live_status():
    return _live().status.__dict__


@app.get("/api/live/cameras")
async def live_cameras():
    from live.engine import list_cameras

    eng = _live()
    return await asyncio.to_thread(list_cameras, 5, eng.camera_in_use())


@app.post("/api/live/start")
async def live_start(req: LiveStartRequest):
    from live.engine import LiveSettings

    st = await asyncio.to_thread(_live().start, LiveSettings(**req.model_dump()))
    return st.__dict__


@app.post("/api/live/stop")
async def live_stop():
    st = await asyncio.to_thread(_live().stop)
    return st.__dict__


@app.get("/live/video.mjpg")
async def live_video():
    """Camera picture with the skeleton drawn in, as an MJPEG stream for an <img> tag."""
    eng = _live()

    async def frames():
        last = -1
        idle = 0.0
        while True:
            fid, jpg = eng.jpeg()
            if jpg is not None and fid != last:
                last = fid
                idle = 0.0
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
            else:
                await asyncio.sleep(0.01)
                idle += 0.01
                if idle > 30 and eng.status.state not in ("loading", "running"):
                    break

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@app.websocket("/ws/live")
async def live_ws(ws: WebSocket):
    """Angle values for every processed frame, plus status changes."""
    await ws.accept()
    eng = _live()
    q = eng.subscribe(asyncio.get_running_loop())
    try:
        await ws.send_json({"type": "status", **eng.status.__dict__})
        while True:
            msg = await q.get()
            await ws.send_json(msg)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        eng.unsubscribe(q)


@app.on_event("shutdown")
async def _shutdown():
    try:
        from live import engine as mod

        await asyncio.to_thread(mod.engine.stop)
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
