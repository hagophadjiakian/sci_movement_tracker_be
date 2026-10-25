"""In-process session queue. Each upload is analysed in a worker thread so several
sessions can be analysed at once while new ones keep arriving."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from schemas.session import SessionInfo, SessionResult, SessionUpload
from services.movement_detector import analyze_session

logger = logging.getLogger("neuromotion.sessions")
DATA_DIR = Path(os.environ.get("SESSION_DIR", "data/sessions"))
MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL_ANALYSES", "4"))

_sessions: dict[str, SessionInfo] = {}
_sem: asyncio.Semaphore | None = None
_tasks: set[asyncio.Task] = set()


def _semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(MAX_PARALLEL)
    return _sem


def load_existing() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted(DATA_DIR.glob("*.info.json")):
        try:
            info = SessionInfo.model_validate_json(p.read_text())
            if info.status in ("queued", "analyzing"):
                info.status, info.error = "error", "Server restarted during analysis; re-upload to retry."
            _sessions[info.id] = info
        except Exception:
            logger.warning("Skipping unreadable %s", p)


def _save(info: SessionInfo) -> None:
    (DATA_DIR / f"{info.id}.info.json").write_text(info.model_dump_json())


def list_sessions() -> list[SessionInfo]:
    return sorted(_sessions.values(), key=lambda s: s.created_at, reverse=True)


def get(session_id: str) -> SessionInfo | None:
    return _sessions.get(session_id)


def submit(up: SessionUpload) -> SessionInfo:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sid = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    (DATA_DIR / f"{sid}.raw.json").write_text(up.model_dump_json())
    info = SessionInfo(
        id=sid, status="queued", created_at=datetime.now(timezone.utc).isoformat(),
        started_at=up.started_at, duration_s=round(up.t[-1] - up.t[0], 2), stop_reason=up.stop_reason,
        sci_level=up.sci_level, exercise_name=up.exercise_name,
    )
    _sessions[sid] = info
    _save(info)
    task = asyncio.create_task(_run(info, up))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return info


async def _run(info: SessionInfo, up: SessionUpload) -> None:
    async with _semaphore():
        info.status = "analyzing"
        _save(info)
        try:
            result: SessionResult = await asyncio.to_thread(analyze_session, up)
            if os.environ.get("GEMINI_API_KEY"):
                try:
                    from services.ai_service import write_session_note
                    result.ai_note = await write_session_note(up, result)
                except Exception as exc:  # the numbers still stand without the note
                    logger.warning("AI note failed: %s", exc)
            info.result, info.status = result, "done"
        except Exception as exc:
            logger.exception("Session %s failed", info.id)
            info.status, info.error = "error", f"{type(exc).__name__}: {exc}"[:300]
        _save(info)
