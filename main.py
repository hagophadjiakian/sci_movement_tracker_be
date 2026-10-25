import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas.movement import MovementRequest, MuscleAnalysisResult
from services.ai_service import analyze_sci_movement

load_dotenv()
logger = logging.getLogger("neuromotion")

app = FastAPI(
    title="NeuroMotion SCI Backend",
    version="1.1.0",
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
