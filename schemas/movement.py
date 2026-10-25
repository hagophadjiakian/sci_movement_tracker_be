from typing import Literal

from pydantic import BaseModel, Field

SciLevel = Literal[
    "C5 Tetraplegia",
    "C6 Tetraplegia",
    "C7 Tetraplegia",
    "T10 Paraplegia",
]


class KinematicData(BaseModel):
    # All angles use 0 degrees = anatomical neutral (arm at side, elbow straight, wrist straight).
    # Null when the hip is not visible (the angle needs the trunk as reference).
    shoulderElevation: float | None = Field(default=None, ge=0, le=180, description="Angle between trunk and upper arm")
    elbowFlexion: float = Field(ge=0, le=180)
    # Unsigned deviation from neutral. Flexion vs extension is NOT distinguished (single camera).
    wristDeviation: float | None = Field(default=None, ge=0, le=180)
    timestamp: float


class MovementRequest(BaseModel):
    sci_level: SciLevel
    side: Literal["left", "right"] = "left"
    angles: KinematicData
    exercise_name: str = Field(max_length=100)
    # Optional free text: known muscle grades, gravity-eliminated or not, speed, etc.
    clinical_context: str | None = Field(default=None, max_length=1000)


class MuscleAnalysisResult(BaseModel):
    primary_muscles: list[str] = Field(
        description="Muscles most likely acting as primary movers for this posture or movement"
    )
    secondary_muscles: list[str] = Field(
        description="Likely synergists or stabilizers"
    )
    myotomes: str = Field(
        description="Spinal myotome level range most consistent with these muscles (e.g. C5-C6)"
    )
    compensation: str = Field(
        description="Possible compensatory or substitute strategy, or 'None evident from angles alone'"
    )
    clinical_note: str = Field(description="Brief physical therapy note for this movement")
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence given that only joint angles are available"
    )
    limitations: str = Field(
        description="What cannot be determined from these angles (e.g. active vs passive movement)"
    )