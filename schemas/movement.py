from typing import Literal

from pydantic import BaseModel, Field

# Every neurological level of injury, C1 to S5 (ISNCSCI).
LEVELS = (
    [f"C{i}" for i in range(1, 9)]
    + [f"T{i}" for i in range(1, 13)]
    + [f"L{i}" for i in range(1, 6)]
    + [f"S{i}" for i in range(1, 6)]
)
SciLevel = Literal[tuple(LEVELS)]  # type: ignore[valid-type]
AisGrade = Literal["A", "B", "C", "D", "E", "unknown"]


def _angle(desc: str, lo: float = -180, hi: float = 180):
    return Field(default=None, ge=lo, le=hi, description=desc)


class SideAngles(BaseModel):
    """Angles for one side, in degrees. 0 = anatomical neutral. None = not visible."""

    shoulder_flexion: float | None = _angle("Upper arm forward (+) / backward (-) in the trunk's sagittal plane")
    shoulder_abduction: float | None = _angle("Upper arm out to the side, trunk frontal plane")
    elbow_flexion: float | None = _angle("0 = straight", 0, 180)
    wrist_deviation: float | None = _angle("Unsigned deviation from straight (flexion vs extension not separated)", 0, 180)
    hip_flexion: float | None = _angle("Thigh forward (+) / backward (-) relative to the trunk")
    hip_abduction: float | None = _angle("Thigh out (+) / across the midline (-) relative to the pelvis")
    knee_flexion: float | None = _angle("0 = straight", 0, 180)
    ankle_dorsiflexion: float | None = _angle("Foot up (+) / pointed (-); 0 = foot at 90 deg to the shank", -90, 90)


class AxialAngles(BaseModel):
    neck_flexion: float | None = _angle("Head forward (+) relative to the trunk")
    trunk_lean: float | None = _angle("Trunk tilt from vertical, forward or back (unsigned)", 0, 180)
    trunk_lateral_lean: float | None = _angle("Trunk tilt to the right (+) / left (-) from vertical")
    pelvic_obliquity: float | None = _angle("Right hip higher (+) / lower (-) than the left")


class KinematicData(BaseModel):
    left: SideAngles = SideAngles()
    right: SideAngles = SideAngles()
    axial: AxialAngles = AxialAngles()
    timestamp: float


class MovementRequest(BaseModel):
    sci_level: SciLevel
    ais_grade: AisGrade = "unknown"
    angles: KinematicData
    exercise_name: str = Field(max_length=100)
    # Optional free text: known muscle grades, gravity-eliminated or not, speed, etc.
    clinical_context: str | None = Field(default=None, max_length=1000)


class MuscleAnalysisResult(BaseModel):
    primary_muscles: list[str] = Field(description="Muscles most likely acting as primary movers for this posture or movement")
    secondary_muscles: list[str] = Field(description="Likely synergists or stabilizers")
    myotomes: str = Field(description="Spinal myotome level range most consistent with these muscles (e.g. L2-L4)")
    compensation: str = Field(description="Possible compensatory or substitute strategy, or 'None evident from angles alone'")
    clinical_note: str = Field(description="Brief physical therapy note for this movement")
    confidence: Literal["low", "medium", "high"] = Field(description="Confidence given that only joint angles are available")
    limitations: str = Field(description="What cannot be determined from these angles (e.g. active vs passive movement)")
