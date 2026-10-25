from typing import Literal

from pydantic import BaseModel, Field, model_validator

from schemas.movement import AisGrade, SciLevel

MAX_SECONDS = 330  # 5 min recording + margin


class SessionEvent(BaseModel):
    t: float = Field(ge=0, description="Seconds from recording start")
    type: Literal["command", "warning", "start", "stop", "note"]
    text: str = Field(max_length=200)


class SessionUpload(BaseModel):
    sci_level: SciLevel
    ais_grade: AisGrade = "unknown"
    exercise_name: str = Field(default="Free movement", max_length=100)
    clinical_context: str | None = Field(default=None, max_length=1000)
    started_at: str = Field(description="ISO time the recording started (client clock)")
    stop_reason: Literal["voice_stop", "time_limit", "button"] = "time_limit"
    # Columnar time series: t[i] goes with joints[name][i]. None = joint not visible in that frame.
    t: list[float]
    joints: dict[str, list[float | None]]
    events: list[SessionEvent] = []
    # Reserved for later: EMG envelopes (channel -> samples on the same t) and camera count.
    emg: dict[str, list[float | None]] = {}
    camera_count: int = Field(default=1, ge=1, le=8)

    @model_validator(mode="after")
    def _check(self):
        n = len(self.t)
        if n < 2:
            raise ValueError("need at least 2 frames")
        if self.t[-1] > MAX_SECONDS:
            raise ValueError(f"recording longer than {MAX_SECONDS} s")
        if len(self.joints) > 200:
            raise ValueError("too many joint channels")
        for k, v in self.joints.items():
            if len(v) != n:
                raise ValueError(f"joint '{k}' has {len(v)} samples, expected {n}")
        return self


SessionStatus = Literal["queued", "analyzing", "done", "error"]


class MovementBout(BaseModel):
    joint: str
    start: float
    end: float
    excursion_deg: float
    peak_speed_dps: float
    direction: Literal["increase", "decrease"]
    oscillation_hz: float | None = None
    clonus_like: bool = False
    first_mover: str | None = None
    lead_ms: float | None = None
    axial_led: bool = False
    # Hip bouts only: how much each segment turned (vs vertical) and which one produced the hip angle change.
    trunk_turn_deg: float | None = None
    thigh_turn_deg: float | None = None
    driver: Literal["thigh", "trunk", "both"] | None = None


class JointSummary(BaseModel):
    joint: str
    visible_fraction: float
    min_deg: float | None
    max_deg: float | None
    rom_deg: float | None = Field(description="5th-95th percentile range")
    noise_deg: float | None = Field(description="Resting jitter estimate (1 SD)")
    bouts: int


class SessionResult(BaseModel):
    duration_s: float
    fps: float
    joints: list[JointSummary]
    bouts: list[MovementBout]
    summary: str
    not_assessed: list[str] = Field(default=[], description="Channels visible too little to judge; unknown, not 'no movement'")
    ai_note: str | None = None
    caveats: list[str]


class SessionInfo(BaseModel):
    id: str
    status: SessionStatus
    created_at: str
    started_at: str
    duration_s: float
    stop_reason: str
    sci_level: str
    exercise_name: str
    error: str | None = None
    result: SessionResult | None = None
