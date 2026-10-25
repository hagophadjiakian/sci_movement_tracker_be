"""Movement detector: slow drifts, sensitivity, fast movements, clonus, not-assessed channels."""

import numpy as np
import pytest

from schemas.session import SessionUpload
from services.movement_detector import analyze_session

FS = 30


def run(signals: dict, sensitivity="medium", dur=40.0):
    t = np.arange(0, dur, 1 / FS)
    joints = {k: [None if np.isnan(v) else round(float(v), 2) for v in f(t)] for k, f in signals.items()}
    up = SessionUpload(sci_level="T6", started_at="2026-10-25T10:00:00Z", t=t.tolist(), joints=joints,
                       sensitivity=sensitivity)
    return analyze_session(up)


rng = np.random.default_rng(3)


def noisy(base=10.0, sd=0.4):
    return lambda t: base + rng.normal(0, sd, len(t))


def slow_lift(amp=4.0, start=15.0, rise=2.5, hold=2.0):
    """Weak, slow movement: amp degrees over `rise` seconds (about 1.6 deg/s), hold, return."""
    def f(t):
        y = 10 + rng.normal(0, 0.4, len(t))
        up_ = np.clip((t - start) / rise, 0, 1)
        down = np.clip((t - start - rise - hold) / rise, 0, 1)
        return y + amp * (up_ - down)
    return f


@pytest.mark.parametrize("sens,found", [("low", False), ("medium", True), ("high", True)])
def test_slow_small_movement_depends_on_sensitivity(sens, found):
    r = run({"left.hip_flexion": slow_lift(amp=4.0)}, sensitivity=sens)
    hits = [b for b in r.bouts if b.joint == "left.hip_flexion" and 13 < b.start < 22]
    assert bool(hits) == found, [(b.start, b.excursion_deg, b.slow) for b in r.bouts]
    if found:
        assert any(b.slow for b in hits)
        assert max(b.excursion_deg for b in hits) == pytest.approx(4.0, abs=1.2)


def test_still_joint_gives_no_bouts_even_on_high():
    r = run({"left.knee_flexion": noisy()}, sensitivity="high")
    assert not [b for b in r.bouts if b.joint == "left.knee_flexion"]


def test_fast_movement_still_found_and_not_slow():
    def lift(t):
        y = 10 + rng.normal(0, 0.4, len(t))
        m = (t >= 10) & (t < 11.5)
        y[m] += 30 * np.sin(np.pi * (t[m] - 10) / 1.5)
        return y
    r = run({"left.knee_flexion": lift})
    b = [x for x in r.bouts if x.joint == "left.knee_flexion"]
    assert len(b) == 1 and not b[0].slow and b[0].excursion_deg == pytest.approx(30, abs=2)


def test_clonus_flagged():
    def clonus(t):
        y = rng.normal(0, 0.4, len(t))
        m = (t >= 20) & (t < 22)
        y[m] += 6 * np.sin(2 * np.pi * 6 * t[m])
        return y
    r = run({"left.ankle_dorsiflexion": clonus})
    assert any(b.clonus_like for b in r.bouts)


def test_hidden_channel_is_not_assessed():
    r = run({"left.knee_flexion": lambda t: np.full(len(t), np.nan), "left.elbow_flexion": noisy()})
    assert "left.knee_flexion" in r.not_assessed and "legs" in r.summary


def test_continuous_movement_not_all_slow():
    """A joint moving most of a short recording at normal speed must not be labelled slow."""
    r = run({"left.hip_flexion": lambda t: 10 + 35 * (0.5 - 0.5 * np.cos(2 * np.pi * t / 4)) + rng.normal(0, 0.4, len(t))},
            sensitivity="high", dur=12)
    b = [x for x in r.bouts if x.joint == "left.hip_flexion"]
    assert b and not all(x.slow for x in b)


def test_angle_wrap_near_180_is_not_a_movement():
    """Lying down, a segment angle sits near +/-180 and flips sign with tiny jitter: no fake 358-degree bout."""
    def wrap(t):
        y = 179 + rng.normal(0, 1.0, len(t))
        return (y + 180) % 360 - 180
    r = run({"left.shank_incl": wrap}, sensitivity="high")
    assert not [b for b in r.bouts if b.excursion_deg > 20], [b.excursion_deg for b in r.bouts]


def test_single_frame_glitches_removed():
    """The tracker sometimes outputs a nonsense angle for one frame: not a movement."""
    def glitchy(t):
        y = 20 + rng.normal(0, 0.4, len(t))
        y[::97] = -150  # isolated garbage frames
        return y
    r = run({"left.knee_flexion": glitchy}, sensitivity="high")
    assert not [b for b in r.bouts if b.joint == "left.knee_flexion"]
    assert "glitch frames removed" in r.summary
