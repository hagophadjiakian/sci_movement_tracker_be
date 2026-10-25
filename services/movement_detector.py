"""Joint-angle movement detection for a recorded session.

Movement is decided per joint from its angle trace, never from raw pixel motion:
an onset needs the angular speed above an adaptive threshold for >= 150 ms AND an
excursion bigger than both 3x that joint's resting jitter and 3 degrees.
"""

from __future__ import annotations

import numpy as np

from schemas.session import JointSummary, MovementBout, SessionResult, SessionUpload

AXIAL_PREFIX = "axial."
MIN_BOUT_S = 0.15
MIN_EXCURSION = 3.0
MERGE_GAP_S = 0.25
MIN_VISIBLE = 0.5
# sensitivity -> (noise multiple, minimum excursion in degrees, speed floor in deg/s)
SENSITIVITY = {"low": (4.0, 5.0, 15.0), "medium": (3.0, 3.0, 10.0), "high": (2.5, 2.0, 5.0)}
DRIFT_MIN_S = 0.5  # a slow drift must last this long  # a channel seen less than this fraction of the time is "not assessed"

REGIONS = (
    ("fingers", ("thumb_", "index_", "middle_", "ring_", "little_", "spread_", "pinch_")),
    ("wrists/forearms", ("wrist_", "forearm_")),
    ("legs", ("hip_", "knee_", "ankle_", "foot_", "thigh_", "shank_")),
    ("arms", ("shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_")),
    ("shoulder girdle", ("shoulder_elevation", "shoulder_protraction")),
    ("neck", ("neck_",)),
    ("trunk/pelvis", ("trunk_", "pelvic_")),
)


def region(name: str) -> str:
    base = name.split(".", 1)[-1]
    for label, keys in REGIONS:
        if any(base.startswith(k) for k in keys):
            return label
    return "other"


def unit(name: str) -> str:
    return "%" if name.endswith(("_aperture", "_elevation")) else "deg"
LEAD_FLAG_MS = 50.0


def _resample(t: np.ndarray, y: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return grid, np.full_like(grid, np.nan)
    out = np.interp(grid, t[ok], y[ok])
    # Do not invent data across gaps longer than 0.3 s.
    idx = np.searchsorted(t[ok], grid)
    idx = np.clip(idx, 1, ok.sum() - 1)
    gap = t[ok][idx] - t[ok][idx - 1]
    out[gap > 0.3] = np.nan
    return grid, out


def _smooth(y: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return y
    k = np.ones(n) / n
    filled = np.where(np.isfinite(y), y, 0.0)
    w = np.convolve(np.isfinite(y).astype(float), k, mode="same")
    s = np.convolve(filled, k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / w
    out[w < 0.5] = np.nan
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


SPIKE_DEG = 25.0  # a single frame this far from its neighbours is a tracking glitch
MAX_EXCURSION_DEG = 180.0  # no joint movement is larger; bigger "movements" are tracking failures


def _unwrap_segments(t: np.ndarray, y: np.ndarray, idx: np.ndarray) -> None:
    breaks = np.flatnonzero(np.diff(t[idx]) > 0.3)
    for seg in np.split(idx, breaks + 1):
        if seg.size < 2:
            continue
        u = np.degrees(np.unwrap(np.radians(y[seg])))
        u -= 360.0 * np.round(np.median(u) / 360.0)  # keep the stretch around its true range
        y[seg] = u


def _clean_deg(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    """Make an angle trace usable: no +/-180 flips (segment near horizontal when lying down), no
    one-frame glitches. Unwrapping happens only inside continuous stretches so gaps cannot add up.
    Returns the cleaned trace and the number of glitch frames removed."""
    y = y.copy()
    idx = np.flatnonzero(np.isfinite(y))
    if idx.size < 3:
        return y, 0
    _unwrap_segments(t, y, idx)
    removed = 0
    v = y[idx]
    if v.size >= 7:
        from numpy.lib.stride_tricks import sliding_window_view

        med = np.median(sliding_window_view(np.pad(v, 3, mode="edge"), 7), axis=1)
        bad = np.abs(v - med) > SPIKE_DEG
        removed = int(bad.sum())
        y[idx[bad]] = np.nan
    idx = np.flatnonzero(np.isfinite(y))
    if idx.size >= 2:
        y[idx] = (y[idx] + 180.0) % 360.0 - 180.0  # forget offsets the glitches may have caused
        _unwrap_segments(t, y, idx)
    return y, removed


def _rolling_baseline(y: np.ndarray, fs: float, lag_lo: float = 1.0, lag_hi: float = 3.0) -> np.ndarray:
    """Median of the signal between lag_hi and lag_lo seconds ago: where the joint was resting."""
    n = len(y)
    out = np.full(n, np.nan)
    step = max(1, int(fs / 5))
    lo, hi = int(lag_lo * fs), int(lag_hi * fs)
    for i in range(hi, n, step):
        w = y[i - hi:i - lo]
        w = w[np.isfinite(w)]
        if w.size > 0.5 * (hi - lo):
            out[i:i + step] = np.median(w)
    return out


def _dominant_hz(seg: np.ndarray, fs: float) -> float | None:
    seg = seg[np.isfinite(seg)]
    if len(seg) < int(fs * 0.6):
        return None
    seg = seg - seg.mean()
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / fs)
    band = freqs > 0.5
    if not band.any():
        return None
    return float(freqs[band][np.argmax(spec[band])])


def analyze_session(up: SessionUpload) -> SessionResult:
    t = np.asarray(up.t, float)
    dur = float(t[-1] - t[0])
    fps_raw = (len(t) - 1) / dur if dur > 0 else 30.0
    fs = float(min(60.0, max(10.0, round(fps_raw))))
    smooth_n = max(1, int(round(fs * 0.1)))
    k_noise, min_exc, speed_floor = SENSITIVITY[up.sensitivity]

    summaries: list[JointSummary] = []
    bouts: list[MovementBout] = []
    glitches = 0
    rejected = 0

    for name, raw in up.joints.items():
        y = np.array([np.nan if v is None else v for v in raw], float)
        vis = float(np.isfinite(y).mean())
        if unit(name) == "deg":
            y, removed = _clean_deg(t, y)
            glitches += removed
        grid, yr = _resample(t, y, fs)
        ys = _smooth(yr, smooth_n)
        good = np.isfinite(ys)
        if good.sum() < fs:  # under 1 s of usable data
            summaries.append(JointSummary(joint=name, visible_fraction=vis, min_deg=None, max_deg=None,
                                          rom_deg=None, noise_deg=None, bouts=0))
            continue
        resid = yr - _smooth(yr, max(3, int(fs * 0.5)))
        r = resid[np.isfinite(resid)]
        noise = float(1.4826 * np.median(np.abs(r - np.median(r)))) if len(r) else 0.5
        noise = max(noise, 0.3)

        speed = np.abs(np.gradient(np.where(good, ys, np.nan), 1.0 / fs))
        sp = speed[np.isfinite(speed)]
        floor = speed_floor / 2 if name.startswith(AXIAL_PREFIX) or "_incl" in name else speed_floor
        adaptive = float(np.median(sp) + 5 * 1.4826 * np.median(np.abs(sp - np.median(sp))))
        # Adapt to jitter, but never so high that ordinary movements count as "slow" (happens when the
        # joint moves for most of a short recording and inflates the median speed).
        sp_thr = min(max(floor, adaptive), 4 * floor)
        exc_thr = max(k_noise * noise, min_exc)
        moving = np.nan_to_num(speed) > sp_thr

        # Slow movements: the joint drifts away from where it rested 1-3 s earlier, even if too slowly
        # to pass the speed threshold (common for weak voluntary attempts after SCI).
        lp = _smooth(ys, max(3, int(fs * 0.5)))
        baseline = _rolling_baseline(lp, fs)
        drift = np.nan_to_num(np.abs(lp - baseline)) > exc_thr
        drift_runs = [(a, b) for a, b in _runs(drift) if (b - a) / fs >= DRIFT_MIN_S]
        drift_long = np.zeros_like(drift)
        for a, b in drift_runs:
            drift_long[a:b] = True
        moving = moving | drift_long

        # Walk each onset back to where the joint really started moving (speed < 25% of threshold),
        # then merge runs separated by a pause where the joint stays away from its baseline
        # (e.g. the still moment at the top of a knee lift: up-hold-down is one movement).
        runs = _runs(moving)
        merged: list[list[int]] = []
        for a, b in runs:
            while a > 0 and np.isfinite(speed[a - 1]) and speed[a - 1] > 0.25 * sp_thr:
                a -= 1
            if merged:
                pa = merged[-1][0]
                pre = ys[max(0, pa - int(0.2 * fs)):pa + 1]
                base = np.nanmedian(pre) if np.isfinite(pre).any() else ys[pa]
                gap = ys[merged[-1][1]:a]
                gap_s = (a - merged[-1][1]) / fs
                away = gap.size == 0 or (np.isfinite(gap).all() and np.min(np.abs(gap - base)) > exc_thr)
                if a <= merged[-1][1] or gap_s < MERGE_GAP_S or (gap_s < 3.0 and away):
                    merged[-1][1] = max(b, merged[-1][1])
                    continue
            merged.append([a, b])
        n_bouts = 0
        for a, b in merged:
            if (b - a) / fs < MIN_BOUT_S:
                continue
            pre = ys[max(0, a - int(0.2 * fs)):a + 1]
            seg = ys[a:b]
            base = np.nanmedian(pre) if np.isfinite(pre).any() else seg[0]
            # For slow drifts the 0.2 s before onset is already moving; use the resting baseline too.
            if np.isfinite(baseline[a]) and drift_long[a:b].any():
                if np.nanmax(np.abs(seg - baseline[a])) > np.nanmax(np.abs(seg - base)):
                    base = baseline[a]
            dev = seg - base
            if not np.isfinite(dev).any():
                continue
            k = int(np.nanargmax(np.abs(dev)))
            exc = float(abs(dev[k]))
            if exc < exc_thr:
                continue
            if exc > MAX_EXCURSION_DEG:
                rejected += 1  # impossible for a joint: the tracker lost the body
                continue
            hz = _dominant_hz(seg, fs)
            n_bouts += 1
            bouts.append(MovementBout(
                joint=name, start=float(grid[a]), end=float(grid[min(b, len(grid) - 1)]),
                excursion_deg=round(exc, 1), peak_speed_dps=round(float(np.nanmax(speed[a:b])), 1),
                direction="increase" if dev[k] > 0 else "decrease",
                oscillation_hz=None if hz is None else round(hz, 1),
                clonus_like=bool(hz is not None and 4.5 <= hz <= 8.5 and (b - a) / fs >= 0.6),
                slow=bool(np.nanmax(speed[a:b]) < sp_thr),
            ))
        p5, p95 = np.nanpercentile(ys, [5, 95])
        summaries.append(JointSummary(
            joint=name, visible_fraction=round(vis, 3), min_deg=round(float(np.nanmin(ys)), 1),
            max_deg=round(float(np.nanmax(ys)), 1), rom_deg=round(float(p95 - p5), 1),
            noise_deg=round(noise, 2), bouts=n_bouts,
        ))

    # Segment check for hip bouts: the hip angle is thigh relative to trunk, so it changes when
    # EITHER segment moves. Compare how much the thigh segment and the trunk segment each turned.
    series = {}
    for name, raw in up.joints.items():
        y = np.array([np.nan if v is None else v for v in raw], float)
        series[name] = _smooth(_resample(t, y, fs)[1], smooth_n)
    grid_all = np.arange(t[0], t[-1], 1.0 / fs)
    trunk = series.get("axial.trunk_flexion")
    for b in bouts:
        if "hip_flexion" not in b.joint:
            continue
        side = b.joint.split(".")[0]
        thigh = series.get(f"{side}.thigh_incl")
        if thigh is None or trunk is None:
            continue
        i0, i1 = np.searchsorted(grid_all, [b.start, b.end])
        def turn(y):
            seg = y[max(0, i0 - int(0.2 * fs)):i1 + 1]
            seg = seg[np.isfinite(seg)]
            return float(seg.max() - seg.min()) if seg.size else 0.0
        dt, dth = turn(trunk), turn(thigh)
        b.trunk_turn_deg, b.thigh_turn_deg = round(dt, 1), round(dth, 1)
        b.driver = "trunk" if dt > 1.5 * dth else "thigh" if dth > 1.5 * dt else "both"

    # First mover: group bouts that overlap in time; the earliest onset leads.
    bouts.sort(key=lambda b: b.start)
    group: list[MovementBout] = []
    group_end = -1.0

    def close(g: list[MovementBout]):
        if len(g) < 2:
            return
        lead = g[0]
        second = next((x for x in g[1:] if x.joint != lead.joint), None)
        lead_ms = None if second is None else round((second.start - lead.start) * 1000, 0)
        for x in g:
            x.first_mover = lead.joint
            x.lead_ms = lead_ms
            x.axial_led = bool(lead.joint.startswith(AXIAL_PREFIX) and not x.joint.startswith(AXIAL_PREFIX)
                               and lead_ms is not None and lead_ms > LEAD_FLAG_MS)

    for b in bouts:
        if b.start > group_end:
            close(group)
            group, group_end = [b], b.end
        else:
            group.append(b)
            group_end = max(group_end, b.end)
    close(group)

    not_assessed = [s.joint for s in summaries if s.visible_fraction < MIN_VISIBLE or s.rom_deg is None]
    assessed_regions = {region(s.joint) for s in summaries if s.joint not in not_assessed}
    hidden_regions = sorted({region(j) for j in not_assessed} - assessed_regions)

    legs = [b for b in bouts if region(b.joint) == "legs"]
    lines = [
        f"{dur / 60:.1f} min recorded at about {fps_raw:.0f} frames/s; {len(bouts)} movement bouts across "
        f"{sum(1 for s in summaries if s.bouts)} channels.",
    ]
    if hidden_regions:
        lines.append(
            "Not visible enough to assess: " + ", ".join(hidden_regions)
            + " (unknown, which is not the same as no movement)."
        )
    if "legs" in assessed_regions:
        lines.append(
            f"Leg bouts: {len(legs)}; led by trunk or pelvis: {sum(b.axial_led for b in legs)}; "
            f"hip bouts where the trunk, not the thigh, did the turning: {sum(b.driver == 'trunk' for b in bouts)}."
        )
    lines.append(f"Clonus-like oscillations (4.5-8.5 Hz): {sum(b.clonus_like for b in bouts)}.")
    n_slow = sum(b.slow for b in bouts)
    if n_slow:
        lines.append(f"Slow movements (drift from rest, below the speed threshold): {n_slow}.")
    total = sum(len(v) for v in up.joints.values()) or 1
    if glitches or rejected:
        lines.append(
            f"Tracking quality: {glitches} glitch frames removed ({100 * glitches / total:.1f}% of samples)"
            + (f", {rejected} impossible movements discarded" if rejected else "") + "."
        )
    lines.append(f"Sensitivity: {up.sensitivity}.")
    peak: dict[str, float] = {}
    for b in bouts:
        peak[b.joint] = max(peak.get(b.joint, 0.0), b.excursion_deg)
    top = sorted(peak.items(), key=lambda kv: -kv[1])[:5]
    if top:
        lines.append("Biggest single movements: " + ", ".join(f"{j} {v:.0f} {unit(j)}" for j, v in top) + ".")

    return SessionResult(
        duration_s=round(dur, 2), fps=round(fps_raw, 1), joints=summaries, bouts=bouts,
        summary=" ".join(lines),
        not_assessed=not_assessed,
        caveats=[
            "Angles come from a single camera; out-of-plane joints (hip rotation, ankle inversion) are unreliable.",
            "Detecting a movement does not show it was voluntary. Pair with timed cues, EMG or a clinical exam.",
            "Bouts led by the trunk or pelvis can move the legs without leg muscle activity.",
        ],
    )
