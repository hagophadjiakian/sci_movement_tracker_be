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

    summaries: list[JointSummary] = []
    bouts: list[MovementBout] = []

    for name, raw in up.joints.items():
        y = np.array([np.nan if v is None else v for v in raw], float)
        vis = float(np.isfinite(y).mean())
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
        floor = 5.0 if name.startswith(AXIAL_PREFIX) or "_incl" in name else 10.0
        sp_thr = max(floor, float(np.median(sp) + 5 * 1.4826 * np.median(np.abs(sp - np.median(sp)))))
        moving = np.nan_to_num(speed) > sp_thr

        # Walk each onset back to where the joint really started moving (speed < 25% of threshold),
        # then merge runs separated by a pause where the joint stays away from its baseline
        # (e.g. the still moment at the top of a knee lift: up-hold-down is one movement).
        runs = _runs(moving)
        exc_thr = max(3 * noise, MIN_EXCURSION)
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
            dev = seg - base
            if not np.isfinite(dev).any():
                continue
            k = int(np.nanargmax(np.abs(dev)))
            exc = float(abs(dev[k]))
            if exc < exc_thr:
                continue
            hz = _dominant_hz(seg, fs)
            n_bouts += 1
            bouts.append(MovementBout(
                joint=name, start=float(grid[a]), end=float(grid[min(b, len(grid) - 1)]),
                excursion_deg=round(exc, 1), peak_speed_dps=round(float(np.nanmax(speed[a:b])), 1),
                direction="increase" if dev[k] > 0 else "decrease",
                oscillation_hz=None if hz is None else round(hz, 1),
                clonus_like=bool(hz is not None and 4.5 <= hz <= 8.5 and (b - a) / fs >= 0.6),
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

    legs = [b for b in bouts if any(j in b.joint for j in ("hip_", "knee_", "ankle_"))]
    lines = [
        f"{dur / 60:.1f} min recorded at about {fps_raw:.0f} frames/s; {len(bouts)} movement bouts across "
        f"{sum(1 for s in summaries if s.bouts)} joints.",
        f"Leg bouts: {len(legs)}; led by trunk or pelvis: {sum(b.axial_led for b in legs)}; "
        f"clonus-like (4.5-8.5 Hz): {sum(b.clonus_like for b in bouts)}; "
        f"hip bouts where the trunk, not the thigh, did the turning: {sum(b.driver == 'trunk' for b in bouts)}.",
    ]
    peak: dict[str, float] = {}
    for b in bouts:
        peak[b.joint] = max(peak.get(b.joint, 0.0), b.excursion_deg)
    top = sorted(peak.items(), key=lambda kv: -kv[1])[:5]
    if top:
        lines.append("Biggest single movements: " + ", ".join(f"{j} {v:.0f} deg" for j, v in top) + ".")

    return SessionResult(
        duration_s=round(dur, 2), fps=round(fps_raw, 1), joints=summaries, bouts=bouts,
        summary=" ".join(lines),
        caveats=[
            "Angles come from a single camera; out-of-plane joints (hip rotation, ankle inversion) are unreliable.",
            "Detecting a movement does not show it was voluntary. Pair with timed cues, EMG or a clinical exam.",
            "Bouts led by the trunk or pelvis can move the legs without leg muscle activity.",
        ],
    )
