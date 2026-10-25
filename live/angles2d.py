"""2D joint and segment angles from one camera (RTMPose keypoints).

One camera sees one plane. So the meaning of an angle depends on where the camera is:
- side view: flexion/extension (shoulder, hip, knee, ankle, trunk forward lean, neck nod)
- front view: abduction, side lean, pelvic obliquity, neck side bend, shoulder shrug
Elbow and knee bends, fingers and pinch opening are computed in both views (shortened when the
movement goes toward the camera). Rotations (shoulder, hip, forearm, ankle inversion) cannot be
measured in 2D and stay empty; they need several cameras.

Channel names match the browser engine (src/lib/angles.ts) so the screen, recorder and analysis
work with either engine. Image coordinates: x to the right, y down (pixels).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from live.points import Points

View = Literal["side", "front", "auto"]
UP = np.array([0.0, -1.0])
DOWN = np.array([0.0, 1.0])

FINGER_IDX = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "little": (17, 18, 19, 20),
}
FINGER_JOINTS = {"thumb": ("cmc", "mcp", "ip"), "index": ("mcp", "pip", "dip"), "middle": ("mcp", "pip", "dip"),
                 "ring": ("mcp", "pip", "dip"), "little": ("mcp", "pip", "dip")}
SIDES = (("left", "l"), ("right", "r"))


def _deg(r: float) -> float:
    return r * 180.0 / math.pi


def ang(a: np.ndarray, b: np.ndarray) -> float | None:
    """Unsigned angle between two 2D vectors, degrees."""
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    if n < 1e-9:
        return None
    return _deg(math.acos(max(-1.0, min(1.0, float(np.dot(a, b)) / n))))


def sang(a: np.ndarray, b: np.ndarray) -> float | None:
    """Signed angle from a to b in image coordinates (+ = clockwise on screen)."""
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        return None
    return _deg(math.atan2(a[0] * b[1] - a[1] * b[0], float(np.dot(a, b))))


def _mid(*pts: np.ndarray | None) -> np.ndarray | None:
    ok = [p for p in pts if p is not None]
    return np.mean(ok, axis=0) if ok else None


def detect_view(p: Points) -> Literal["side", "front"] | None:
    """Front (or at the feet of someone lying) vs side, from how wide the body looks. None = can't tell."""
    sh_l, sh_r = p.get("l_shoulder"), p.get("r_shoulder")
    hip_l, hip_r = p.get("l_hip"), p.get("r_hip")
    hip = _mid(hip_l, hip_r)
    if sh_l is not None and sh_r is not None:
        width = float(np.linalg.norm(sh_l - sh_r))
        ear = _mid(p.get("l_ear"), p.get("r_ear"))
        ref = None
        if hip is not None:
            ref = float(np.linalg.norm(_mid(sh_l, sh_r) - hip))
        elif ear is not None:
            ref = 2.5 * float(np.linalg.norm(_mid(sh_l, sh_r) - ear))
        if ref:
            return "front" if width / ref > 0.35 else "side"
    # Shoulders cut off (e.g. camera at the feet, close up): use hip width against thigh length.
    if hip_l is not None and hip_r is not None:
        thighs = [np.linalg.norm(p.get(f"{s}_knee") - p.get(f"{s}_hip"))
                  for s in ("l", "r") if p.get(f"{s}_knee") is not None and p.get(f"{s}_hip") is not None]
        if thighs:
            return "front" if float(np.linalg.norm(hip_l - hip_r)) / float(np.mean(thighs)) > 0.25 else "side"
    return None


class ViewSelector:
    """Keeps the camera view stable: with view='auto' it changes only after `hold` frames in a row
    disagree, so the meaning of the angles does not flip from frame to frame."""

    def __init__(self, fixed: View = "auto", hold: int = 20):
        self.fixed = fixed
        self.hold = hold
        self.current: Literal["side", "front"] | None = None if fixed == "auto" else fixed  # type: ignore[assignment]
        self._streak = 0

    def update(self, p: Points) -> Literal["side", "front"]:
        if self.fixed != "auto":
            return self.fixed  # type: ignore[return-value]
        raw = detect_view(p)
        if raw is None:
            self._streak = 0
            return self.current or "front"
        if self.current is None:
            self.current = raw
        elif raw != self.current:
            self._streak += 1
            if self._streak >= self.hold:
                self.current, self._streak = raw, 0
        else:
            self._streak = 0
        return self.current


def detect_facing(p: Points) -> Literal["left", "right"] | None:
    """Which way the person faces on screen (side view)."""
    nose = p.get("nose")
    ear = _mid(p.get("l_ear"), p.get("r_ear"))
    if nose is not None and ear is not None and abs(nose[0] - ear[0]) > 2:
        return "left" if nose[0] < ear[0] else "right"
    votes = 0
    for s in ("l", "r"):
        toe, heel = p.get(f"{s}_bigtoe"), p.get(f"{s}_heel")
        if toe is not None and heel is not None:
            votes += 1 if toe[0] < heel[0] else -1
    if votes:
        return "left" if votes > 0 else "right"
    return None


def compute_2d(p: Points, view: View = "auto") -> tuple[dict[str, float], dict]:
    vals: dict[str, float | None] = {}
    v = (detect_view(p) or "front") if view == "auto" else view
    facing = detect_facing(p) if v == "side" else None
    meta = {
        "engine": "rtmpose",
        "dims": "2d",
        "view": v,
        "facing": facing,
        "frame": None,
        "hand": {"left": None, "right": None},
    }

    sh_mid = _mid(p.get("l_shoulder"), p.get("r_shoulder"))
    hip_mid = _mid(p.get("l_hip"), p.get("r_hip"))
    trunk = (sh_mid - hip_mid) if (sh_mid is not None and hip_mid is not None) else None
    meta["frame"] = "trunk" if trunk is not None else ("shoulders" if sh_mid is not None else None)

    def put(k: str, x: float | None):
        if x is not None and math.isfinite(x):
            vals[k] = round(x, 1)

    # ---- both views: elbow and knee bend -----------------------------------
    for side, s in SIDES:
        sh, el, wr = p.get(f"{s}_shoulder"), p.get(f"{s}_elbow"), p.get(f"{s}_wrist")
        if sh is not None and el is not None and wr is not None:
            a = ang(sh - el, wr - el)
            put(f"{side}.elbow_flexion", None if a is None else 180 - a)
        hp, kn, an = p.get(f"{s}_hip"), p.get(f"{s}_knee"), p.get(f"{s}_ankle")
        if hp is not None and kn is not None and an is not None:
            a = ang(hp - kn, an - kn)
            put(f"{side}.knee_flexion", None if a is None else 180 - a)

    if v == "side" and facing is not None:
        sgn = 1.0 if facing == "left" else -1.0
        if trunk is not None:
            tf = sang(DOWN, -trunk)
            if tf is not None:
                put("axial.trunk_flexion", -sgn * tf)
                put("axial.trunk_lean", abs(tf))
        for side, s in SIDES:
            sh, el = p.get(f"{s}_shoulder"), p.get(f"{s}_elbow")
            if trunk is not None and sh is not None and el is not None:
                x = sang(-trunk, el - sh)
                put(f"{side}.shoulder_flexion", None if x is None else sgn * x)
            hp, kn, an = p.get(f"{s}_hip"), p.get(f"{s}_knee"), p.get(f"{s}_ankle")
            if hp is not None and kn is not None:
                thigh = kn - hp
                if trunk is not None:
                    x = sang(-trunk, thigh)
                    put(f"{side}.hip_flexion", None if x is None else sgn * x)
                x = sang(DOWN, thigh)
                put(f"{side}.thigh_incl", None if x is None else sgn * x)
            if kn is not None and an is not None:
                x = sang(DOWN, an - kn)
                put(f"{side}.shank_incl", None if x is None else sgn * x)
            toe, heel = p.get(f"{s}_bigtoe"), p.get(f"{s}_heel")
            if kn is not None and an is not None and toe is not None and heel is not None:
                a = ang(kn - an, toe - heel)
                put(f"{side}.ankle_dorsiflexion", None if a is None else 90 - a)
        nose = p.get("nose")
        ear = _mid(p.get("l_ear"), p.get("r_ear"))
        if trunk is not None and nose is not None and ear is not None:
            x = sang(trunk, nose - ear)
            put("axial.neck_flexion", None if x is None else -sgn * x - 90)

    if v == "front":
        if trunk is not None:
            lean = _deg(math.atan2(-trunk[0], -trunk[1]))
            put("axial.trunk_lateral_lean", lean)
            put("axial.trunk_lean", abs(lean))
        for side, s in SIDES:
            out_sign = -1.0 if side == "left" else 1.0  # person's left is on the image right
            sh, el = p.get(f"{s}_shoulder"), p.get(f"{s}_elbow")
            if trunk is not None and sh is not None and el is not None:
                x = sang(-trunk, el - sh)
                put(f"{side}.shoulder_abduction", None if x is None else out_sign * x)
            hp, kn = p.get(f"{s}_hip"), p.get(f"{s}_knee")
            if trunk is not None and hp is not None and kn is not None:
                x = sang(-trunk, kn - hp)
                put(f"{side}.hip_abduction", None if x is None else out_sign * x)
        hl, hr = p.get("l_hip"), p.get("r_hip")
        if hl is not None and hr is not None:
            put("axial.pelvic_obliquity", _deg(math.atan2(hl[1] - hr[1], hl[0] - hr[0])))
        sl, sr = p.get("l_shoulder"), p.get("r_shoulder")
        el_, er_ = p.get("l_ear"), p.get("r_ear")
        nose = p.get("nose")
        if sl is not None and sr is not None:
            width = float(np.linalg.norm(sl - sr))
            if el_ is not None and er_ is not None:
                head = _deg(math.atan2(-(el_[1] - er_[1]), el_[0] - er_[0]))
                shl = _deg(math.atan2(-(sl[1] - sr[1]), sl[0] - sr[0]))
                put("axial.neck_lateral_flexion", head - shl)
                d = float(np.linalg.norm(el_ - er_))
                if nose is not None and d > 1e-6:
                    ratio = -(nose[0] - (el_[0] + er_[0]) / 2) / (0.6 * d)
                    put("axial.neck_rotation", _deg(math.asin(max(-1.0, min(1.0, ratio)))))
            for side, s in SIDES:
                ear_s, sh_s = p.get(f"{s}_ear"), p.get(f"{s}_shoulder")
                if ear_s is not None and sh_s is not None and width > 1e-6:
                    put(f"{side}.shoulder_elevation", -100.0 * (sh_s[1] - ear_s[1]) / width)

    # ---- hands (wholebody only), both views -------------------------------
    for side, s in SIDES:
        h = p.hands.get(s)
        if h is None:
            continue
        meta["hand"][side] = "hand"
        el, wr = p.get(f"{s}_elbow"), p.get(f"{s}_wrist")
        if el is not None and wr is not None:
            put(f"{side}.wrist_flexion", ang(wr - el, h[9] - h[0]))  # unsigned in 2D
        for f, (a, b, c, d) in FINGER_IDX.items():
            segs = [h[a] - h[0], h[b] - h[a], h[c] - h[b], h[d] - h[c]]
            for j, name in enumerate(FINGER_JOINTS[f]):
                put(f"{side}.{f}_{name}", ang(segs[j], segs[j + 1]))
        put(f"{side}.thumb_abduction", ang(h[2] - h[1], h[5] - h[0]))
        put(f"{side}.spread_index_middle", ang(h[6] - h[5], h[10] - h[9]))
        put(f"{side}.spread_middle_ring", ang(h[10] - h[9], h[14] - h[13]))
        put(f"{side}.spread_ring_little", ang(h[14] - h[13], h[18] - h[17]))
        hand_len = float(np.linalg.norm(h[9] - h[0]))
        if hand_len > 1e-6:
            put(f"{side}.pinch_aperture", 100.0 * float(np.linalg.norm(h[4] - h[8])) / hand_len)

    meta["unsigned"] = ["wrist_flexion", "fingers"]
    return {k: x for k, x in vals.items() if x is not None}, meta
