"""A stand-in pose estimator for tests and demos without model downloads.

Produces one person seen from the side, facing left, lifting the left knee and bending the left
elbow slowly, with a hand that opens and closes. Set LIVE_ESTIMATOR=synthetic to use it.
"""

from __future__ import annotations

import math
import time

import numpy as np

from live.points import HAND_START, KEYPOINT_COUNT, LAYOUTS


class SyntheticEstimator:
    def __init__(self, model: str):
        self.model = model
        self.t0 = time.monotonic()

    def __call__(self, image: np.ndarray):
        h, w = image.shape[:2]
        t = time.monotonic() - self.t0
        k = np.zeros((KEYPOINT_COUNT[self.model], 2))
        sc = np.zeros(KEYPOINT_COUNT[self.model])
        L = LAYOUTS[self.model]
        u = h / 4.0  # body scale
        cx, hip_y = w * 0.5, h * 0.55

        def put(name, x, y):
            if name in L:
                k[L[name]] = (x, y)
                sc[L[name]] = 0.9

        lift = 35 * (0.5 - 0.5 * math.cos(2 * math.pi * t / 4.0))   # hip flexion 0..35 deg
        bend = 80 * (0.5 - 0.5 * math.cos(2 * math.pi * t / 3.0))   # elbow 0..80 deg
        for s, dx in (("l", -3), ("r", 3)):
            hip = np.array([cx + dx, hip_y])
            sh = hip + np.array([0, -1.2 * u])
            put(f"{s}_hip", *hip)
            put(f"{s}_shoulder", *sh)
            a = math.radians(lift if s == "l" else 0)
            knee = hip + 0.9 * u * np.array([-math.sin(a), math.cos(a)])
            ankle = knee + np.array([0, 0.85 * u])
            put(f"{s}_knee", *knee)
            put(f"{s}_ankle", *ankle)
            put(f"{s}_heel", *(ankle + np.array([0.08 * u, 0.08 * u])))
            put(f"{s}_bigtoe", *(ankle + np.array([-0.3 * u, 0.1 * u])))
            put(f"{s}_smalltoe", *(ankle + np.array([-0.25 * u, 0.1 * u])))
            el = sh + np.array([0, 0.6 * u])
            b = math.radians(bend if s == "l" else 10)
            wr = el + 0.55 * u * np.array([-math.sin(b), math.cos(b)])
            put(f"{s}_elbow", *el)
            put(f"{s}_wrist", *wr)
            if s in HAND_START.get(self.model, {}):
                start = HAND_START[self.model][s]
                f = math.radians(60 * (0.5 - 0.5 * math.cos(2 * math.pi * t / 2.5)))
                axis = (wr - el) / max(1e-6, np.linalg.norm(wr - el))
                across = np.array([axis[1], -axis[0]])
                k[start] = wr
                for fi, off in enumerate((0.6, 0.3, 0.0, -0.3, -0.6)):
                    base = wr + 0.35 * u * axis * (0.5 if fi == 0 else 1) + 0.08 * u * off * across
                    d = axis * math.cos(f) + across * math.sin(f) * (-1 if fi else 1)
                    p1 = base + 0.15 * u * d
                    p2 = p1 + 0.1 * u * d
                    p3 = p2 + 0.08 * u * d
                    idx = start + 1 + 4 * fi
                    k[idx:idx + 4] = (base, p1, p2, p3)
                sc[start:start + 21] = 0.9
        put("nose", cx - 0.25 * u, hip_y - 1.55 * u)
        put("l_ear", cx + 0.02 * u, hip_y - 1.6 * u)
        put("r_ear", cx + 0.04 * u, hip_y - 1.6 * u)
        put("l_eye", cx - 0.18 * u, hip_y - 1.62 * u)
        put("r_eye", cx - 0.17 * u, hip_y - 1.62 * u)
        return k[None], sc[None]
