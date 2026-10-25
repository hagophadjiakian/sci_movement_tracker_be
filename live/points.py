"""Turn RTMPose keypoint arrays into named points.

Supported layouts (rtmlib solutions):
- body       COCO-17 (no feet, no hands)
- body_feet  Halpe-26 (adds big toe, small toe, heel per foot)
- wholebody  COCO-WholeBody-133 (adds feet, face and 21 points per hand)

"l"/"r" are the person's own left and right. A point is kept only when its score passes the
threshold and it lies inside the picture; otherwise it is None. Nothing is guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

COCO_BODY = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]
_COCO = {n: i for i, n in enumerate(COCO_BODY)}

LAYOUTS: dict[str, dict[str, int]] = {
    "body": dict(_COCO),
    "body_feet": {**_COCO, "l_bigtoe": 20, "r_bigtoe": 21, "l_smalltoe": 22, "r_smalltoe": 23, "l_heel": 24, "r_heel": 25},
    "wholebody": {**_COCO, "l_bigtoe": 17, "l_smalltoe": 18, "l_heel": 19, "r_bigtoe": 20, "r_smalltoe": 21, "r_heel": 22},
}
KEYPOINT_COUNT = {"body": 17, "body_feet": 26, "wholebody": 133}
# First index of each 21-point hand (same order as MediaPipe: wrist, thumb 1-4, index 5-8, ... little 17-20).
HAND_START = {"wholebody": {"l": 91, "r": 112}}

SCORE_THR = 0.35
MIN_HAND_FRACTION = 0.03  # wrist-to-middle-knuckle length vs image diagonal
EDGE = 0.01


@dataclass
class Points:
    width: int
    height: int
    named: dict[str, np.ndarray | None] = field(default_factory=dict)
    hands: dict[str, list[np.ndarray] | None] = field(default_factory=lambda: {"l": None, "r": None})

    def get(self, name: str) -> np.ndarray | None:
        return self.named.get(name)

    def has(self, *names: str) -> bool:
        return all(self.named.get(n) is not None for n in names)


def _inside(xy: np.ndarray, w: int, h: int) -> bool:
    return EDGE * w < xy[0] < (1 - EDGE) * w and EDGE * h < xy[1] < (1 - EDGE) * h


def to_points(kpts: np.ndarray, scores: np.ndarray, layout: str, width: int, height: int,
              thr: float = SCORE_THR) -> Points:
    """kpts: (K, 2) pixel coordinates for ONE person; scores: (K,)."""
    pts = Points(width=width, height=height)
    for name, i in LAYOUTS[layout].items():
        if i < len(kpts) and scores[i] >= thr and _inside(kpts[i], width, height):
            pts.named[name] = np.asarray(kpts[i], dtype=float)
        else:
            pts.named[name] = None
    for side, start in HAND_START.get(layout, {}).items():
        hand = [np.asarray(kpts[start + j], dtype=float) for j in range(21)]
        ok = [scores[start + j] >= thr and _inside(hand[j], width, height) for j in range(21)]
        # Use the hand only when the wrist and most finger points are confidently found, and the hand
        # is big enough in the picture for finger angles to mean anything (real videos: small, distant
        # hands gave nonsense finger angles).
        hand_len = float(np.linalg.norm(hand[9] - hand[0]))
        big_enough = hand_len >= MIN_HAND_FRACTION * float(np.hypot(width, height))
        pts.hands[side] = hand if ok[0] and sum(ok) >= 17 and big_enough else None
    return pts


def pick_person(keypoints: np.ndarray, scores: np.ndarray, previous_center: np.ndarray | None,
                thr: float = SCORE_THR) -> int | None:
    """Choose which detected person to follow.

    Stays with the person nearest the one followed last frame (so a helper walking through the picture
    does not steal the tracking); otherwise takes the biggest confidently-detected person.
    """
    if keypoints is None or len(keypoints) == 0:
        return None
    body = slice(0, 17)
    candidates = []
    for i in range(len(keypoints)):
        good = scores[i][body] >= thr
        if good.sum() < 5:
            continue
        k = keypoints[i][body][good]
        size = float((k[:, 0].max() - k[:, 0].min()) * (k[:, 1].max() - k[:, 1].min()))
        center = k.mean(axis=0)
        candidates.append((i, size, center))
    if not candidates:
        return None
    if previous_center is not None:
        i, _, c = min(candidates, key=lambda c: float(np.linalg.norm(c[2] - previous_center)))
        biggest = max(candidates, key=lambda c: c[1])
        # Keep following unless that person jumped implausibly far (then they were probably lost).
        span = max(1.0, np.sqrt(biggest[1]))
        if np.linalg.norm(c - previous_center) < 0.5 * span:
            return i
    return max(candidates, key=lambda c: c[1])[0]
