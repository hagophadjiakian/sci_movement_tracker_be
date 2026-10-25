"""Keep left and right from swapping between frames (Python version of src/lib/sideTracker.ts).

If the "left" points suddenly jump to where the "right" points just were, and vice versa, the model
has exchanged the two limbs; swap them back. Hands follow the arms, feet follow the legs.
"""

from __future__ import annotations

import numpy as np

from live.points import Points

ARM_PAIRS = [("l_elbow", "r_elbow"), ("l_wrist", "r_wrist")]
LEG_PAIRS = [("l_knee", "r_knee"), ("l_ankle", "r_ankle"), ("l_bigtoe", "r_bigtoe"),
             ("l_smalltoe", "r_smalltoe"), ("l_heel", "r_heel")]
ARM_KEYS = [("l_elbow", "r_elbow"), ("l_wrist", "r_wrist")]
LEG_KEYS = [("l_knee", "r_knee"), ("l_ankle", "r_ankle")]


class SideTracker:
    def __init__(self):
        self.prev: dict[str, np.ndarray | None] | None = None
        self.swaps = 0

    def reset(self):
        self.prev = None

    def _should_swap(self, p: Points, keys) -> bool:
        if self.prev is None:
            return False
        scale = max(p.width, p.height)
        same = crossed = 0.0
        for l, r in keys:
            cl, cr, pl, pr = p.get(l), p.get(r), self.prev.get(l), self.prev.get(r)
            if any(x is None for x in (cl, cr, pl, pr)):
                return False
            same += np.linalg.norm(cl - pl) + np.linalg.norm(cr - pr)
            crossed += np.linalg.norm(cl - pr) + np.linalg.norm(cr - pl)
        same /= scale
        crossed /= scale
        return same > 0.08 and crossed < 0.5 * same

    def fix(self, p: Points) -> tuple[bool, bool]:
        arms = self._should_swap(p, ARM_KEYS)
        if arms:
            for l, r in ARM_PAIRS:
                p.named[l], p.named[r] = p.named.get(r), p.named.get(l)
            p.hands["l"], p.hands["r"] = p.hands.get("r"), p.hands.get("l")
        legs = self._should_swap(p, LEG_KEYS)
        if legs:
            for l, r in LEG_PAIRS:
                p.named[l], p.named[r] = p.named.get(r), p.named.get(l)
        if arms or legs:
            self.swaps += 1
        self.prev = dict(p.named)
        return arms, legs
