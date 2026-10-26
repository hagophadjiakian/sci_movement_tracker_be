"""Synthetic 2D poses with known angles -> live.angles2d must return them with the right sign."""

import math

import numpy as np
import pytest

from live.angles2d import compute_2d
from live.points import HAND_START, KEYPOINT_COUNT, LAYOUTS, to_points

W, H = 1280, 720
UP = np.array([0.0, -1.0])


def unit(v):
    return v / np.linalg.norm(v)


def rot_toward(a, b, deg):
    """Direction a turned toward direction b by deg (in the plane they span)."""
    a = unit(a)
    bp = b - np.dot(b, a) * a
    bp = unit(bp)
    r = math.radians(deg)
    return math.cos(r) * a + math.sin(r) * bp


class Pose:
    def __init__(self):
        self.k = np.zeros((KEYPOINT_COUNT["wholebody"], 2))
        self.s = np.zeros(KEYPOINT_COUNT["wholebody"])

    def put(self, name, xy):
        i = LAYOUTS["wholebody"][name]
        self.k[i] = xy
        self.s[i] = 0.9

    def hand(self, side, wrist, axis, mcp=0.0, pip=0.0, open_thumb=30.0):
        start = HAND_START["wholebody"][side]
        axis = unit(axis)
        across = np.array([axis[1], -axis[0]])
        self.k[start] = wrist
        # Seen edge-on (side view of the hand), the finger fan points toward the camera: no in-plane offset.
        for fi, off in enumerate((0, 0.0, 0.0, 0.0, 0.0)):
            if fi == 0:
                base = wrist + 20 * axis + 15 * across
                d = rot_toward(axis, across, open_thumb)
            else:
                base = wrist + 60 * axis + 25 * off * across
                d = axis
            d1 = rot_toward(d, -across if fi else across, mcp if fi else 0)
            d2 = rot_toward(d1, -across, pip if fi else 0)
            p1 = base + 25 * d1
            p2 = p1 + 18 * d2
            p3 = p2 + 14 * d2
            idx = start + 1 + 4 * fi
            self.k[idx:idx + 4] = (base, p1, p2, p3)
        self.s[start:start + 21] = 0.9

    def compute(self, view):
        pts = to_points(self.k, self.s, "wholebody", W, H)
        return compute_2d(pts, view)


def side_pose(facing="left", trunk=0.0, sh=0.0, elbow=0.0, hip=0.0, knee=0.0, dorsi=0.0, nod=0.0,
              wrist=0.0, mcp=0.0, pip=0.0):
    F = np.array([-1.0, 0.0]) if facing == "left" else np.array([1.0, 0.0])
    p = Pose()
    hip_pt = np.array([W / 2, H * 0.55])
    T = rot_toward(UP, F, trunk)
    Fp = unit(F - np.dot(F, T) * T)
    sh_pt = hip_pt + 180 * T
    for s, dx in (("l", -2.0), ("r", 2.0)):
        p.put(f"{s}_hip", hip_pt + [dx, 0])
        p.put(f"{s}_shoulder", sh_pt + [dx, 0])
    arm = rot_toward(-T, Fp, sh)
    el = sh_pt + 90 * arm
    fore = rot_toward(arm, rot_toward(arm, Fp if abs(sh) < 80 else T, 90), elbow) if elbow else arm
    wr = el + 80 * fore
    thigh = rot_toward(-T, Fp, hip)
    kn = hip_pt + 130 * thigh
    fwd_of_thigh = rot_toward(thigh, Fp, 90) if abs(hip) < 80 else rot_toward(thigh, T, 90)
    shank = rot_toward(thigh, -fwd_of_thigh, knee) if knee else thigh
    an = kn + 120 * shank
    foot_fwd = rot_toward(shank, Fp if abs(np.dot(shank, Fp)) < 0.9 else T, 90)
    foot = rot_toward(foot_fwd, -shank, dorsi) if dorsi else foot_fwd
    for s in ("l", "r"):
        p.put(f"{s}_elbow", el)
        p.put(f"{s}_wrist", wr)
        p.put(f"{s}_knee", kn)
        p.put(f"{s}_ankle", an)
        p.put(f"{s}_heel", an - 15 * foot + 8 * shank)
        p.put(f"{s}_bigtoe", an + 45 * foot + 8 * shank)
        p.put(f"{s}_smalltoe", an + 40 * foot + 8 * shank)
    ear = sh_pt + 45 * T
    nose_dir = rot_toward(Fp, -T, nod) if nod else Fp
    p.put("l_ear", ear)
    p.put("nose", ear + 25 * nose_dir)
    hand_axis = rot_toward(fore, rot_toward(fore, Fp, 90), wrist) if wrist else fore
    p.hand("l", wr, hand_axis, mcp=mcp, pip=pip)
    return p


@pytest.mark.parametrize("facing", ["left", "right"])
@pytest.mark.parametrize("kw,expect", [
    ({}, {"axial.trunk_flexion": 0, "left.shoulder_flexion": 0, "left.hip_flexion": 0, "left.knee_flexion": 0,
          "left.ankle_dorsiflexion": 0, "axial.neck_flexion": 0}),
    ({"trunk": 25}, {"axial.trunk_flexion": 25, "left.hip_flexion": 0, "left.thigh_incl": -25}),
    ({"sh": 70}, {"left.shoulder_flexion": 70}),
    ({"sh": -30}, {"left.shoulder_flexion": -30}),
    ({"elbow": 90}, {"left.elbow_flexion": 90}),
    ({"hip": 40}, {"left.hip_flexion": 40, "left.thigh_incl": 40}),
    ({"trunk": 30, "hip": 30}, {"left.hip_flexion": 30, "left.thigh_incl": 0, "axial.trunk_flexion": 30}),
    ({"hip": 90, "knee": 90}, {"left.hip_flexion": 90, "left.knee_flexion": 90, "left.shank_incl": 0}),
    ({"dorsi": 15}, {"left.ankle_dorsiflexion": 15}),
    ({"dorsi": -25}, {"left.ankle_dorsiflexion": -25}),
    ({"nod": 25}, {"axial.neck_flexion": 25}),
    ({"elbow": 60, "wrist": 35}, {"left.wrist_flexion": 35}),
    ({"mcp": 50, "pip": 40}, {"left.index_mcp": 50, "left.index_pip": 40, "left.little_mcp": 50}),
])
def test_side_view(facing, kw, expect):
    vals, meta = side_pose(facing, **kw).compute("side")
    assert meta["view"] == "side" and meta["facing"] == facing
    for k, want in expect.items():
        assert k in vals, f"{k} missing"
        assert vals[k] == pytest.approx(want, abs=1.5), f"{k}: {vals[k]} != {want}"


def front_pose(lean_r=0.0, abd_l=0.0, abd_r=0.0, hip_abd_l=0.0, hip_abd_r=0.0, oblq=0.0, tilt_r=0.0,
               turn_r=0.0, shrug_l=0.0):
    p = Pose()
    hip_mid = np.array([W / 2, H * 0.6])
    T = rot_toward(UP, np.array([-1.0, 0.0]), lean_r)  # leaning to the person's right = image left
    left_dir = unit(np.array([-T[1], T[0]]) * -1)  # perpendicular to T toward image right
    if left_dir[0] < 0:
        left_dir = -left_dir
    sh_mid = hip_mid + 200 * T
    shL, shR = sh_mid + 70 * left_dir, sh_mid - 70 * left_dir
    shL = shL + shrug_l * 0.01 * 140 * T
    p.put("l_shoulder", shL)
    p.put("r_shoulder", shR)
    hl_dir = np.array([math.cos(math.radians(oblq)), math.sin(math.radians(oblq))])
    hipL, hipR = hip_mid + 45 * hl_dir, hip_mid - 45 * hl_dir
    p.put("l_hip", hipL)
    p.put("r_hip", hipR)
    for s, shp, a, out in (("l", shL, abd_l, left_dir), ("r", shR, abd_r, -left_dir)):
        arm = rot_toward(-T, out, a) if a else -T
        el = shp + 100 * arm
        p.put(f"{s}_elbow", el)
        p.put(f"{s}_wrist", el + 90 * arm)
    for s, hp, a, out in (("l", hipL, hip_abd_l, left_dir), ("r", hipR, hip_abd_r, -left_dir)):
        th = rot_toward(-T, out, a) if a else -T
        kn = hp + 150 * th
        p.put(f"{s}_knee", kn)
        p.put(f"{s}_ankle", kn + 140 * th)
    ear_mid = sh_mid + 70 * T
    sh_ang = math.degrees(math.atan2(-(shL[1] - shR[1]), shL[0] - shR[0]))
    b = math.radians(sh_ang + tilt_r)
    lat = np.array([math.cos(b), -math.sin(b)])
    p.put("l_ear", ear_mid + 30 * lat)
    p.put("r_ear", ear_mid - 30 * lat)
    p.put("nose", ear_mid - np.array([0.6 * 60 * math.sin(math.radians(turn_r)), 10]))
    return p


@pytest.mark.parametrize("kw,expect", [
    ({}, {"axial.trunk_lateral_lean": 0, "left.shoulder_abduction": 0, "axial.pelvic_obliquity": 0,
          "axial.neck_lateral_flexion": 0, "axial.neck_rotation": 0}),
    ({"lean_r": 12}, {"axial.trunk_lateral_lean": 12, "left.shoulder_abduction": 0, "right.shoulder_abduction": 0}),
    ({"lean_r": -10}, {"axial.trunk_lateral_lean": -10}),
    ({"abd_l": 60}, {"left.shoulder_abduction": 60, "right.shoulder_abduction": 0}),
    ({"abd_r": 45}, {"right.shoulder_abduction": 45}),
    ({"abd_r": -15}, {"right.shoulder_abduction": -15}),
    ({"hip_abd_l": 20, "hip_abd_r": 10}, {"left.hip_abduction": 20, "right.hip_abduction": 10}),
    ({"oblq": 8}, {"axial.pelvic_obliquity": 8}),
    ({"tilt_r": 20}, {"axial.neck_lateral_flexion": 20}),
    ({"turn_r": 30}, {"axial.neck_rotation": 30}),
    ({"turn_r": -20}, {"axial.neck_rotation": -20}),
])
def test_front_view(kw, expect):
    vals, meta = front_pose(**kw).compute("front")
    assert meta["view"] == "front"
    for k, want in expect.items():
        assert k in vals, f"{k} missing"
        assert vals[k] == pytest.approx(want, abs=1.5), f"{k}: {vals[k]} != {want}"


def test_front_shrug_raises_elevation():
    a, _ = front_pose().compute("front")
    b, _ = front_pose(shrug_l=10).compute("front")
    assert b["left.shoulder_elevation"] > a["left.shoulder_elevation"] + 5
    # The other side moves only a little (shoulder width changes slightly when one side rises).
    assert abs(b["right.shoulder_elevation"] - a["right.shoulder_elevation"]) < 3


def test_auto_view_detection():
    assert front_pose().compute("auto")[1]["view"] == "front"
    assert side_pose().compute("auto")[1]["view"] == "side"


def test_hidden_points_give_no_value():
    p = side_pose(hip=30)
    p.s[LAYOUTS["wholebody"]["l_knee"]] = 0.1  # model unsure about the knee
    p.s[LAYOUTS["wholebody"]["r_knee"]] = 0.1
    vals, _ = p.compute("side")
    assert "left.hip_flexion" not in vals and "left.knee_flexion" not in vals
    assert "left.shoulder_flexion" in vals


def test_rotations_never_reported_in_2d():
    vals, _ = side_pose(elbow=90).compute("side")
    for k in ("left.shoulder_rotation", "left.forearm_pronation", "left.hip_rotation", "left.ankle_inversion"):
        assert k not in vals


def test_view_selector_does_not_flip_on_single_frames():
    from live.angles2d import ViewSelector

    sel = ViewSelector("auto", hold=5)
    front, side = front_pose(), side_pose()
    fp = to_points(front.k, front.s, "wholebody", W, H)
    sp = to_points(side.k, side.s, "wholebody", W, H)
    seq = [fp] * 10 + [sp] * 3 + [fp] * 3  # 3 odd frames must not switch
    assert all(sel.update(p) == "front" for p in seq)
    assert [sel.update(sp) for _ in range(5)][-1] == "side"  # sustained change does switch


def test_view_from_hips_when_shoulders_cut_off():
    from live.angles2d import detect_view

    p = front_pose()
    for n in ("l_shoulder", "r_shoulder", "l_ear", "r_ear", "nose"):
        p.s[LAYOUTS["wholebody"][n]] = 0.0
    assert detect_view(to_points(p.k, p.s, "wholebody", W, H)) == "front"


def lying_pose(hip=0.0, knee=0.0, head_left=True):
    """Lying on the back, filmed from the side: trunk horizontal, nose pointing at the ceiling."""
    p = Pose()
    hip_pt = np.array([W / 2 + 100, H * 0.6])
    T = np.array([-1.0, 0.0]) if head_left else np.array([1.0, 0.0])  # hips -> shoulders
    sh = hip_pt + 220 * T
    for s, dy in (("l", -2.0), ("r", 2.0)):
        p.put(f"{s}_hip", hip_pt + [0, dy])
        p.put(f"{s}_shoulder", sh + [0, dy])
    up = np.array([0.0, -1.0])
    thigh = rot_toward(-T, up, hip) if hip else -T
    kn = hip_pt + 150 * thigh
    shank = rot_toward(thigh, -rot_toward(thigh, up if hip < 80 else T, 90), knee) if knee else thigh
    for s in ("l", "r"):
        p.put(f"{s}_knee", kn)
        p.put(f"{s}_ankle", kn + 140 * shank)
    ear = sh + 40 * T
    p.put("l_ear", ear)
    p.put("nose", ear + 25 * up + 10 * T)
    return p


@pytest.mark.parametrize("head_left", [True, False])
@pytest.mark.parametrize("hip,knee", [(0, 0), (45, 0), (70, 60), (90, 90)])
def test_lying_leg_raise(head_left, hip, knee):
    vals, meta = lying_pose(hip, knee, head_left).compute("side")
    assert meta.get("posture") == "lying"
    assert vals["left.hip_flexion"] == pytest.approx(hip, abs=1.5)
    assert vals["left.knee_flexion"] == pytest.approx(knee, abs=1.5)
