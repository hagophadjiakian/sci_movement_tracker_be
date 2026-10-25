"""Real MediaPipe pose + hand landmarks from a video -> JSON (same data the browser engine sees).

Used for real-video regression tests: the JSON is replayed through the web app's own angle code
(sci_movement_tracker_fe: npm run replay) and then through services/movement_detector.py.

Needs the legacy MediaPipe package (models are bundled, no download):
    pip install mediapipe==0.10.21 opencv-python      # Python 3.9-3.12
Usage:
    python tools/extract_landmarks.py clip.mp4 clip.lm.json
Keep patient videos out of git.
"""
import json, sys
import cv2, mediapipe as mp

src, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
pose = mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
frames = []
i = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    h, w = f.shape[:2]
    if w > 1280:
        f = cv2.resize(f, (1280, int(h * 1280 / w)))
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    r = pose.process(rgb)
    hr = hands.process(rgb)
    fr = {"t": round(i / fps, 4)}
    if r.pose_landmarks and r.pose_world_landmarks:
        fr["image"] = [[p.x, p.y, p.z, p.visibility] for p in r.pose_landmarks.landmark]
        fr["world"] = [[p.x, p.y, p.z, p.visibility] for p in r.pose_world_landmarks.landmark]
    if hr.multi_hand_landmarks and hr.multi_hand_world_landmarks:
        fr["hands"] = [{"image": [[p.x, p.y, p.z] for p in a.landmark], "world": [[p.x, p.y, p.z] for p in b.landmark]}
                       for a, b in zip(hr.multi_hand_landmarks, hr.multi_hand_world_landmarks)]
    frames.append(fr)
    i += 1
json.dump({"fps": fps, "frames": frames}, open(out, "w"))
print(f"{src.split('/')[-1]}: {i} frames, pose in {sum('image' in f for f in frames)}, hands in {sum('hands' in f for f in frames)}")
