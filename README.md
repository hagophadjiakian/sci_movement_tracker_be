# sci_movement_tracker_be

FastAPI backend for the NeuroMotion SCI tracker:

- `POST /api/sessions`: stores a recording and analyses it in the background (several in parallel)
- `POST /api/analyze-movement`: AI estimate for one pose (Gemini, optional)
- **Live engine** (`/api/live/*`, `/live/video.mjpg`, `/ws/live`): runs your camera and the RTMPose
  model on this computer and streams joint angles to the web app

## Install

```bash
python -m venv venv
venv\Scripts\activate            # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt   # API only
pip install -r requirements-live.txt   # + live RTMPose engine (camera)
```

NVIDIA graphics card (much faster live tracking):

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

Copy `env.example` to `.env` and add `GEMINI_API_KEY` if you want the AI notes.

## Run

```bash
python main.py        # http://127.0.0.1:8000
```

Then start the web app (`sci_movement_tracker_fe`, `npm run dev`) and choose
**Engine → Python (RTMPose)**, then **Start camera**.

The first start downloads the pose models (100–300 MB) into `~/.cache/rtmlib`; later starts take seconds.
Only one program can use a webcam at a time: close Zoom/Teams/the browser engine first.

### Live engine options

| Option | Choices | Notes |
|---|---|---|
| Camera | `0`, `1`, ... or a video file path | A video file plays at real speed (loops); handy for re-analysing clips |
| Model | Body + hands + feet / Body + feet / Body only | Hands cost the most speed |
| Model size | Fast / Balanced / Most accurate | |
| Device | auto / NVIDIA GPU / CPU | auto uses the GPU when `onnxruntime-gpu` finds one |
| View | auto / side / front | 2D: side view measures flexion/extension, front view abduction and side lean |

The engine follows the biggest person and then stays with the same person, so a helper walking
through the picture does not take over the tracking.

## Tests

```bash
pip install pytest
python -m pytest -q
```

Tests use a stand-in pose estimator (`LIVE_ESTIMATOR=synthetic`), so they need no camera and no model download.
