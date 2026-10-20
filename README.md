# Fall Detection System

Computer-vision system that detects when a person falls (elderly at home,
worker alone in a warehouse) so a caregiver/safety team can be alerted —
the same workflow as professional systems: **detect → snapshot → alert**.

## How it works
1. **Pose estimation** — YOLOv8-pose (ONNX, CPU) finds 17 body keypoints.
2. **Fall evidence score (0–4)** per frame:
   - torso angle vs vertical (> 55° = horizontal body)
   - body bounding-box aspect ratio (lying = wide, not tall)
   - head height dropped to hip level
   - hips down at floor (ankle) level
3. **Fast-drop check** — hips fell > 22% of frame height within 0.5 s
   (a real fall is fast; sitting down is slow).
4. **State machine** — NORMAL → SUSPECT → **FALLEN** (alarm + evidence
   snapshot + event log). Recovers automatically when the person stands.

## Run
```bash
python main.py --source video.mp4      # or --source 0 for webcam
python -m pytest test_fall.py         # 6 unit tests
python simulate_test.py               # scripted stand->fall->rise test
```

## Verified results
- Unit tests: 6/6 pass (standing, sitting, bending = safe; lying = fall).
- Real-person footage (walking, bending, sitting on sofa): **0 false alarms**
  — bending triggers SUSPECT (watching), never FALLEN.
- Simulated fall sequence: **FALL detected at 5.00 s**, snapshot saved,
  system recovered when the figure stood back up.
- Real fall footage (man falls backward onto a bed, staged clip):
  **FALL detected at 5.54 s** (frame 133), evidence snapshot saved.
  system recovered when the figure stood back up.

## Honest limits
- Rule-based geometry, not a trained fall classifier — unusual poses
  (lying on a sofa to nap) can resemble a fall; the SUSPECT stage and
  sustain timers exist to reduce false alarms.
- One person at a time (largest detection), fixed camera works best.

By **Usman Kalera** — Founder, Rakhzaar Tech · BS AI, Superior University Lahore

## Setup
```bash
pip install -r requirements.txt
# model file (not in repo - 6 MB ONNX):
# download yolov8n-pose.onnx from the Ultralytics YOLOv8 releases page
# https://github.com/ultralytics/assets/releases (yolov8n-pose.pt -> export ONNX)
# or export it yourself:  yolo export model=yolov8n-pose.pt format=onnx
```
Place `yolov8n-pose.onnx` next to `main.py` and run.
