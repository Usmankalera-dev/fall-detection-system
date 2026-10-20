"""
Run the Fall Detection System on a video file (or webcam: --source 0).

Outputs (in output/):
  - annotated video with skeleton + status banner
  - fall snapshots (evidence images, like a caregiver alert would send)
  - events.csv log (frame, time, event)
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np

from fall_detector import PoseEstimator, FallMonitor, fall_features

SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]

COLORS = {"NORMAL": (0, 200, 0), "SUSPECT": (0, 200, 255), "FALLEN": (0, 0, 255)}


def draw(frame, det, state, feat, frame_idx, fps):
    x1, y1, x2, y2 = [int(v) for v in det["box"]]
    color = COLORS[state]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    kps = det["kps"]
    for a, b in SKELETON:
        if kps[a][2] > 0.35 and kps[b][2] > 0.35:
            cv2.line(frame, tuple(kps[a][:2].astype(int)),
                     tuple(kps[b][:2].astype(int)), (255, 200, 0), 2)
    for k in kps:
        if k[2] > 0.35:
            cv2.circle(frame, tuple(k[:2].astype(int)), 3, (255, 255, 255), -1)
    label = state
    if feat is not None:
        label += f"  torso {feat['torso_angle']:.0f} deg"
    cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if state == "FALLEN":
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (0, 0, 200), -1)
        cv2.putText(frame, "!! FALL DETECTED - ALERT !!", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"t={frame_idx / fps:.1f}s", (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="video path or 0 for webcam")
    ap.add_argument("--model", default="yolov8n-pose.onnx")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(os.path.join(args.out, "annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    pose = PoseEstimator(args.model)
    monitor = FallMonitor(fps=fps)
    events = []
    idx = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = pose.detect(frame)
        state, feat = "NORMAL", None
        if dets:
            det = max(dets, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
            feat = fall_features(det["kps"])
            state, event = monitor.update(feat, idx, h)
            if event == "FALL":
                snap = os.path.join(args.out, f"fall_snapshot_{idx}.jpg")
                cv2.imwrite(snap, frame)
                events.append((idx, round(idx / fps, 2), "FALL"))
                print(f"[frame {idx}] FALL DETECTED -> snapshot saved")
            frame = draw(frame, det, state, feat, idx, fps)
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()
    with open(os.path.join(args.out, "events.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time_s", "event"])
        wr.writerows(events)
    print(f"Done: {idx} frames in {time.time() - t0:.1f}s, "
          f"{len(events)} fall event(s). Outputs in {args.out}/")


if __name__ == "__main__":
    main()
