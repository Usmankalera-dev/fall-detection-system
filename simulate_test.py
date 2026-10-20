"""
Simulated fall test sequence.
A scripted keypoint timeline (stand -> stumble -> fall -> lie -> rise)
is rendered as an animated figure and fed through the SAME
fall_features + FallMonitor used on real video. Every frame is
labeled SIMULATED so nobody mistakes it for real footage.
Output: output-sim/demo-fall-simulated.mp4 + fall snapshot + events.
"""
import math
import os

import cv2
import numpy as np

from fall_detector import fall_features, FallMonitor

W, H, FPS = 1280, 720, 24
OUT = "output-sim"
os.makedirs(OUT, exist_ok=True)

# keyframe poses: 17 COCO points (x, y)
def P(nose, lsh, rsh, lel, rel, lwr, rwr, lhip, rhip, lkne, rkne, lank, rank):
    k = np.zeros((17, 3))
    pts = {0: nose, 5: lsh, 6: rsh, 7: lel, 8: rel, 9: lwr, 10: rwr,
           11: lhip, 12: rhip, 13: lkne, 14: rkne, 15: lank, 16: rank,
           1: nose, 2: nose, 3: nose, 4: nose}
    for i, (x, y) in pts.items():
        k[i] = (x, y, 0.95)
    return k

STAND = P((640, 140), (612, 212), (668, 212), (600, 300), (680, 300),
          (596, 385), (684, 385), (620, 392), (660, 392), (617, 512),
          (663, 512), (614, 632), (666, 632))
STUMBLE = P((712, 205), (678, 268), (728, 252), (660, 350), (748, 330),
            (646, 420), (762, 400), (636, 408), (676, 402), (628, 520),
            (676, 516), (618, 634), (684, 630))
LYING = P((428, 606), (498, 596), (503, 628), (560, 610), (566, 640),
          (618, 616), (622, 646), (664, 604), (668, 634), (796, 608),
          (800, 636), (928, 610), (932, 638))
CROUCH = P((668, 330), (640, 392), (694, 388), (618, 460), (716, 452),
           (606, 528), (726, 512), (622, 470), (664, 468), (604, 540),
           (688, 536), (600, 632), (690, 630))

# (start_frame, from_pose, to_pose)
TIMELINE = [
    (0, STAND, STAND),        # 0-3 s standing
    (72, STAND, STUMBLE),     # 3-4 s stumble
    (96, STUMBLE, LYING),     # 4-5 s the fall (fast)
    (120, LYING, LYING),      # 5-10 s lying still
    (240, LYING, CROUCH),     # 10-11 s getting up
    (264, CROUCH, STAND),     # 11-12 s standing again
    (288, STAND, STAND),      # 12-13 s recovered
]
TOTAL = 312


def pose_at(f):
    for i in range(len(TIMELINE) - 1):
        f0, a, _ = TIMELINE[i]
        f1, _, b = TIMELINE[i + 1][0], None, TIMELINE[i + 1][1]
        if f0 <= f < f1:
            t = (f - f0) / max(1, f1 - f0)
            # gravity ease-in during the fall segment
            if i == 2:
                t = t * t
            else:
                t = t * t * (3 - 2 * t)
            return a * (1 - t) + b * t
    return TIMELINE[-1][1].copy()


SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
COLORS = {"NORMAL": (0, 200, 0), "SUSPECT": (0, 200, 255), "FALLEN": (0, 0, 255)}


def draw_room(img):
    img[:] = (196, 178, 158)                      # wall
    cv2.rectangle(img, (0, 560), (W, H), (120, 96, 74), -1)   # floor
    cv2.rectangle(img, (880, 400), (1230, 560), (90, 90, 140), -1)  # sofa
    cv2.rectangle(img, (60, 180), (240, 560), (105, 105, 105), -1)  # door
    cv2.putText(img, "SIMULATED POSE TEST - no real person",
                (14, H - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_person(img, kps):
    def pt(i):
        return tuple(kps[i][:2].astype(int))
    for a, b in SKELETON:
        cv2.line(img, pt(a), pt(b), (60, 60, 60), 16, cv2.LINE_AA)
    for a, b in [(5, 11), (6, 12), (11, 12), (5, 6)]:
        cv2.line(img, pt(a), pt(b), (160, 90, 40), 26, cv2.LINE_AA)  # torso/shirt
    cv2.circle(img, pt(0), 27, (190, 160, 130), -1, cv2.LINE_AA)     # head


def main():
    writer = cv2.VideoWriter(os.path.join(OUT, "demo-fall-simulated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    mon = FallMonitor(fps=FPS)
    events = []
    snap_done = False
    for f in range(TOTAL):
        kps = pose_at(f)
        kps = kps.copy()
        kps[:, 0] += 2.5 * math.sin(f / 9.0)  # gentle sway / breathing
        feat = fall_features(kps)
        state, ev = mon.update(feat, f, H)
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        draw_room(frame)
        draw_person(frame, kps)
        vis = kps[kps[:, 2] > 0.35]
        x1, y1 = int(vis[:, 0].min()) - 24, int(vis[:, 1].min()) - 24
        x2, y2 = int(vis[:, 0].max()) + 24, int(vis[:, 1].max()) + 24
        c = COLORS[state]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        cv2.putText(frame, f"{state}  torso {feat['torso_angle']:.0f} deg",
                    (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, c, 2)
        if state == "FALLEN":
            cv2.rectangle(frame, (0, 0), (W, 46), (0, 0, 200), -1)
            cv2.putText(frame, "!! FALL DETECTED - ALERT !!", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            if ev == "FALL" and not snap_done:
                cv2.imwrite(os.path.join(OUT, "fall_snapshot.jpg"), frame)
                snap_done = True
        if ev == "FALL":
            events.append((f, round(f / FPS, 2)))
            print(f"[frame {f}] FALL DETECTED at {f / FPS:.2f}s")
        cv2.putText(frame, f"t={f / FPS:.1f}s", (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        writer.write(frame)
    writer.release()
    print(f"Simulated run done: {len(events)} fall event(s) -> {events}")


if __name__ == "__main__":
    main()
