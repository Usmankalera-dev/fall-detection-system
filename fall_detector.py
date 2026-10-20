"""
Fall Detection System - core logic
----------------------------------
Pose estimation (YOLOv8-pose, ONNX) + rule-based fall classification
on COCO keypoints + a temporal state machine.

Fall evidence (per frame, from keypoints):
  1. Torso angle   : shoulder-mid -> hip-mid line vs vertical.
                     Standing ~0 deg, lying ~90 deg.
  2. Body aspect   : width/height of the keypoint bounding box.
                     Standing < 0.8, lying > 1.2.
  3. Head drop     : nose height close to hip height (both near floor).
  4. Hip drop speed: hips fell fast in the last ~0.5 s (a real fall is
                     a fast drop, sitting down is slow).

State machine: NORMAL -> SUSPECT -> FALLEN (alarm + snapshot)
               FALLEN -> NORMAL when the person stands back up.
"""
import math
import numpy as np

# COCO keypoint indices
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LSH, RSH = 5, 6
LEL, REL = 7, 8
LWR, RWR = 9, 10
LHIP, RHIP = 11, 12
LKNE, RKNE = 13, 14
LANK, RANK = 15, 16

KP_CONF = 0.35  # keypoint considered visible above this confidence


def _pt(kps, idx):
    """Return (x, y) of a keypoint or None if not visible."""
    x, y, c = kps[idx]
    if c < KP_CONF:
        return None
    return np.array([x, y], dtype=float)


def _mid(kps, a, b):
    pa, pb = _pt(kps, a), _pt(kps, b)
    if pa is None or pb is None:
        return None
    return (pa + pb) / 2.0


def fall_features(kps):
    """Compute fall evidence from one pose. kps: (17,3) x,y,conf array.
    Returns dict or None when the pose is too incomplete."""
    sh = _mid(kps, LSH, RSH)
    hip = _mid(kps, LHIP, RHIP)
    nose = _pt(kps, NOSE)
    ank = _mid(kps, LANK, RANK)
    if sh is None or hip is None:
        return None

    torso = hip - sh
    torso_len = float(np.linalg.norm(torso)) + 1e-6
    # angle of torso vs straight-down vertical (0 = upright)
    torso_angle = math.degrees(math.acos(min(1.0, abs(torso[1]) / torso_len)))
    if torso[1] < 0:  # hips above shoulders (upside down) - still horizontal-ish
        torso_angle = 180.0 - torso_angle

    vis = kps[kps[:, 2] >= KP_CONF]
    w = float(vis[:, 0].max() - vis[:, 0].min())
    h = float(vis[:, 1].max() - vis[:, 1].min()) + 1e-6
    aspect = w / h

    head_drop = None
    if nose is not None:
        # nose level relative to hips, in torso lengths.
        # upright: nose is ~1.4 torso-lengths ABOVE hips (negative dy up)
        head_drop = float((nose[1] - hip[1]) / torso_len)  # y grows downward

    hip_low = None
    if ank is not None:
        # hips close to ankle height => whole body near the floor
        hip_low = float(abs(hip[1] - ank[1]) / torso_len)

    return {
        "torso_angle": torso_angle,
        "aspect": aspect,
        "head_drop": head_drop,
        "hip_low": hip_low,
        "shoulder_mid": sh,
        "hip_mid": hip,
    }


def fall_score(feat):
    """0-4 evidence score for one frame."""
    if feat is None:
        return 0
    s = 0
    if feat["torso_angle"] > 55:
        s += 1
    if feat["aspect"] > 1.15:
        s += 1
    if feat["head_drop"] is not None and feat["head_drop"] > -0.45:
        s += 1  # head no longer well above hips
    if feat["hip_low"] is not None and feat["hip_low"] < 0.55:
        s += 1  # hips down at floor level
    return s


class FallMonitor:
    """Temporal state machine over per-frame fall scores."""

    NORMAL, SUSPECT, FALLEN = "NORMAL", "SUSPECT", "FALLEN"

    def __init__(self, fps=30, suspect_secs=0.7, recover_secs=2.5):
        self.fps = fps
        self.suspect_frames = max(1, int(suspect_secs * fps))
        self.recover_frames = max(1, int(recover_secs * fps))
        self.state = self.NORMAL
        self._hot = 0    # consecutive frames with score >= 2
        self._cold = 0   # consecutive frames with score < 2
        self._hip_hist = []  # (frame_idx, hip_y) for drop-speed check

    def update(self, feat, frame_idx, frame_h):
        """Returns (state, event) where event is 'FALL' on the frame
        the alarm triggers, else None."""
        score = fall_score(feat)
        event = None

        # fast hip-drop: hips fell > 22% of frame height within 0.5 s
        fast_drop = False
        if feat is not None:
            hip_y = float(feat["hip_mid"][1])
            self._hip_hist.append((frame_idx, hip_y))
            window = int(0.5 * self.fps)
            self._hip_hist = [(f, y) for f, y in self._hip_hist
                              if frame_idx - f <= window]
            if len(self._hip_hist) >= 3:
                drop = hip_y - self._hip_hist[0][1]
                if drop > 0.22 * frame_h and score >= 2:
                    fast_drop = True

        if score >= 2:
            self._hot += 1
            self._cold = 0
        else:
            self._cold += 1
            self._hot = 0

        if self.state == self.NORMAL:
            if self._hot >= 2 or fast_drop:
                self.state = self.SUSPECT
        elif self.state == self.SUSPECT:
            if fast_drop or self._hot >= self.suspect_frames:
                self.state = self.FALLEN
                event = "FALL"
            elif self._cold >= self.recover_frames:
                self.state = self.NORMAL
        elif self.state == self.FALLEN:
            if self._cold >= self.recover_frames:
                self.state = self.NORMAL
        return self.state, event


# ---------------- YOLOv8-pose ONNX wrapper ----------------

class PoseEstimator:
    def __init__(self, model_path, conf=0.40):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(model_path,
                                         providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.conf = conf

    def _letterbox(self, img, size=640):
        h, w = img.shape[:2]
        r = min(size / h, size / w)
        nh, nw = int(h * r), int(w * r)
        import cv2
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        top, left = (size - nh) // 2, (size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, left, top

    def detect(self, img):
        """Returns list of dicts: {box:(x1,y1,x2,y2), kps:(17,3)} in
        original image coordinates."""
        import cv2
        canvas, r, dx, dy = self._letterbox(img)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.in_name: blob})[0]  # (1,56,8400)
        pred = out[0].transpose(1, 0)  # (8400,56)
        boxes, scores, kps_all = [], [], []
        for row in pred:
            if row[4] < self.conf:
                continue
            cx, cy, bw, bh = row[:4]
            boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            scores.append(float(row[4]))
            kps_all.append(row[5:].reshape(17, 3))
        if not boxes:
            return []
        boxes = np.array(boxes, dtype=np.float32)
        keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores, self.conf, 0.45)
        results = []
        for i in np.array(keep).flatten():
            b = boxes[i]
            x1 = (b[0] - dx) / r; y1 = (b[1] - dy) / r
            x2 = (b[2] - dx) / r; y2 = (b[3] - dy) / r
            k = kps_all[i].copy()
            k[:, 0] = (k[:, 0] - dx) / r
            k[:, 1] = (k[:, 1] - dy) / r
            results.append({"box": (x1, y1, x2, y2), "kps": k})
        return results
