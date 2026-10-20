"""Unit tests: synthetic COCO poses -> fall features / score / monitor."""
import numpy as np
from fall_detector import fall_features, fall_score, FallMonitor


def pose(points):
    k = np.zeros((17, 3))
    for i, (x, y) in points.items():
        k[i] = (x, y, 0.95)
    return k


STANDING = pose({0: (320, 60), 5: (295, 120), 6: (345, 120),
                 11: (300, 260), 12: (340, 260), 13: (298, 345),
                 14: (342, 345), 15: (300, 430), 16: (340, 430)})
SITTING = pose({0: (320, 95), 5: (295, 150), 6: (345, 150),
                11: (300, 300), 12: (340, 300), 13: (400, 310),
                14: (405, 315), 15: (400, 430), 16: (405, 430)})
LYING = pose({0: (120, 420), 5: (180, 412), 6: (185, 445),
              11: (320, 420), 12: (325, 450), 13: (450, 425),
              14: (455, 448), 15: (570, 428), 16: (575, 450)})
BENDING = pose({0: (390, 300), 5: (370, 260), 6: (400, 265),
                11: (310, 250), 12: (350, 255), 13: (305, 340),
                14: (348, 340), 15: (302, 435), 16: (350, 435)})


def test_standing_is_safe():
    assert fall_score(fall_features(STANDING)) == 0


def test_sitting_is_safe():
    # sitting on a chair must NOT look like a fall
    assert fall_score(fall_features(SITTING)) <= 1


def test_lying_is_a_fall():
    f = fall_features(LYING)
    assert f["torso_angle"] > 55
    assert fall_score(f) >= 3


def test_bending_is_not_full_fall():
    assert fall_score(fall_features(BENDING)) <= 2


def test_monitor_triggers_and_recovers():
    m = FallMonitor(fps=30)
    fallen = fall_features(LYING)
    standing = fall_features(STANDING)
    events = []
    state = None
    for i in range(40):
        state, ev = m.update(fallen, i, 480)
        if ev:
            events.append(ev)
    assert state == "FALLEN"
    assert events == ["FALL"]
    for i in range(40, 140):
        state, _ = m.update(standing, i, 480)
    assert state == "NORMAL"


def test_monitor_ignores_brief_bend():
    m = FallMonitor(fps=30)
    bending = fall_features(BENDING)
    for i in range(10):  # 0.33 s bend - too brief to alarm
        state, ev = m.update(bending, i, 480)
        assert ev is None
