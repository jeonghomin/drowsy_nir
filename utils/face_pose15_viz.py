"""
YOLO 15-keypoint 얼굴 pose 시각화 (눈당 6점 + 코 + 입꼬리).

kpt 순서 (convert_aihubv2_to_yolo_pose.py --kpt 15):
  0-5:  왼쪽 눈 — outer(36), upper_outer(37), upper_inner(38), inner(39), lower_inner(40), lower_outer(41)
  6-11: 오른쪽 눈 — inner(42), upper_inner(43), upper_outer(44), outer(45), lower_outer(46), lower_inner(47)
  12: nose, 13: L_mouth, 14: R_mouth
"""

from __future__ import annotations

import numpy as np
import cv2

L_EYE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
R_EYE_EDGES = [(6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 6)]
BRIDGE_EDGES = [(3, 12), (6, 12), (12, 13), (12, 14), (13, 14)]

LIMB_COLOR_EYE = (0, 255, 128)
LIMB_COLOR_FACE = (200, 200, 0)
KPT_COLORS = [
    (255, 120, 0), (255, 180, 0), (255, 200, 0), (255, 80, 0), (200, 140, 0), (200, 100, 0),
    (0, 200, 120), (0, 230, 150), (0, 255, 180), (0, 160, 80), (0, 190, 100), (0, 220, 140),
    (0, 0, 255), (0, 255, 255), (255, 0, 255),
]
NAMES = [
    "Lo", "Luo", "Lui", "Li", "Lli", "Llo",
    "Ri", "Rui", "Ruo", "Ro", "Rlo", "Rli",
    "nose", "Lm", "Rm",
]


def draw_face15_skeleton(
    im,
    kpts_xy: np.ndarray,
    conf: np.ndarray | None = None,
    conf_thres: float = 0.25,
    line_thickness: int = 1,
    radius: int = 1,
    draw_labels: bool = False,
):
    if kpts_xy is None or len(kpts_xy) < 15:
        return im
    kpts_xy = np.asarray(kpts_xy, dtype=np.float32)[:15]
    if conf is not None:
        conf = np.asarray(conf, dtype=np.float32)[:15]
    h, w = im.shape[:2]

    def ok(i):
        return conf is None or conf[i] >= conf_thres

    def pt(i):
        return int(kpts_xy[i, 0]), int(kpts_xy[i, 1])

    def draw_edge(a, b, col):
        if not ok(a) or not ok(b):
            return
        p1, p2 = pt(a), pt(b)
        if not (0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h):
            return
        cv2.line(im, p1, p2, col, line_thickness, cv2.LINE_AA)

    for a, b in L_EYE_EDGES + R_EYE_EDGES:
        draw_edge(a, b, LIMB_COLOR_EYE)
    for a, b in BRIDGE_EDGES:
        draw_edge(a, b, LIMB_COLOR_FACE)

    for i in range(15):
        if not ok(i):
            continue
        x, y = pt(i)
        if not (0 <= x < w and 0 <= y < h):
            continue
        cv2.circle(im, (x, y), radius, KPT_COLORS[i], -1, cv2.LINE_AA)
        if draw_labels:
            cv2.putText(
                im, NAMES[i], (x + 3, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.22, KPT_COLORS[i], 1, cv2.LINE_AA,
            )
    return im
