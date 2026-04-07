"""프레임 오버레이: condition/state, explain, EAR·Mahal·스켈레톤."""

import cv2

try:
    from utils.face_pose15_viz import draw_face15_skeleton
    from utils.face_pose5_viz import draw_face5_skeleton
except ImportError:
    from face_pose15_viz import draw_face15_skeleton
    from face_pose5_viz import draw_face5_skeleton

STATE_COLORS = {
    "Attentive": (0, 200, 0),
    "Accept": (0, 200, 0),
    "Drowsy": (0, 0, 255),
    "Distracted": (0, 165, 255),
    "Flag": (0, 200, 255),
    "Reject": (128, 128, 128),
    "Calibrating": (255, 200, 0),
}


def overlay_color(result):
    """Reject > Flag > Calibrating > 행동 state."""
    st = result.get("state", "")
    cond = result.get("condition", "")
    if st == "Reject" or cond == "Reject":
        return STATE_COLORS["Reject"]
    if st == "Flag" or cond == "Flag":
        return STATE_COLORS["Flag"]
    if st == "Calibrating" or cond == "Calibrating":
        return STATE_COLORS["Calibrating"]
    return STATE_COLORS.get(st, (255, 255, 255))


def explain_lines(text, max_chars=88, max_lines=2):
    t = (text or "").strip() or "-"
    if len(t) <= max_chars:
        return [t]
    lines = [t[:max_chars]]
    rest = t[max_chars:].lstrip()
    if rest and max_lines > 1:
        lines.append((rest[:max_chars] + ("..." if len(rest) > max_chars else "")))
    return lines[:max_lines]


def draw_overlay(
    frame,
    result,
    kps=None,
    confs=None,
    baseline=False,
    jitter_std=0.0,
    inject_ood=False,
    draw_skeleton=True,
):
    h, w = frame.shape[:2]
    state = result.get("state", "")
    cond = result.get("condition", "")
    color = overlay_color(result)
    tag = "[BASELINE]" if baseline else "[FULL]"
    extra = ""
    if jitter_std > 0:
        extra += f"  jitter={jitter_std:.1f}px"
    if inject_ood:
        extra += "  [OOD-INJ]"

    explain = result.get("explain", "")
    ex_lines = explain_lines(explain)
    top_h = 158 if len(ex_lines) < 2 else 176
    cv2.rectangle(frame, (0, 0), (w, top_h), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"{tag}  condition={cond}  state={state}{extra}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
    )
    cv2.putText(
        frame,
        f"EAR:{result.get('ear', 0):.3f}  "
        f"Yaw:{result.get('yaw', 0):.1f}  Pitch:{result.get('pitch', 0):.1f}",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )
    if not baseline:
        cv2.putText(
            frame,
            f"Mahal:{result.get('mahal_dist', 0):.2f}  "
            f"Conf:{result.get('confidence', 0):.2f}",
            (10, 83),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        y_ex = 108
    else:
        y_ex = 88
    for i, ln in enumerate(ex_lines):
        cv2.putText(
            frame,
            ln,
            (10, y_ex + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (200, 220, 200),
            1,
            lineType=cv2.LINE_AA,
        )

    if kps is not None and draw_skeleton:
        c = confs if confs is not None else None
        nk = len(kps)
        if nk >= 15:
            draw_face15_skeleton(frame, kps[:15], c, conf_thres=0.3, draw_labels=True)
        else:
            draw_face5_skeleton(frame, kps[:5], c, conf_thres=0.3, draw_labels=True)

    cv2.rectangle(frame, (0, h - 8), (w, h), color, -1)
    return frame
