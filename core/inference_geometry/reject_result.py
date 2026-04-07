"""YOLO 미검출 시 analyzer 대신 쓰는 결과."""

REJECT_NO_DETECTION = {
    "state": "Reject",
    "condition": "Reject",
    "explain": "Reject: no face detection (no YOLO box / keypoints).",
    "ear": 0,
    "yaw": 0,
    "pitch": 0,
    "r_sym": 1.0,
    "mahal_dist": float("inf"),
    "is_valid": False,
    "confidence": 0,
}
