"""trace CSV: condition + state."""

import math

TRACE_COLUMNS = [
    "frame_index",
    "timestamp_ms",
    "condition",
    "state",
    "ear",
    "yaw",
    "pitch",
    "mahal_dist",
    "confidence",
    "is_valid",
]


def csv_float(x):
    if x is None:
        return ""
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return ""
    return f"{x:.6f}"


def trace_row(frame_index, ts_ms, result):
    return [
        frame_index,
        ts_ms,
        result.get("condition", ""),
        result.get("state", ""),
        csv_float(result.get("ear")),
        csv_float(result.get("yaw")),
        csv_float(result.get("pitch")),
        csv_float(result.get("mahal_dist")),
        csv_float(result.get("confidence")),
        int(bool(result.get("is_valid", False))),
    ]
