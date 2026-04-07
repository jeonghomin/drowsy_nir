"""비디오 캡처 + 포즈 + DMS 분석 루프."""

import os
import csv
import time

import cv2
from ultralytics import YOLO

try:
    from ..dms_geometry import DMSGeometryAnalyzer, DMSBaselineAnalyzer
except ImportError:
    from dms_geometry import DMSGeometryAnalyzer, DMSBaselineAnalyzer

from .overlay import draw_overlay
from .trace import TRACE_COLUMNS, trace_row
from .keypoints import best_keypoints_from_result, add_keypoint_jitter, apply_ood_injection
from .reject_result import REJECT_NO_DETECTION


def run_inference(
    cfg,
    source,
    save_dir=None,
    max_frames=0,
    baseline=False,
    start_sec=0.0,
    jitter_std=0.0,
    inject_ood=False,
    ood_mode="cheek",
    ood_start=10.0,
    ood_end=15.0,
    ood_shift=50.0,
    no_skeleton=False,
    trace_csv=None,
):
    pose = YOLO(cfg["pose_model"])
    analyzer = DMSBaselineAnalyzer(cfg) if baseline else DMSGeometryAnalyzer(cfg)
    mode_tag = "baseline" if baseline else "full"
    if jitter_std > 0:
        mode_tag += f"_jitter{jitter_std:.0f}"
    if inject_ood:
        mode_tag += f"_ood-{ood_mode}_{ood_start:.0f}-{ood_end:.0f}s_shift{ood_shift:.0f}"
    info = f"Mode: {mode_tag}"
    if jitter_std > 0:
        info += f"  (jitter σ={jitter_std:.1f}px)"
    if inject_ood:
        info += f"  (OOD [{ood_mode}] {ood_start:.0f}~{ood_end:.0f}s, ±{ood_shift:.0f}px)"
    print(info)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"열 수 없음: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if start_sec > 0:
        calib_n = cfg.get("geometry", {}).get("calibration_frames", 30)
        print(f"Calibration: 0s부터 {calib_n}프레임 읽기...")
        for _ in range(calib_n):
            ret, frame = cap.read()
            if not ret:
                break
            results = pose(frame, verbose=False)
            kps_c, confs_c = best_keypoints_from_result(results)
            if kps_c is not None:
                vid_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                analyzer.update(kps_c, confs_c, w, h, timestamp=vid_ts)
        print(f"Calibration 완료 → Seek: {start_sec:.1f}s")
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

    writer = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"geometry_{mode_tag}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"저장: {out_path}")

    print(f"Source: {source} | {w}x{h} @ {fps:.0f}fps | {total} frames")

    trace_f = trace_w = None
    if trace_csv:
        tf = open(trace_csv, "w", newline="", encoding="utf-8")
        trace_w = csv.writer(tf)
        trace_w.writerow(TRACE_COLUMNS)
        trace_f = tf
        print(f"Trace CSV: {trace_csv}")

    counts, idx, t0 = {}, 0, time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts_ms = int(round(cap.get(cv2.CAP_PROP_POS_MSEC)))
        frame_index = idx

        results = pose(frame, verbose=False)

        kps = confs_kp = None
        ood_active = False
        kps, confs_kp = best_keypoints_from_result(results)
        if kps is not None:
            kps = add_keypoint_jitter(kps, jitter_std)
            cur_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            ood_active = apply_ood_injection(
                kps,
                confs_kp,
                cur_sec,
                inject_ood,
                ood_mode,
                ood_start,
                ood_end,
                ood_shift,
            )
            vid_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            result = analyzer.update(kps, confs_kp, w, h, timestamp=vid_ts)
        else:
            result = dict(REJECT_NO_DETECTION)

        if trace_w is not None:
            trace_w.writerow(trace_row(frame_index, ts_ms, result))

        counts[result["state"]] = counts.get(result["state"], 0) + 1

        if writer:
            writer.write(
                draw_overlay(
                    frame.copy(),
                    result,
                    kps,
                    confs_kp,
                    baseline,
                    jitter_std,
                    inject_ood=inject_ood and ood_active,
                    draw_skeleton=not no_skeleton,
                )
            )

        idx += 1
        if 0 < max_frames <= idx:
            break
        if idx % 100 == 0:
            print(
                f"  [{idx}/{total}] {result['state']:>12}  "
                f"EAR={result['ear']:.3f} Yaw={result['yaw']:.1f} "
                f"Pitch={result['pitch']:.1f} "
                f"M={result['mahal_dist']:.2f}  "
                f"{idx / (time.time() - t0):.0f}fps"
            )

    cap.release()
    if writer:
        writer.release()
    if trace_f is not None:
        trace_f.close()

    elapsed = time.time() - t0
    print(f"\n완료: {idx} frames / {elapsed:.1f}s ({idx / elapsed:.0f}fps)")
    print(f"상태 분포: {counts}")
