"""YOLO 결과에서 키포인트 추출, jitter, OOD 주입."""

import numpy as np


def best_keypoints_from_result(results):
    if len(results[0].boxes) == 0 or results[0].keypoints is None:
        return None, None
    best = results[0].boxes.conf.argmax()
    kps = results[0].keypoints.xy[best].cpu().numpy()
    nk = int(results[0].keypoints.xy.shape[-2])
    confs = (
        results[0].keypoints.conf[best].cpu().numpy()
        if results[0].keypoints.conf is not None
        else np.ones(nk, dtype=np.float32)
    )
    return kps, confs


def add_keypoint_jitter(kps, jitter_std):
    if jitter_std <= 0:
        return kps
    return kps + np.random.randn(*kps.shape).astype(kps.dtype) * jitter_std


def apply_ood_injection(kps, confs_kp, cur_sec, enabled, ood_mode, ood_start, ood_end, ood_shift):
    if not enabled:
        return False
    if not (ood_start <= cur_sec <= ood_end):
        return False
    if ood_mode == "cheek":
        kps[0:6, 1] += ood_shift
        confs_kp[0:6] = 0.95
    elif ood_mode == "stretch":
        kps[1, 1] -= ood_shift
        kps[2, 1] -= ood_shift
        kps[4, 1] += ood_shift
        kps[5, 1] += ood_shift
        confs_kp[1] = 0.95
        confs_kp[2] = 0.95
        confs_kp[4] = 0.95
        confs_kp[5] = 0.95
    return True
