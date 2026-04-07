"""
DMSGeometryAnalyzer — 15-keypoint 기하학 기반 운전자 상태 판별.

15 Keypoints (convert --kpt 15):
    0–5: 왼눈 (outer, upper_outer, upper_inner, inner, lower_inner, lower_outer)
    6–11: 오른눈 (inner, upper_inner, upper_outer, outer, lower_outer, lower_inner)
    12: nose, 13–14: 입꼬리

Feature Vector: x = [EAR, Yaw(deg), Pitch(deg), R_sym]
EAR: Soukupova 6-point 공식 (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||) 좌우 평균.
R_sym: dist(L_eye_center, nose) / dist(R_eye_center, nose). 정면일 때 ≈ 1.0.
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np

NUM_FACE_KEYPOINTS = 15

# 표준 3D 얼굴 모델 (15-point, mm, nose 원점 근처)
# 0-5: L eye (outer, upper_outer, upper_inner, inner, lower_inner, lower_outer)
# 6-11: R eye (inner, upper_inner, upper_outer, outer, lower_outer, lower_inner)
# 12: nose, 13-14: mouth corners
MODEL_3D = np.array(
    [
        [-45.0, -32.0, -28.0],   # 0 L_outer
        [-40.0, -42.0, -26.0],   # 1 L_upper_outer
        [-28.0, -42.0, -26.0],   # 2 L_upper_inner
        [-20.0, -32.0, -28.0],   # 3 L_inner
        [-28.0, -22.0, -26.0],   # 4 L_lower_inner
        [-40.0, -22.0, -26.0],   # 5 L_lower_outer
        [20.0, -32.0, -28.0],    # 6 R_inner
        [28.0, -42.0, -26.0],    # 7 R_upper_inner
        [40.0, -42.0, -26.0],    # 8 R_upper_outer
        [45.0, -32.0, -28.0],    # 9 R_outer
        [40.0, -22.0, -26.0],    # 10 R_lower_outer
        [28.0, -22.0, -26.0],    # 11 R_lower_inner
        [0.0, 0.0, 0.0],         # 12 nose
        [-25.0, 30.0, -20.0],    # 13 L_mouth
        [25.0, 30.0, -20.0],     # 14 R_mouth
    ],
    dtype=np.float64,
)

DLIB_NOSE = 30
DLIB_L_MOUTH = 48
DLIB_R_MOUTH = 54


def compute_ear(kps):
    """
    Soukupova 6-point EAR (좌우 평균).
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    L eye: p1=0(outer), p2=1(upper_outer), p3=2(upper_inner),
           p4=3(inner), p5=4(lower_inner), p6=5(lower_outer)
    R eye: p1=9(outer), p2=8(upper_outer), p3=7(upper_inner),
           p4=6(inner), p5=11(lower_inner), p6=10(lower_outer)
    """
    kps = np.asarray(kps, dtype=np.float64)
    if kps.shape[0] < NUM_FACE_KEYPOINTS:
        return None
    eps = 1e-6
    # Left eye
    v1_l = np.linalg.norm(kps[1] - kps[5])   # upper_outer - lower_outer
    v2_l = np.linalg.norm(kps[2] - kps[4])   # upper_inner - lower_inner
    h_l = np.linalg.norm(kps[0] - kps[3])    # outer - inner
    # Right eye
    v1_r = np.linalg.norm(kps[8] - kps[10])  # upper_outer - lower_outer
    v2_r = np.linalg.norm(kps[7] - kps[11])  # upper_inner - lower_inner
    h_r = np.linalg.norm(kps[9] - kps[6])    # outer - inner
    if h_l < eps or h_r < eps:
        return None
    ear_l = (v1_l + v2_l) / (2.0 * h_l + eps)
    ear_r = (v1_r + v2_r) / (2.0 * h_r + eps)
    return 0.5 * (ear_l + ear_r)


def compute_symmetry_ratio(kps):
    """R_sym = dist(L_eye_center, nose) / dist(R_eye_center, nose).
    L_eye_center = mean(kps[0:6]), R_eye_center = mean(kps[6:12]), nose = kps[12].
    """
    kps = np.asarray(kps, dtype=np.float64)
    if kps.shape[0] < NUM_FACE_KEYPOINTS:
        return None
    l_center = kps[0:6].mean(axis=0)
    r_center = kps[6:12].mean(axis=0)
    nose = kps[12]
    dl = np.linalg.norm(l_center - nose)
    dr = np.linalg.norm(r_center - nose)
    if dr < 1e-6:
        return None
    return dl / dr


def estimate_head_pose(kps, img_w, img_h):
    """solvePnP → (yaw_deg, pitch_deg)."""
    kps = np.asarray(kps, dtype=np.float64)
    if kps.shape[0] < NUM_FACE_KEYPOINTS:
        return None, None

    cam = np.array(
        [
            [float(img_w), 0, img_w / 2.0],
            [0, float(img_w), img_h / 2.0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    ok, rvec, _ = cv2.solvePnP(
        MODEL_3D,
        kps[:NUM_FACE_KEYPOINTS].astype(np.float64),
        cam,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        return None, None

    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(-R[2, 0], sy))
    yaw = np.degrees(
        np.arctan2(R[1, 0], R[0, 0]) if sy > 1e-6 else np.arctan2(-R[1, 2], R[1, 1])
    )
    return yaw, pitch


def dlib70_to_15pts(pts):
    """dlib 70-point → 15-point (픽셀)."""
    return np.array(
        [
            pts[36],  # L_outer
            pts[37],  # L_upper_outer
            pts[38],  # L_upper_inner
            pts[39],  # L_inner
            pts[40],  # L_lower_inner
            pts[41],  # L_lower_outer
            pts[42],  # R_inner
            pts[43],  # R_upper_inner
            pts[44],  # R_upper_outer
            pts[45],  # R_outer
            pts[46],  # R_lower_outer
            pts[47],  # R_lower_inner
            pts[DLIB_NOSE],
            pts[DLIB_L_MOUTH],
            pts[DLIB_R_MOUTH],
        ],
        dtype=np.float64,
    )


def extract_features(kps, confs, img_w, img_h, min_conf=0.5):
    """15-keypoint → [EAR, Yaw, Pitch, R_sym] or None."""
    kps = np.asarray(kps, dtype=np.float64)
    confs = np.asarray(confs, dtype=np.float64)
    if kps.shape[0] < NUM_FACE_KEYPOINTS or confs.shape[0] < NUM_FACE_KEYPOINTS:
        return None
    if np.any(confs[:NUM_FACE_KEYPOINTS] < min_conf):
        return None
    ear = compute_ear(kps[:NUM_FACE_KEYPOINTS])
    if ear is None:
        return None
    yaw, pitch = estimate_head_pose(kps[:NUM_FACE_KEYPOINTS], img_w, img_h)
    if yaw is None:
        return None
    r_sym = compute_symmetry_ratio(kps[:NUM_FACE_KEYPOINTS])
    if r_sym is None:
        return None
    return np.array([ear, yaw, pitch, r_sym], dtype=np.float64)


def extract_features_from_gt(pts_70, img_w, img_h):
    """GT 70-point landmark → [EAR, Yaw, Pitch, R_sym] or None."""
    kps = dlib70_to_15pts(pts_70)
    ear = compute_ear(kps)
    if ear is None:
        return None
    yaw, pitch = estimate_head_pose(kps, img_w, img_h)
    if yaw is None:
        return None
    r_sym = compute_symmetry_ratio(kps)
    if r_sym is None:
        return None
    return np.array([ear, yaw, pitch, r_sym], dtype=np.float64)


def resolve_drowsy_params(g: dict) -> Tuple[float, float]:
    """
    졸음(EAR) 임계·지속시간.
    geometry.drowsy_sensitivity 가 1~3 이면 프리셋이 ear_drowsy_ratio / drowsy_duration_sec 보다 우선.
    1=덜 예민, 2=기본, 3=더 예민 (ratio↑ → 조금만 감아도 신호, duration↓ → 빨리 Drowsy).
    """
    sens = g.get("drowsy_sensitivity")
    if sens is not None:
        try:
            level = int(sens)
        except (TypeError, ValueError):
            level = 2
        level = max(1, min(3, level))
        presets = {
            1: (0.48, 3.0),
            2: (0.55, 2.0),
            3: (0.70, 1.0),
        }
        return presets[level]
    return (
        float(g.get("ear_drowsy_ratio", 0.7)),
        float(g.get("drowsy_duration_sec", 3)),
    )


def step_drowsy_low_ear_accumulator(
    now: float,
    ear_below_thresh: bool,
    grace_sec: float,
    t_drowsy_start: Optional[float],
    t_grace_start: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """
    EAR < 임계 누적 시작 시각 t_drowsy_start 유지. 임계 이상이면:
    - grace_sec <= 0 : 즉시 리셋 (기존 동작).
    - grace_sec > 0  : 연속으로 임계 이상인 구간이 grace_sec(초) 넘을 때만 리셋.
      짧은 스파이크(칼만 튐) 후 다시 감으면 t_grace_start 만 None 으로 초기화.
    """
    if grace_sec <= 0.0:
        if ear_below_thresh:
            return (t_drowsy_start or now), None
        return None, None

    if ear_below_thresh:
        return (t_drowsy_start or now), None

    if t_drowsy_start is None:
        return None, None

    if t_grace_start is None:
        return t_drowsy_start, now

    if (now - t_grace_start) >= grace_sec:
        return None, None

    return t_drowsy_start, t_grace_start


class KalmanFilter1D:
    def __init__(self, Q=1e-3, R=0.05, x0=0.0):
        self.Q, self.R_base, self.R = Q, R, R
        self.x_hat, self.P = x0, 1.0

    def predict(self):
        self.P += self.Q

    def step(self, z, r_scale=1.0):
        self.R = self.R_base * r_scale
        self.predict()
        K = self.P / (self.P + self.R)
        self.x_hat += K * (z - self.x_hat)
        self.P *= 1 - K
        return self.x_hat

    @property
    def state(self):
        return self.x_hat


class DMSGeometryAnalyzer:
    """키포인트 기하학 기반 DMS.

    Condition (신뢰 게이트):
      - Reject: 탐지 불가 (feat None).
      - Flag: 탐지는 되었으나 M-Dist가 신뢰 구간 밖 → Attentive/Distracted/Drowsy 미출력.
      - Accept: M-Dist가 수용 구간 내 → 행동 라벨 Attentive / Distracted / Drowsy 허용.

    ood_flag_mode:
      - soft_band: M <= ood_soft_threshold → Accept 구간. M > ood_soft → Flag;
        ood_soft < M <= ood_threshold 일 때 칼만은 R×2 업데이트, M > ood_threshold 는 predict-only.
      - single_tau: M <= ood_threshold → Accept, M > ood_threshold → Flag (칼만은 Accept에서만 step).
    """

    def __init__(self, cfg):
        g = cfg.get("geometry", {})

        self.ood_th = float(g.get("ood_threshold", 60.0))
        self.ood_soft = float(g.get("ood_soft_threshold", 40.0))
        self.min_conf = g.get("min_keypoint_conf", 0.5)

        mode = g.get("ood_flag_mode", "soft_band")
        self.flag_mode = mode if mode in ("soft_band", "single_tau") else "soft_band"

        self.ear_ratio, self.drowsy_dur = resolve_drowsy_params(g)
        self.yaw_th = g.get("yaw_distracted_deg", 30.0)
        self.drowsy_grace_sec = float(g.get("drowsy_ear_open_grace_sec", 0.0))

        self.distract_dur = g.get("distracted_duration_sec", 1)
        self.calib_n = g.get("calibration_frames", 30)

        k = g.get("kalman", {})
        pn, mn = k.get("process_noise", 1e-3), k.get("measurement_noise", 0.05)
        self._pn, self._mn = pn, mn
        self._kf_x0 = (0.28, 0.0, 0.0, 1.0)
        self.kf = [KalmanFilter1D(pn, mn, v) for v in self._kf_x0]

        self._load_stats(g)
        self._calib_buf = []
        self._calib_ear = None
        self._calibrated = False
        self._t_drowsy = self._t_distract = None
        self._t_drowsy_grace_start = None

    def _load_stats(self, g):
        p = g.get("stats")
        if p:
            d = np.load(p)
            self.mu, self.inv_cov = d["mu"], d["inv_cov"]
        else:
            ndim = 4
            self.mu, self.inv_cov = np.zeros(ndim), np.eye(ndim)

    def _mahal(self, x):
        d = x - self.mu
        return float(np.sqrt(max(0.0, d @ self.inv_cov @ d)))

    def _make_result(self, state, ear, yaw, pitch, mahal, valid, conf, r_sym=1.0, condition=None, explain=""):
        if condition is None:
            _cmap = {
                "Reject": "Reject",
                "Flag": "Flag",
                "Calibrating": "Calibrating",
                "Attentive": "Accept",
                "Drowsy": "Accept",
                "Distracted": "Accept",
            }
            condition = _cmap.get(state, "Accept")
        return {
            "state": state,
            "condition": condition,
            "explain": str(explain),
            "ear": float(ear),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "r_sym": float(r_sym),
            "mahal_dist": float(mahal),
            "is_valid": valid,
            "confidence": float(np.clip(conf, 0, 1)),
        }

    def _reject(self, ear=0, yaw=0, pitch=0, mahal=float("inf"), r_sym=1.0):
        ex = (
            f"Reject: invalid 15-pt features (any keypoint conf < {self.min_conf}, "
            "or EAR/head-pose failed)."
        )
        return self._make_result(
            "Reject", ear, yaw, pitch, mahal, False, 0.0, r_sym, condition="Reject", explain=ex
        )

    def _accept_bound_m(self):
        """M-Dist 이하일 때만 행동 분류(Attentive/Distracted/Drowsy) 허용."""
        if self.flag_mode == "single_tau":
            return self.ood_th
        return min(self.ood_soft, self.ood_th)

    def _in_accept_band(self, M):
        return M <= self._accept_bound_m()

    def _apply_kalman(self, feat, M):
        if self._in_accept_band(M):
            for kf, v in zip(self.kf, feat):
                kf.step(v, 1.0)
            return
        if self.flag_mode == "single_tau":
            for kf in self.kf:
                kf.predict()
            return
        if M <= self.ood_th:
            for kf, v in zip(self.kf, feat):
                kf.step(v, 2.0)
        else:
            for kf in self.kf:
                kf.predict()

    def _flag_confidence(self, M):
        return float(max(0.0, min(1.0, 1.0 - M / max(self.ood_th, 1e-6))))

    def update(self, kps, confs, img_w, img_h, timestamp=None):
        now = timestamp or time.monotonic()

        feat = extract_features(kps, confs, img_w, img_h, self.min_conf)
        if feat is None:
            self._t_drowsy = self._t_distract = None
            self._t_drowsy_grace_start = None
            return self._reject()

        M = self._mahal(feat)
        self._apply_kalman(feat, M)

        ear, yaw, pitch, r_sym = [kf.state for kf in self.kf]
        # callibration 시작
        if not self._calibrated:
            self._calib_buf.append(feat.copy())
            if len(self._calib_buf) >= self.calib_n:
                buf = np.array(self._calib_buf)
                self._calib_ear = buf[:, 0].mean()
                self._calibrated = True
            n = len(self._calib_buf)
            cal_ex = (
                f"Calibrating: collecting frames {n}/{self.calib_n} "
                "for EAR baseline calibration."
            )
            return self._make_result(
                "Calibrating", ear, yaw, pitch, M, True, 0.5, r_sym,
                condition="Calibrating", explain=cal_ex,
            )

        if not self._in_accept_band(M):
            self._t_drowsy = self._t_distract = None
            self._t_drowsy_grace_start = None
            fc = self._flag_confidence(M)
            bnd = self._accept_bound_m()
            flag_ex = (
                f"Flag: M-dist={M:.2f} > accept_bound={bnd:.2f} "
                f"({self.flag_mode}); defer behavior output."
            )
            return self._make_result(
                "Flag", ear, yaw, pitch, M, True, fc, r_sym, condition="Flag", explain=flag_ex
            )

        ear_low = ear < self._calib_ear * self.ear_ratio
        self._t_drowsy, self._t_drowsy_grace_start = step_drowsy_low_ear_accumulator(
            now,
            ear_low,
            self.drowsy_grace_sec,
            self._t_drowsy,
            self._t_drowsy_grace_start,
        )

        if abs(yaw) > self.yaw_th:
            self._t_distract = self._t_distract or now
        else:
            self._t_distract = None

        bnd = self._accept_bound_m()
        acc_prefix = f"Accept: M-dist={M:.2f} <= {bnd:.2f}; "

        if self._t_drowsy and (now - self._t_drowsy) >= self.drowsy_dur:
            d_ex = (
                f"{acc_prefix}EAR < calib*ratio sustained >= {self.drowsy_dur:.1f}s -> Drowsy."
            )
            return self._make_result(
                "Drowsy", ear, yaw, pitch, M, True,
                min(1, (now - self._t_drowsy) / self.drowsy_dur), r_sym,
                explain=d_ex,
            )
        if self._t_distract and (now - self._t_distract) >= self.distract_dur:
            di_ex = (
                f"{acc_prefix}|yaw| > {self.yaw_th:.0f} deg sustained >= {self.distract_dur:.1f}s "
                "-> Distracted."
            )
            return self._make_result(
                "Distracted", ear, yaw, pitch, M, True,
                min(1, (now - self._t_distract) / self.distract_dur), r_sym,
                explain=di_ex,
            )

        att_ex = f"{acc_prefix}EAR/yaw rules -> Attentive."
        return self._make_result(
            "Attentive",
            ear,
            yaw,
            pitch,
            M,
            True,
            max(0.0, 1.0 - M / max(self.ood_th, 1e-6)),
            r_sym,
            condition="Accept",
            explain=att_ex,
        )

    def reset(self):
        self.kf = [KalmanFilter1D(self._pn, self._mn, v) for v in self._kf_x0]
        self._calib_buf = []
        self._calib_ear = None
        self._calibrated = False
        self._t_drowsy = self._t_distract = None
        self._t_drowsy_grace_start = None


class DMSBaselineAnalyzer:
    """Baseline: raw EAR/Yaw/Pitch만 사용. Kalman·Mahalanobis OOD 없음."""

    def __init__(self, cfg):
        g = cfg.get("geometry", {})
        self.min_conf = g.get("min_keypoint_conf", 0.5)
        self.ear_ratio, self.drowsy_dur = resolve_drowsy_params(g)
        self.yaw_th = g.get("yaw_distracted_deg", 30.0)
        self.drowsy_grace_sec = float(g.get("drowsy_ear_open_grace_sec", 0.0))
        self.distract_dur = g.get("distracted_duration_sec", 1)
        self.calib_n = g.get("calibration_frames", 30)

        self._calib_buf = []
        self._calib_ear = None
        self._calibrated = False
        self._t_drowsy = self._t_distract = None
        self._t_drowsy_grace_start = None

    def _make_result(self, state, ear, yaw, pitch, r_sym=1.0, explain=""):
        _cmap = {
            "Reject": "Reject",
            "Calibrating": "Calibrating",
            "Attentive": "Accept",
            "Drowsy": "Accept",
            "Distracted": "Accept",
        }
        condition = _cmap.get(state, "Accept")
        return {
            "state": state,
            "condition": condition,
            "explain": str(explain),
            "ear": float(ear),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "r_sym": float(r_sym),
            "mahal_dist": 0.0,
            "is_valid": state != "Reject",
            "confidence": 1.0 if state != "Reject" else 0.0,
        }

    def update(self, kps, confs, img_w, img_h, timestamp=None):
        now = timestamp or time.monotonic()

        feat = extract_features(kps, confs, img_w, img_h, self.min_conf)
        if feat is None:
            self._t_drowsy = self._t_distract = None
            self._t_drowsy_grace_start = None
            rex = (
                f"Reject: invalid 15-pt features (any keypoint conf < {self.min_conf}, "
                "or EAR/head-pose failed)."
            )
            return self._make_result("Reject", 0, 0, 0, explain=rex)

        ear, yaw, pitch, r_sym = feat

        if not self._calibrated:
            self._calib_buf.append(feat.copy())
            if len(self._calib_buf) >= self.calib_n:
                buf = np.array(self._calib_buf)
                self._calib_ear = buf[:, 0].mean()
                self._calibrated = True
            n = len(self._calib_buf)
            cal_ex = (
                f"Calibrating: collecting frames {n}/{self.calib_n} "
                "for EAR baseline calibration."
            )
            return self._make_result("Calibrating", ear, yaw, pitch, r_sym, explain=cal_ex)

        ear_low = ear < self._calib_ear * self.ear_ratio
        self._t_drowsy, self._t_drowsy_grace_start = step_drowsy_low_ear_accumulator(
            now,
            ear_low,
            self.drowsy_grace_sec,
            self._t_drowsy,
            self._t_drowsy_grace_start,
        )

        if abs(yaw) > self.yaw_th:
            self._t_distract = self._t_distract or now
        else:
            self._t_distract = None

        acc_prefix = "Accept (baseline, no M-dist); "

        if self._t_drowsy and (now - self._t_drowsy) >= self.drowsy_dur:
            return self._make_result(
                "Drowsy", ear, yaw, pitch, r_sym,
                explain=f"{acc_prefix}low EAR sustained >= {self.drowsy_dur:.1f}s -> Drowsy.",
            )
        if self._t_distract and (now - self._t_distract) >= self.distract_dur:
            return self._make_result(
                "Distracted", ear, yaw, pitch, r_sym,
                explain=(
                    f"{acc_prefix}|yaw| > {self.yaw_th:.0f} deg sustained >= "
                    f"{self.distract_dur:.1f}s -> Distracted."
                ),
            )

        return self._make_result(
            "Attentive", ear, yaw, pitch, r_sym,
            explain=f"{acc_prefix}EAR/yaw rules -> Attentive.",
        )

    def reset(self):
        self._calib_buf = []
        self._calib_ear = None
        self._calibrated = False
        self._t_drowsy = self._t_distract = None
        self._t_drowsy_grace_start = None
