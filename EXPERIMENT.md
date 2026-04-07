# exp3: 4D Feature + Condition 게이트 (Flag)

## 1. 목적

4D Feature Vector `[EAR, Yaw, Pitch, R_sym]` 위에 **Condition 게이트**를 둔다 — **Accept** 구간에서만 Attentive / Distracted / Drowsy를 확정하고, 분포 이탈은 **Flag**, 키포인트 불가는 **Reject**.

## 2. Feature Vector

```
x = [EAR, Yaw, Pitch, R_sym]   (4D)
```

- **15-keypoint 파이프라인**: $L\_eye\_center = mean(kpt\,0\text{–}5)$, $R\_eye\_center = mean(kpt\,6\text{–}11)$, $nose = kpt\,12$.
- **EAR**: Soukupova 6-point 공식 (눈당 6점, 좌우 평균).
- **R_sym**: 한쪽 눈 좌표가 이탈 → $R_{sym}$ 급변 → M-Dist 증가 → **Flag**.

## 3. 4D 통계량

ai-hubv2 학습 데이터 41,052 샘플에서 추출:

| 차원 | μ | σ | min | max |
|:----:|-----:|-----:|------:|------:|
| EAR | 0.2748 | 0.1498 | 0.0006 | 1.4504 |
| Yaw(°) | -2.6568 | 9.7535 | -85.36 | 107.72 |
| Pitch(°) | 16.5601 | 36.1766 | -81.03 | 87.32 |
| R_sym | 1.2402 | 0.5622 | 0.0902 | 38.12 |

### Threshold 설정

| 파라미터 | 값 | 설명 |
|:---------|:--:|:-----|
| ood_threshold | 6.0 | Flag 구간 상한 (칼만 predict-only 경계) |
| ood_soft_threshold | 4.0 | Accept vs Flag 경계 (15pt 4D 95th ≈ 3.11) |
| ood_flag_mode | soft_band | `soft_band` 또는 `single_tau` |

## 4. Condition · State (`trace.csv`)

`condition`과 `state`는 **서로 다른 열**이지만, 한 프레임에서 **아래 규칙으로만** 조합된다.

### `condition` (측정·신뢰 게이트, 프레임당 하나)

| condition | 의미 |
|-----------|------|
| **Calibrating** | 초기 `calibration_frames` 동안 기준 EAR만 모음. 행동 알람 없음. |
| **Reject** | 유효 특징 없음 (`feat is None`, 키포인트 conf 부족 등). **판별 불가**. |
| **Flag** | (Proposed만) 특징은 있는데 **M-Dist가 수용 경계 밖** → **확정 행동·알람 유보**. |
| **Accept** | M-Dist가 수용 구간 안 → **EAR/Yaw 타이머로 행동 분류 허용**. |

### `condition` → `state`

| condition | state (가능한 값) |
|-----------|-------------------|
| Calibrating | **Calibrating** 만 |
| Reject | **Reject** 만 |
| Flag | **Flag** 만 |
| **Accept** | **Attentive** 또는 **Distracted** 또는 **Drowsy** 중 **하나** |

### Baseline

M-Dist가 없어 **`condition == Flag`인 경우는 없다**.
`Calibrating` / `Reject` / `Accept`(이때만 state가 Attentive·Distracted·Drowsy)만 나온다.

## 5. Failure Case 결과

### 5.1 7-3 첫 2분 (1800f) · `runs/experiment_2min_clean_failure_7-3`

동일 15fps 클립, OOD는 10~15s. **Proposed = Full + soft_band + Flag.**

#### Eye-to-Cheek +50px

| | Baseline | Proposed |
|--|--:|--:|
| **Flag** | 0 | **76** |
| Calibrating | 30 | 30 |
| **condition Accept** | 1759 | 1683 |
| Reject | 11 | 11 |
| Attentive | 1378 | 1069 |
| Distracted | **61** | 0 |
| Drowsy | 320 | 614 |

#### Eye-Stretch ±150px

Cheek과 **동일한 OOD 길이(76f)** 라 행동 분포는 위와 동일.

| | Baseline | Proposed |
|--|--:|--:|
| **Flag** | 0 | **76** |
| Calibrating | 30 | 30 |
| **condition Accept** | 1759 | 1683 |
| Reject | 11 | 11 |
| Attentive | 1378 | 1069 |
| Distracted | **61** | 0 |
| Drowsy | 320 | 614 |

#### 픽셀 노이즈 σ=10 (OOD 없음)

| | Baseline | Proposed |
|--|--:|--:|
| **Flag** | 0 | **1** |
| Calibrating | 30 | 30 |
| **condition Accept** | 1759 | 1758 |
| Reject | 11 | 11 |
| Attentive | 1654 | 1199 |
| Drowsy | 105 | 559 |

**재현 시 참고**: `make_noisy_clip.py`는 매 실행마다 프레임별 난수 노이즈를 쓰므로, 재생성 시 수~수십 프레임 단위로 달라질 수 있다.

재현: `runs/experiment_2min_clean_failure_7-3/README.md`.

### 5.2 3-2 첫 2분 (Attentive 검증) · `runs/experiment_2min_3-2_attentive`

| | Baseline | Proposed |
|--|--:|--:|
| **Flag** | 0 | 0 |
| Calibrating | 30 | 30 |
| **condition Accept** | **1770** | **1770** |
| Reject | 0 | 0 |
| Attentive | **1770** | **1770** |
| Drowsy / Distracted | 0 | 0 |

재현: `runs/experiment_2min_3-2_attentive/README.md`.

### 5.3 3-1 첫 2분 · `runs/experiment_2min_3-1_failure`

§5.1과 같은 프로토콜(cheek / stretch / 픽셀 가우시안)을 **3-1** 첫 2분에 적용. **3-1은 30fps → 3600프레임**.

- **cheek**: Baseline Distracted **121** / Proposed Flag **151**, Drowsy 둘 다 0.
- **stretch**: Baseline **Distracted 0**. Proposed Flag **151**.
- **σ=10·σ=20**: Baseline·Proposed 모두 **Calibrating 30 + Attentive 3570**.

재현: `runs/experiment_2min_3-1_failure/README.md`.

---

## 6. 파일 구조

```
exp3/
├── EXPERIMENT.md
├── README.md
├── train_pose.py
├── configs/
│   ├── geometry_dms.yaml               # 기본 설정 (soft_band, grace 0.5s)
│   ├── geometry_dms_ear_grace_off.yaml  # grace=0 비교용
│   ├── geometry_dms_kalman_mid.yaml     # Kalman Q/R 중간 튜닝
│   └── geometry_dms_kalman_fast.yaml    # Kalman Q/R 공격 튜닝
├── core/
│   ├── dms_geometry.py                  # Condition + Flag + Grace
│   ├── inference_geometry/              # 영상 추론 패키지
│   ├── compute_geometry_stats.py
│   └── geometry_stats_4d.npz
├── scripts/
│   ├── convert_aihubv2_to_yolo_pose.py
│   ├── make_noisy_clip.py
│   └── check_distribution.py
├── utils/
│   └── face_pose15_viz.py
└── runs/
    ├── experiment_2min_clean_failure_7-3/
    ├── experiment_2min_3-1_failure/
    └── experiment_2min_3-2_attentive/
```

---

## 7. 재현

### 7.0 실행 환경

- **Python**: OpenCV (`cv2`), Ultralytics, PyYAML, PyTorch(+CUDA 권장).
- **예시**: `conda` 환경에서
  `python -m core.inference_geometry --config configs/geometry_dms.yaml --source <video.mp4>`
- **모듈 진입**: `python -m core.inference_geometry`는 `core/inference_geometry/__main__.py`를 통해 CLI 실행.

### 7.1 노이즈·입력 경로

- **픽셀 가우시안**: `scripts/make_noisy_clip.py` → MP4 후 `python -m core.inference_geometry --source ...`.
- **키포인트 jitter**: `--jitter_std`.

### 7.2 2분 failure + Attentive

- **7-3 (cheek / stretch / σ=10)**: `runs/experiment_2min_clean_failure_7-3/README.md` 의 bash 블록.
- **3-1 (cheek / stretch / σ=10·20, 30fps)**: `runs/experiment_2min_3-1_failure/README.md` 의 bash 블록.
- **3-2 Attentive**: `runs/experiment_2min_3-2_attentive/README.md` 의 bash 블록.

### 7.3 OOD inject 예시

```bash
cd exp3
python -m core.inference_geometry --config configs/geometry_dms.yaml \
    --source <clean_clip.mp4> --save_dir runs/.../stretch150_proposed \
    --inject_ood --ood_mode stretch --ood_start 10 --ood_end 15 --ood_shift 150
```

---

## 8. 보조 실험: Kalman 튜닝 · 런타임 Grace · PERCLOS 정합

**공통 조건**: `7-3.mp4` **처음 1800프레임**(15fps), `--no_skeleton`, `configs/geometry_dms.yaml` 계열.
**PERCLOS 정합**: 은표준 `media-pipe/7-3/7-3_silver_1min_ear_perclos.csv`, GT = `perclos_pct > 15%`, 스크립트 `media-pipe/grace_period_recall.py`. **G(초)**는 창 끝 시각 이후 허용 지연(오프라인 평가).

### 8.1 Baseline vs Full

| 모드 | Calibrating | Attentive | Drowsy | Reject |
|:--|--:|--:|--:|--:|
| **Baseline** (`--baseline`) | 30 | 1344 | 415 | 11 |
| **Full** (Proposed) | 30 | 1128 | 631 | 11 |

### 8.2 Kalman Q/R 튜닝

| 설정 파일 | Q | R | G=0s | G=3s | G=10s |
|:--|--:|--:|:--:|:--:|:--:|
| `geometry_dms.yaml` (기본) | 1e-3 | 0.08 | **0.523** | 0.636 | 0.932 |
| `geometry_dms_kalman_mid.yaml` | 2e-3 | 0.06 | 0.477 | 0.636 | 0.932 |
| `geometry_dms_kalman_fast.yaml` | 5e-3 | 0.04 | 0.409 | 0.614 | 0.932 |

**요약**: Q↑·R↓ 공격 튜닝이 G=0 근처 recall을 악화. 필터된 EAR 요동 → 연속 시간 조건의 타이머가 더 자주 끊김. 기본 Kalman 유지 권장.

### 8.3 런타임 Grace (`drowsy_ear_open_grace_sec`)

`EAR >= calib*ratio`일 때 즉시 리셋하지 않고, 연속으로 뜬 상태가 `grace_sec`를 넘길 때만 리셋.

| 설정 | grace | Attentive | Drowsy | Reject | Cal |
|:--|:--:|--:|--:|--:|--:|
| `geometry_dms_ear_grace_off.yaml` | **0** | 1128 | 631 | 11 | 30 |
| `geometry_dms.yaml` | **0.5s** | 970 | **789** | 11 | 30 |

**PERCLOS>15% 정합**:

| 평가 G | grace **0** | grace **0.5s** |
|:--|:--:|:--:|
| 0s | 0.523 (23/44) | **0.636** (28/44) |
| 3s | 0.636 | **0.818** (36/44) |
| 10s | 0.932 | **0.978** (43/44) |

### 8.4 런타임 Grace — Attentive (3-2) 오탐 점검

`3-2.mp4` 처음 1800프레임, Full only.

| 설정 | Calibrating | Attentive | Drowsy |
|:--|--:|--:|--:|
| grace **0** | 30 | **1770** | **0** |
| grace **0.5s** | 30 | **1770** | **0** |

**결론**: Attentive 구간에서는 grace ON/OFF 모두 Drowsy 0. Grace가 Drowsy 프레임을 늘리는 현상은 실제 졸음 구간(7-3)에서만 두드러짐.
