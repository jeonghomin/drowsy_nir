# EfficientNet-Free DMS Pipeline — exp3

YOLOv8-Pose 15-keypoint의 **기하학적 분석만으로** 운전자 상태를 판별하는 경량 파이프라인.
EfficientNet 분류 모델을 완전히 제거하고, 좌표의 상관관계만 사용하여 이미지 도메인 변화에 강건하게 설계됨.
**Condition 게이트 (Flag)**, **drowsy_sensitivity 프리셋**, **EAR open grace**, **PERCLOS 정합 검증** 포함.

## 데이터셋

### 학습 데이터: AI-Hub 졸음운전 예방을 위한 운전자 상태 정보 영상

- **출처**: [AI-Hub (데이터 번호 173)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=173)
- **사용 부분**: **준 통제 환경 데이터** (Keypoint) — 100명, 50,000장
   LOv8-Pose 15-keypoint 포맷 변환
- **다운로드**: AI-Hub 회원가입 후 데이터 신청 (내국인만 가능)

### 평가 데이터: DROZY (ULg Multimodality Drowsiness Database)

- **출처**: [ORBi — ULiège](https://orbi.uliege.be/handle/2268/191620)
- **논문**: Massoz et al., "The ULg Multimodality Drowsiness Database (called DROZY) and examples of use", IEEE WACV 2016

## Setup

### 요구사항

- Python 3.10+


### 설치

```bash
# 1. conda 가상환경 생성
conda create -n dms python=3.11 -y
conda activate dms

# 2. PyTorch (CUDA 버전에 맞게 선택)
#    https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 3. 프로젝트 의존성 설치
cd exp3
pip install -r requirements.txt
```

> `ultralytics` 설치 시 `opencv-python`, `numpy`, `tqdm`, `pyyaml` 등이 함께 설치됩니다.


### 빠른 시작

```bash
# 추론 (Full pipeline)
python -m core.inference_geometry \
    --config configs/geometry_dms.yaml \
    --source /path/to/video.mp4 \
    --save_dir output/

# 학습
python train_pose.py --config configs/train_pose.yaml
```

## 파이프라인 개요


## 졸음 예민도 프리셋 (`drowsy_sensitivity`)

| 레벨 | ear_drowsy_ratio | drowsy_duration_sec | 설명 |
|:----:|:----------------:|:-------------------:|:-----|
| 1 | 0.48 | 3.0s | 덜 예민 |
| 2 | 0.55 | 2.0s | 기본 |
| 3 | 0.70 | 1.0s | 더 예민 |

`geometry_dms.yaml`에 `drowsy_sensitivity: 3` 설정 시 수동 `ear_drowsy_ratio` / `drowsy_duration_sec` 무시.


## 프로젝트 구조

```
exp3/
├── README.md
├── EXPERIMENT.md              # 실험 설계 및 결과 상세
├── train_pose.py              # YOLOv8-Pose 학습 스크립트
│
├── configs/
│   ├── geometry_dms.yaml              # DMS 파이프라인 설정 (soft_band, grace 등)
│   ├── geometry_dms_ear_grace_off.yaml  # grace=0 비교용
│   ├── geometry_dms_kalman_mid.yaml   # Kalman Q/R 중간 튜닝
│   ├── geometry_dms_kalman_fast.yaml  # Kalman Q/R 공격 튜닝
│   └── train_pose.yaml               # Pose 모델 학습 설정
│
├── core/
│   ├── dms_geometry.py        # DMSGeometryAnalyzer (Condition + Flag + Grace)
│   │                          # DMSBaselineAnalyzer (raw, no Kalman/M-Dist)
│   ├── inference_geometry/    # 영상 추론 패키지
│   │   ├── __main__.py        # python -m core.inference_geometry
│   │   ├── runner.py          # 캡처·YOLO·분석 루프
│   │   ├── overlay.py         # condition·state·explain 오버레이
│   │   ├── trace.py           # trace.csv (condition, state, explain 등)
│   │   ├── keypoints.py       # jitter / OOD 주입
│   │   └── cli.py             # argparse 진입점
│   ├── compute_geometry_stats.py  # μ, Σ⁻¹ 통계량 생성
│   └── geometry_stats_4d.npz  # 4D 통계량 (41,052 samples)
│
├── scripts/
│   ├── convert_aihubv2_to_yolo_pose.py  # dlib 70pts → YOLO 15kpt 변환
│   ├── make_noisy_clip.py     # 가우시안 노이즈 영상 생성
│   └── check_distribution.py  # 데이터 분포 확인
│
├── utils/
│   ├── face_pose15_viz.py     # 15-keypoint 시각화
│   ├── config.py
│   └── seed.py
│
├── weights/
│   ├── yolov8n-pose.pt        # 사전학습 가중치
│   └── face_landmarker_v2_with_blendshapes.task  # MediaPipe (보조 실험용)
│
├── media-pipe/                # MediaPipe 은표준 PERCLOS 실험 (§8)
│
└── runs/                      # 학습/추론/failure case 결과
```

## 설정 파일 (`configs/geometry_dms.yaml`)

```yaml
pose_model: .../face15kp_v1/weights/best.pt

geometry:
  stats: core/geometry_stats_4d.npz     # 4D: [EAR, Yaw, Pitch, R_sym]

  ood_flag_mode: soft_band              # soft_band | single_tau
  ood_threshold: 6.0                    # Flag 구간 상한 (predict-only)
  ood_soft_threshold: 4.0              # Accept 구간 상한 (R×2 step)
  min_keypoint_conf: 0.5               # conf < 0.5 → Reject

  drowsy_sensitivity: 3                # 1=덜예민, 2=기본, 3=더예민
  yaw_distracted_deg: 30.0
  distracted_duration_sec: 1
  drowsy_ear_open_grace_sec: 0.5       # EAR open grace

  kalman:
    process_noise: 1.0e-3
    measurement_noise: 0.08

  calibration_frames: 30

device: cuda:0
```

## 실행 방법

### 1. 데이터 변환 (dlib 70pts → YOLO 15kpt)

```bash
cd exp3
python scripts/convert_aihubv2_to_yolo_pose.py \
    --src_dir /path/to/ai-hubv2/annotations \
    --dst_dir /path/to/ai-hubv2/yolo_pose/labels \
    --kpt 15
```

### 2. 모델 학습

```bash
python train_pose.py --config configs/train_pose.yaml
```

### 3. 통계량 생성

```bash
python -m core.compute_geometry_stats \
    --label_dir /path/to/ai-hubv2/train_labels \
    --output core/geometry_stats_4d.npz
```

### 4. 추론

```bash
# Full pipeline (Kalman + Mahalanobis + Condition Flag + Grace)
python -m core.inference_geometry \
    --config configs/geometry_dms.yaml \
    --source /path/to/video.mp4 \
    --save_dir output/

# Baseline (raw 값, Kalman/M-Dist/Flag 없음)
python -m core.inference_geometry \
    --config configs/geometry_dms.yaml \
    --source /path/to/video.mp4 \
    --save_dir output/ --baseline

# 특정 시점부터 (0초에서 calibration 후 seek)
python -m core.inference_geometry \
    --config configs/geometry_dms.yaml \
    --source /path/to/video.mp4 \
    --save_dir output/ --start_sec 230

# 스켈레톤 없이 state만 표시
python -m core.inference_geometry \
    --config configs/geometry_dms.yaml \
    --source /path/to/video.mp4 \
    --save_dir output/ --no_skeleton

# trace CSV 저장 (PERCLOS 등 후처리용)
python -m core.inference_geometry \
    --config configs/geometry_dms.yaml \
    --source /path/to/video.mp4 \
    --save_dir output/ --trace_csv output/trace.csv
```

### 5. OOD Injection 테스트

```bash
# Eye-to-Cheek: 왼눈 6점 전체 +50px ↓
python -m core.inference_geometry \
    --config configs/geometry_dms.yaml \
    --source /path/to/video.mp4 --save_dir output/ \
    --inject_ood --ood_mode cheek --ood_start 10 --ood_end 15 --ood_shift 50

# Eye-Stretch: 왼눈 upper 2점 ↑, lower 2점 ↓
python -m core.inference_geometry \
    --config configs/geometry_dms.yaml \
    --source /path/to/video.mp4 --save_dir output/ \
    --inject_ood --ood_mode stretch --ood_start 10 --ood_end 15 --ood_shift 150
```

### 6. 노이즈 영상 생성

```bash
python scripts/make_noisy_clip.py \
    --source /path/to/video.mp4 \
    --output noisy_sigma10.mp4 \
    --sigma 10 --start_sec 0 --duration 120
```

## 방어 레이어 요약

```
Layer 1: Keypoint Confidence Filter    → 모델 불확실성 차단 (→ Reject)
Layer 2: Mahalanobis Distance (4D)     → 구조적 이상 탐지 (→ Flag)
         soft_band: Accept / Flag(R×2) / Flag(predict-only)
Layer 3: 1D Kalman Filter (×4)         → 시계열 노이즈 제거, OOD 중 값 보존
Layer 4: EAR Open Grace Period         → 칼만 스파이크 완화, 타이머 유지
Layer 5: Time-based Logic              → 순간 떨림 무시, 지속적 상태만 판정
```
