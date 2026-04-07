"""argparse + 진입점."""

import argparse
import yaml

from .runner import run_inference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/geometry_dms.yaml")
    ap.add_argument("--source", required=True)
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--start_sec", type=float, default=0.0, help="시작 시점 (초)")
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="Baseline: Kalman/Mahalanobis OOD 없이 raw 값으로 판별",
    )
    ap.add_argument(
        "--jitter_std",
        type=float,
        default=0.0,
        help="키포인트 가우시안 jitter σ (px). 0이면 비활성",
    )
    ap.add_argument("--inject_ood", action="store_true", help="OOD injection 활성화")
    ap.add_argument(
        "--ood_mode",
        choices=["cheek", "stretch"],
        default="cheek",
        help="cheek: 왼눈 6점 전체 ↓ / stretch: upper 2점 ↑ lower 2점 ↓",
    )
    ap.add_argument("--ood_start", type=float, default=10.0, help="OOD 주입 시작 시점 (초)")
    ap.add_argument("--ood_end", type=float, default=15.0, help="OOD 주입 종료 시점 (초)")
    ap.add_argument("--ood_shift", type=float, default=50.0, help="OOD 주입 시 shift 크기 (px)")
    ap.add_argument(
        "--no_skeleton",
        action="store_true",
        help="키포인트/스켈레톤 시각화 비활성화 (state만 표시)",
    )
    ap.add_argument(
        "--trace_csv",
        default=None,
        help="프레임별 timestamp_ms·state 등 로그 CSV (PERCLOS 등과 merge용)",
    )
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_inference(
        cfg,
        args.source,
        args.save_dir,
        args.max_frames,
        args.baseline,
        args.start_sec,
        args.jitter_std,
        args.inject_ood,
        args.ood_mode,
        args.ood_start,
        args.ood_end,
        args.ood_shift,
        args.no_skeleton,
        args.trace_csv,
    )
