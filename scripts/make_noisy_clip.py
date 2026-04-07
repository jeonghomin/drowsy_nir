"""
영상 구간을 잘라서 가우시안 노이즈를 주입한 클립 생성.

Usage:
    python scripts/make_noisy_clip.py \
        --source /path/to/video.mp4 \
        --out_dir runs/failure_cases/01_gaussian_jitter \
        --start_sec 90 --duration 45 \
        --sigma 10 20 40
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_clip(source, out_path, start_sec, duration, sigma, fps, w, h):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    max_frames = int(duration * fps)
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if sigma > 0:
            noise = np.random.randn(*frame.shape).astype(np.float32) * sigma
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"  σ={sigma:>2d} → {out_path.name}  ({i+1} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--start_sec", type=float, default=90)
    ap.add_argument("--duration", type=float, default=45)
    ap.add_argument("--sigma", type=int, nargs="+", default=[0, 10, 20, 40])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    stem = Path(args.source).stem
    print(f"Source: {args.source} | {w}x{h} @ {fps:.0f}fps")
    print(f"Clip: {args.start_sec}s ~ {args.start_sec + args.duration}s")

    sec_tag = f"_{int(args.start_sec)}s" if args.start_sec > 0 else ""
    for s in args.sigma:
        noise_tag = "clean" if s == 0 else f"noise_s{s}"
        out_path = out_dir / f"{stem}{sec_tag}_{noise_tag}.mp4"
        make_clip(args.source, out_path, args.start_sec, args.duration, s, fps, w, h)

    print(f"\n완료: {out_dir}")


if __name__ == "__main__":
    main()
