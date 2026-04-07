"""
YOLOv8-Pose Fine-tuning Script.

Usage:
    python train_pose.py --config configs/train_pose.yaml
    python train_pose.py --config configs/train_pose.yaml --epochs 100 --batch 32
"""

import argparse
import yaml
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_pose.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--fraction", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch is not None:
        cfg["batch"] = args.batch
    if args.device is not None:
        cfg["device"] = args.device
    if args.fraction is not None:
        cfg["fraction"] = args.fraction

    model_path = cfg.pop("model")
    model = YOLO(model_path)

    if args.resume:
        cfg["resume"] = True

    print(f"Model: {model_path}")
    print(f"Config: {args.config}")
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")
    print("-" * 60)

    model.train(**cfg)


if __name__ == "__main__":
    main()
