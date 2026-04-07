"""
ai-hubv2 GT 70-point landmark → [EAR, Yaw, Pitch, R_sym] 통계량 계산.

Usage:
    cd exp1_2 && python -m core.compute_geometry_stats \
        --label_dir /path/to/ai-hubv2/train_labels \
        --output core/geometry_stats.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from .dms_geometry import extract_features_from_gt
except ImportError:
    from dms_geometry import extract_features_from_gt


def parse_keypoints(label, min_pts=68):
    kp = label.get("ObjectInfo", {}).get("KeyPoints", {})
    pts, n = kp.get("Points", []), kp.get("Count", 0)
    if n < min_pts or len(pts) < n * 2:
        return None
    return np.array([float(v) for v in pts[:n * 2]]).reshape(n, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_dir", required=True)
    ap.add_argument("--output", default="geometry_stats.npz")
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    label_dir = Path(args.label_dir)
    folders = sorted(d for d in label_dir.iterdir() if d.is_dir())
    print(f"폴더 수: {len(folders)}")

    features, skipped = [], 0

    for folder in folders:
        for jf in sorted(folder.glob("*.json")):
            try:
                with open(jf, encoding="utf-8") as f:
                    label = json.load(f)
            except (json.JSONDecodeError, OSError):
                skipped += 1
                continue

            pts = parse_keypoints(label)
            if pts is None:
                skipped += 1
                continue

            w = int(label["FileInfo"]["Width"])
            h = int(label["FileInfo"]["Height"])
            feat = extract_features_from_gt(pts, w, h)

            if feat is None or np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                skipped += 1
                continue

            features.append(feat)
            if 0 < args.max_samples <= len(features):
                break
        if 0 < args.max_samples <= len(features):
            break

    feats = np.array(features)
    ndim = feats.shape[1]
    print(f"유효 샘플: {len(feats)}, 스킵: {skipped}, 차원: {ndim}")

    mu = feats.mean(axis=0)
    cov = np.cov(feats, rowvar=False)
    inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(ndim))

    names = ["EAR", "Yaw(°)", "Pitch(°)", "R_sym"][:ndim]
    print(f"\n=== 통계량 ===")
    for i, n in enumerate(names):
        print(f"  {n:>8}: mu={mu[i]:.4f}  std={np.sqrt(cov[i, i]):.4f}  "
              f"min={feats[:, i].min():.4f}  max={feats[:, i].max():.4f}")

    diffs = feats - mu
    dists = np.sqrt(np.sum(diffs @ inv_cov * diffs, axis=1))
    for p in [50, 90, 95, 99, 99.5]:
        print(f"  M-dist {p:5.1f}th: {np.percentile(dists, p):.4f}")

    print(f"\n추천 threshold:")
    print(f"  ood_threshold (99th):      {np.percentile(dists, 99):.2f}")
    print(f"  ood_soft_threshold (95th): {np.percentile(dists, 95):.2f}")

    np.savez(args.output, mu=mu, covariance=cov, inv_cov=inv_cov)
    print(f"\n저장: {args.output} (dim={ndim})")


if __name__ == "__main__":
    main()
