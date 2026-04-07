"""
ai-hubv2 (dlib 70-point) → YOLOv8-Pose 변환 스크립트.

기본(--kpt 15): 각 눈당 6점(dlib 원본) + 코 + 입꼬리 (Soukupova EAR 공식용).
옵션(--kpt 11): 각 눈당 4점 (mean 처리, 레거시).
옵션(--kpt 5): 눈 중심 2점 + 코 + 입 (레거시).

15-keypoint (dlib 인덱스, 0-based):
    L_eye: 36(외), 37(상외), 38(상내), 39(내), 40(하내), 41(하외)
    R_eye: 42(내), 43(상내), 44(상외), 45(외), 46(하외), 47(하내)
    nose 30, L_mouth 48, R_mouth 54

출력 포맷 (YOLO Pose):
    <class> <cx> <cy> <w> <h> <px1> <py1> <v1> ... 
    좌표 0~1 정규화, visibility 2 = visible

Usage:
    python scripts/convert_aihubv2_to_yolo_pose.py \\
        --img_dir  .../ai-hubv2/train_images \\
        --lbl_dir  .../ai-hubv2/train_labels \\
        --out_dir  .../ai-hubv2/yolo_pose/train \\
        [--kpt 15] [--pad_ratio 0.2]
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

DLIB_LEFT_EYE_CENTER = [37, 38, 40, 41]
DLIB_RIGHT_EYE_CENTER = [43, 44, 46, 47]
DLIB_NOSE_TIP = 30
DLIB_LEFT_MOUTH = 48
DLIB_RIGHT_MOUTH = 54


def parse_keypoints(label: dict, n_expected: int = 70):
    """JSON에서 (N, 2) keypoint 배열 파싱."""
    kp = label.get("ObjectInfo", {}).get("KeyPoints", {})
    pts = kp.get("Points", [])
    n = kp.get("Count", 0)
    if n < 68 or len(pts) < n * 2:
        return None
    return np.array([float(v) for v in pts[: n * 2]]).reshape(n, 2)


def compute_face_bbox(pts, img_w, img_h, pad_ratio=0.2):
    """70-point에서 얼굴 bbox 계산."""
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    w = x_max - x_min
    h = y_max - y_min
    px = w * pad_ratio
    py = h * pad_ratio

    x1 = max(0, x_min - px)
    y1 = max(0, y_min - py)
    x2 = min(img_w, x_max + px)
    y2 = min(img_h, y_max + py)

    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h

    return cx, cy, bw, bh


def extract_5_keypoints(pts, img_w, img_h):
    """70-point → 5-keypoint (normalized 0~1)."""
    le = pts[DLIB_LEFT_EYE_CENTER].mean(axis=0)
    re = pts[DLIB_RIGHT_EYE_CENTER].mean(axis=0)
    kps = np.array([le, re, pts[DLIB_NOSE_TIP], pts[DLIB_LEFT_MOUTH], pts[DLIB_RIGHT_MOUTH]])
    kps[:, 0] /= img_w
    kps[:, 1] /= img_h
    return kps


def extract_11_keypoints(pts, img_w, img_h):
    """70-point → 11-keypoint: 눈당 4점 + 코 + 입 (normalized 0~1). 레거시."""
    kps = np.array(
        [
            pts[36],
            (pts[37] + pts[38]) / 2.0,
            pts[39],
            (pts[40] + pts[41]) / 2.0,
            pts[42],
            (pts[43] + pts[44]) / 2.0,
            pts[45],
            (pts[46] + pts[47]) / 2.0,
            pts[DLIB_NOSE_TIP],
            pts[DLIB_LEFT_MOUTH],
            pts[DLIB_RIGHT_MOUTH],
        ],
        dtype=np.float64,
    )
    kps[:, 0] /= img_w
    kps[:, 1] /= img_h
    return kps


def extract_15_keypoints(pts, img_w, img_h):
    """70-point → 15-keypoint: 눈당 6점(dlib 원본) + 코 + 입 (normalized 0~1)."""
    kps = np.array(
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
            pts[DLIB_NOSE_TIP],
            pts[DLIB_LEFT_MOUTH],
            pts[DLIB_RIGHT_MOUTH],
        ],
        dtype=np.float64,
    )
    kps[:, 0] /= img_w
    kps[:, 1] /= img_h
    return kps


def convert_one(json_path, img_w, img_h, n_kpt: int, pad_ratio=0.2):
    """JSON 하나를 YOLO Pose 라벨 문자열로 변환."""
    with open(json_path, "r", encoding="utf-8") as f:
        label = json.load(f)

    pts = parse_keypoints(label)
    if pts is None:
        return None

    cx, cy, bw, bh = compute_face_bbox(pts, img_w, img_h, pad_ratio)
    if n_kpt == 5:
        kps = extract_5_keypoints(pts, img_w, img_h)
    elif n_kpt == 11:
        kps = extract_11_keypoints(pts, img_w, img_h)
    else:
        kps = extract_15_keypoints(pts, img_w, img_h)

    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for i in range(n_kpt):
        parts.append(f"{kps[i, 0]:.6f} {kps[i, 1]:.6f} 2")

    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--lbl_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--kpt", type=int, choices=(5, 11, 15), default=15,
                        help="키포인트 수 (기본 15: 눈 6+6 + 코 + 입)")
    parser.add_argument("--pad_ratio", type=float, default=0.2)
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="이미지를 복사 대신 심볼릭 링크로 생성",
    )
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    lbl_dir = Path(args.lbl_dir)
    out_img = Path(args.out_dir) / "images"
    out_lbl = Path(args.out_dir) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    folders = sorted([d for d in lbl_dir.iterdir() if d.is_dir()])
    print(f"폴더 수: {len(folders)} | kpt={args.kpt}")

    total = 0
    skipped = 0

    for folder in folders:
        json_files = sorted(folder.glob("*.json"))
        for jf in json_files:
            stem = jf.stem
            img_name = stem + ".jpg"
            img_src = img_dir / folder.name / img_name

            if not img_src.exists():
                skipped += 1
                continue

            try:
                with open(jf, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                img_w = int(meta["FileInfo"]["Width"])
                img_h = int(meta["FileInfo"]["Height"])
            except (json.JSONDecodeError, KeyError, OSError):
                skipped += 1
                continue

            line = convert_one(jf, img_w, img_h, args.kpt, args.pad_ratio)
            if line is None:
                skipped += 1
                continue

            out_txt = out_lbl / (stem + ".txt")
            with open(out_txt, "w") as f:
                f.write(line + "\n")

            out_img_path = out_img / img_name
            if not out_img_path.exists():
                if args.symlink:
                    os.symlink(img_src.resolve(), out_img_path)
                else:
                    shutil.copy2(img_src, out_img_path)

            total += 1

        if total % 5000 == 0 and total > 0:
            print(f"  ... {total} done")

    print(f"\n완료: {total} labels, {skipped} skipped")
    print(f"출력: {args.out_dir}")

    sample = sorted(out_lbl.glob("*.txt"))[:3]
    print("\n=== 샘플 라벨 ===")
    for s in sample:
        print(f"  {s.name}: {open(s).read().strip()}")


if __name__ == "__main__":
    main()
