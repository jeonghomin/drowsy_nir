import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm

BASE = Path("datasets/ai-hub")  # 사용자 환경에 맞게 수정
LABEL_ROOT = BASE / "realroadenv_label"
VEHICLE_TYPES = ["bus", "sedan", "taxi", "truck"]

annotation_counts = Counter()
eye_open_counts = Counter()
mouth_open_counts = Counter()
phone_visible = 0
cigar_visible = 0
total = 0

for vtype in VEHICLE_TYPES:
    label_dir = LABEL_ROOT / vtype
    if not label_dir.exists():
        continue
    for subject_dir in tqdm(sorted(label_dir.iterdir()), desc=vtype):
        if not subject_dir.is_dir():
            continue
        for json_file in subject_dir.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            total += 1
            annotation_counts[data["Annotation"]] += 1

            bb = data["ObjectInfo"]["BoundingBox"]
            leye = bb.get("Leye", {})
            reye = bb.get("Reye", {})
            mouth = bb.get("Mouth", {})

            leye_open = leye.get("Opened", None)
            reye_open = reye.get("Opened", None)
            eye_open_counts[(leye_open, reye_open)] += 1

            mouth_open = mouth.get("Opened", None)
            mouth_open_counts[mouth_open] += 1

            if bb.get("Phone", {}).get("isVisible", False):
                phone_visible += 1
            if bb.get("Cigar", {}).get("isVisible", False):
                cigar_visible += 1

print(f"\nTotal: {total}")
print(f"\nAnnotation distribution: {dict(sorted(annotation_counts.items()))}")
print(f"\nEye open (Leye, Reye): {dict(sorted(eye_open_counts.items(), key=lambda x: -x[1]))}")
print(f"\nMouth open: {dict(sorted(mouth_open_counts.items(), key=lambda x: -x[1]))}")
print(f"\nPhone visible: {phone_visible}")
print(f"Cigar visible: {cigar_visible}")
