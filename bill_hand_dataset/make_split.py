import os
import random
import shutil
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
LABELS = os.path.join(BASE, "auto_labels")
YOLO = os.path.join(BASE, "yolo")
SEED = 42
VAL_FRAC = 0.2


def validate():
    bad = []
    n_boxes = 0
    labeled = []
    stems = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(RAW)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    for stem in stems:
        lab = os.path.join(LABELS, stem + ".txt")
        if not os.path.exists(lab):
            bad.append(f"missing label: {stem}")
            continue
        with open(lab) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            continue
        labeled.append(stem)
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                bad.append(f"bad line in {stem}.txt: {ln!r}")
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                bad.append(f"non-numeric in {stem}.txt: {ln!r}")
                continue
            if not (0.0 <= vals[0] <= 1.0 and 0.0 <= vals[1] <= 1.0 and 0.0 < vals[2] <= 1.0 and 0.0 < vals[3] <= 1.0):
                bad.append(f"out-of-range in {stem}.txt: {ln!r}")
            n_boxes += 1
    print(f"validated: {len(labeled)} labeled images, {n_boxes} boxes")
    for b in bad:
        print("  " + b)
    return labeled


def split(labeled):
    random.seed(SEED)
    order = labeled[:]
    random.shuffle(order)
    n_val = max(1, int(round(len(order) * VAL_FRAC)))
    val_set = set(order[:n_val])
    return val_set


def copy_split(labeled, val_set):
    for split in ("train", "val"):
        os.makedirs(os.path.join(YOLO, "images", split), exist_ok=True)
        os.makedirs(os.path.join(YOLO, "labels", split), exist_ok=True)
    counts = {"train": 0, "val": 0}
    for stem in labeled:
        split = "val" if stem in val_set else "train"
        src_img = next(
            os.path.join(RAW, stem + ext)
            for ext in (".jpg", ".jpeg", ".png")
            if os.path.exists(os.path.join(RAW, stem + ext))
        )
        shutil.copy2(src_img, os.path.join(YOLO, "images", split, os.path.basename(src_img)))
        shutil.copy2(os.path.join(LABELS, stem + ".txt"),
                     os.path.join(YOLO, "labels", split, stem + ".txt"))
        counts[split] += 1
    print(f"train: {counts['train']}, val: {counts['val']}")
    return counts


def main():
    labeled = validate()
    val_set = split(labeled)
    counts = copy_split(labeled, val_set)
    return 0


if __name__ == "__main__":
    sys.exit(main())
