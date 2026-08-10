"""Import a Roboflow YOLO export into the local ``yolo/`` training layout.

Takes an extracted Roboflow export (``train/valid/test`` folders, each with
``images/`` and ``labels/``) and copies it into ``bill_hand_dataset/yolo``
with splits mapped to ``train`` / ``val``.

Handles two things a plain Roboflow export does not:

* **Polygon labels** — if the Roboflow project is *instance segmentation*, the
  label lines contain a class id followed by N point pairs (``cls x1 y1 x2 y2
  ...``). Those are converted to YOLO detection boxes (``cls cx cy w h``) so
  the data works with ``yolo detect`` training.
* **Missing validation split** — if the export only contains ``train`` (e.g. a
  version generated with a 100% train split), ``--val-frac`` carves a
  deterministic random subset out of the train images for validation.

Usage::

    python import_roboflow.py <path-to-extracted-roboflow-export> [--val-frac 0.2]
"""

import os
import random
import shutil
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
YOLO = os.path.join(BASE, "yolo")

SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "val",
}

SEED = 42


def to_box_lines(text: str) -> tuple[list[str], int]:
    """Convert polygon label lines to bbox lines, returning (lines, n_converted).

    A YOLO detection line has exactly 5 tokens: ``cls cx cy w h``.
    A segmentation line has more: ``cls x1 y1 x2 y2 ...``. For those, the
    bounding box is the min/max over all polygon points.
    """
    out: list[str] = []
    converted = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 5:
            out.append(line)
            continue
        try:
            coords = [float(p) for p in parts[1:]]
        except ValueError:
            out.append(line)  # malformed - leave as-is for the user to inspect
            continue
        if len(coords) < 4 or len(coords) % 2 != 0:
            out.append(line)
            continue
        xs = coords[0::2]
        ys = coords[1::2]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        out.append(
            f"{parts[0]} {(xmin + xmax) / 2:.6f} {(ymin + ymax) / 2:.6f} "
            f"{xmax - xmin:.6f} {ymax - ymin:.6f}"
        )
        converted += 1
    return out, converted


def merge(export_root: str, split: str, val_fraction: float) -> tuple[int, list[str], int]:
    """Copy one export split into ``yolo/``, converting polygons to boxes."""
    src_img = os.path.join(export_root, split, "images")
    src_lab = os.path.join(export_root, split, "labels")
    dst_img = os.path.join(YOLO, "images", SPLIT_MAP[split])
    dst_lab = os.path.join(YOLO, "labels", SPLIT_MAP[split])
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lab, exist_ok=True)

    imgs = sorted(
        f for f in os.listdir(src_img)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    stems: list[str] = []
    converted_total = 0
    for name in imgs:
        stem = os.path.splitext(name)[0]
        stems.append(stem)
        shutil.copy2(os.path.join(src_img, name), os.path.join(dst_img, name))
        lab_src = os.path.join(src_lab, stem + ".txt")
        if os.path.exists(lab_src):
            with open(lab_src) as f:
                lines, converted = to_box_lines(f.read())
            converted_total += converted
            with open(os.path.join(dst_lab, stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        else:
            print(f"warning: no label for {name}")

    # Carve a validation set out of train if the export had no val/test split.
    if split == "train" and val_fraction > 0 and "val" not in os.listdir(export_root):
        random.seed(SEED)
        shuffled = stems[:]
        random.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        for stem in shuffled[:n_val]:
            # move the exact files that were copied above
            for ext in (".jpg", ".jpeg", ".png"):
                src_file = os.path.join(YOLO, "images", "train", stem + ext)
                if os.path.exists(src_file):
                    shutil.move(src_file, os.path.join(YOLO, "images", "val", stem + ext))
                    break
            lab_file = os.path.join(YOLO, "labels", "train", stem + ".txt")
            if os.path.exists(lab_file):
                shutil.move(lab_file, os.path.join(YOLO, "labels", "val", stem + ".txt"))
        print(f"carved val set: {n_val} of {len(stems)} train images -> yolo/val")
        return len(imgs), stems, converted_total

    return len(imgs), stems, converted_total


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python import_roboflow.py <path-to-extracted-roboflow-export> [--val-frac 0.2]")
        print("The export must contain train/valid(/test) folders, each with images/ and labels/.")
        return 1

    export_root = sys.argv[1]
    val_fraction = 0.0
    if "--val-frac" in sys.argv:
        try:
            val_fraction = float(sys.argv[sys.argv.index("--val-frac") + 1])
        except (IndexError, ValueError):
            print("error: --val-frac needs a number, e.g. 0.2")
            return 1

    if not os.path.isdir(export_root):
        print(f"error: not a directory: {export_root}")
        return 1

    total_converted = 0
    for split in ("train", "valid", "val", "test"):
        if os.path.isdir(os.path.join(export_root, split, "images")):
            n, _, conv = merge(export_root, split, val_fraction)
            total_converted += conv
            print(f"{split}: copied {n} images -> yolo/{SPLIT_MAP[split]}")

    train_labs = len(os.listdir(os.path.join(YOLO, "labels", "train")))
    val_labs = len(os.listdir(os.path.join(YOLO, "labels", "val")))
    if total_converted:
        print(f"polygon->box conversions: {total_converted}")
    print(f"\nDone. yolo/labels/train = {train_labs}, yolo/labels/val = {val_labs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
