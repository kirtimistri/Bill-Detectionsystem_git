#!/usr/bin/env python3
"""Head-to-head model comparison on the labeled test set.

Benchmarks any number of .pt models on the SAME test images (with_bill /
without_bill labels + the phone-photo set v2 missed) so you can pick the
model that works best before modifying your project.

Usage:
    ./.venv/bin/python examples/YOLO-Hand-Bill-Detection/compare_models.py \
        models/hand_bill_detector_v4.pt \
        "/home/kirti/Documents/Bill detection Ml/models/yolo_hand.pt" \
        [--imgsz 416 640] [--conf 0.25]

What it reports per model:
    - with-bill caught / total         (recall on real bills)
    - clean-image false alarms / total (precision)
    - phone-photo detections (001.jpg, Media.jpeg, test_bill3.png, test_bill5.png)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from main import bill_ok  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parent.parent
TEST_DIR = REPO_ROOT / "test_images"
PHONE_PHOTOS = ["001.jpg", "Media.jpeg", "test_bill3.png", "test_bill5.png"]


def _class0_boxes(model, img, imgsz: int, conf: float):
    """Run a model; return (box, conf) pairs for class-0 detections only.

    Class 0 is the bill class in every model here (v4 also learns a no_bill
    class that must never be reported as a detection).
    """
    r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
    if r.boxes is None or not len(r.boxes):
        return []
    return [(b, c) for b, c, cl in zip(r.boxes.xyxy.cpu().numpy(),
                                       r.boxes.conf.cpu().numpy(),
                                       r.boxes.cls.cpu().numpy()) if int(cl) == 0]


def _read(path: Path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def benchmark(model, imgsz: int, conf: float):
    imgs = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    with_bill = [p for p in imgs if p.name.startswith("with_bill")]
    without = [p for p in imgs if p.name.startswith("without_bill")]

    tp = 0
    for p in with_bill:
        img = _read(p)
        if any(bill_ok(img, b, True, True, True) for b, _ in _class0_boxes(model, img, imgsz, conf)):
            tp += 1
    fp = 0
    for p in without:
        img = _read(p)
        if any(bill_ok(img, b, True, True, True) for b, _ in _class0_boxes(model, img, imgsz, conf)):
            fp += 1

    phones = []
    for name in PHONE_PHOTOS:
        p = TEST_DIR / name
        if not p.exists():
            phones.append(f"{name[:9]}:n/a")
            continue
        img = _read(p)
        kept = [bb for bb in _class0_boxes(model, img, imgsz, conf) if bill_ok(img, bb[0], True, True, True)]
        best = max((c for _, c in kept), default=0.0)
        phones.append(f"{name[:9]}:{len(kept)}/{best:.2f}")
    return tp, len(with_bill), fp, len(without), " ".join(phones)


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-to-head YOLO model comparison on the labeled test set")
    parser.add_argument("models", nargs="+", help="paths to the .pt models to compare")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[416, 640], help="inference sizes (default 416 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold (default 0.25)")
    args = parser.parse_args()

    for imgsz in args.imgsz:
        print(f"\n=== imgsz {imgsz} (conf {args.conf}) ===")
        print(f"{'model':<58} {'with-bill':<12} {'false-alarms':<14} phones (dets/best-conf)")
        for mpath in args.models:
            path = Path(mpath)
            try:
                model = YOLO(str(path))
                tp, nwb, fp, nnb, phones = benchmark(model, imgsz, args.conf)
            except Exception as exc:
                print(f"{str(path)[:56]:<58} LOAD FAIL: {exc}")
                continue
            print(f"{str(path)[:56]:<58} {tp}/{nwb:<9} {fp}/{nnb:<10} {phones}")


if __name__ == "__main__":
    main()
