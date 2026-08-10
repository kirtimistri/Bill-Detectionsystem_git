"""Auto-annotate the raw dataset with a pre-trained money/banknote detector.

Runs a YOLO model over every image in ``bill_hand_dataset/raw/`` and:

* writes **YOLO-format labels** (``0 <cx> <cy> <w> <h>``, class ``0`` = ``hand_bill``)
  into ``bill_hand_dataset/labels/<stem>.txt``,
* saves **annotated previews** into ``bill_hand_dataset/previews/`` so you can
  quickly eyeball the boxes,
* generates an **HTML gallery** (``bill_hand_dataset/previews/index.html``)
  listing every image with its box count for review.

The boxes are a starting point only - open the labels in Roboflow and fix them up.

Usage::

    python auto_annotate.py                          # defaults below
    python auto_annotate.py --conf 0.30              # stricter threshold
    python auto_annotate.py --classes 2 3            # keep only model classes 2,3
    python auto_annotate.py --imgsz 640 --device cpu
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"
LABELS = BASE / "labels"
PREVIEWS = BASE / "previews"
DEFAULT_MODEL = BASE / "pretrained" / "money_detector.pt"


def parse_opt() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL,
        help="path to the pre-trained YOLO detector (.pt)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="confidence threshold for detections",
    )
    parser.add_argument(
        "--classes", type=int, nargs="+", default=None,
        help="optional: only keep these model class ids (see model names at load time)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--device", default="", help='cuda device, "0", "cpu", or "" for auto')
    return parser.parse_args()


def main(opt: argparse.Namespace) -> int:
    """Run the auto-annotation pipeline."""
    if not opt.model.exists():
        print(f"error: model not found: {opt.model}")
        return 1
    if not RAW.is_dir():
        print(f"error: raw images folder not found: {RAW}")
        return 1

    model = YOLO(str(opt.model))
    print(f"model:  {opt.model} ({model.task})")
    print(f"names:  {model.names}")
    print(f"classes filter: {opt.classes if opt.classes is not None else 'all'}")
    print(f"conf:   {opt.conf}   imgsz: {opt.imgsz}\n")

    LABELS.mkdir(exist_ok=True)
    # Previews are regenerated on every run - clear stale ones.
    if PREVIEWS.is_dir():
        shutil.rmtree(PREVIEWS)
    PREVIEWS.mkdir()

    images = sorted(RAW.glob("*.jpg")) + sorted(RAW.glob("*.jpeg")) + sorted(RAW.glob("*.png"))
    if not images:
        print("error: no jpg/jpeg/png images found in raw/")
        return 1

    keep_classes = set(opt.classes) if opt.classes is not None else None

    summary = []          # (stem, n_boxes, mean_conf)
    per_class = {}        # model class id -> count
    total_boxes = 0
    with_boxes = 0
    total_empty = 0

    for img_path in images:
        stem = img_path.stem
        result = model.predict(
            str(img_path), imgsz=opt.imgsz, conf=opt.conf, device=opt.device, verbose=False
        )[0]

        boxes = result.boxes
        n = 0
        confs = []
        label_lines: list[str] = []
        frame = result.orig_img.copy()

        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            confs_all = boxes.conf.cpu().numpy()
            h, w = frame.shape[:2]

            for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, confs_all):
                if keep_classes is not None and int(c) not in keep_classes:
                    continue
                # YOLO normalized format: class cx cy w h
                cx = (x1 + x2) / 2.0 / w
                cy = (y1 + y2) / 2.0 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                cx, cy, bw, bh = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy)), \
                    max(0.0, min(1.0, bw)), max(0.0, min(1.0, bh))
                if bw <= 0 or bh <= 0:
                    continue
                label_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                per_class[int(c)] = per_class.get(int(c), 0) + 1
                n += 1
                confs.append(float(cf))

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
                cv2.putText(
                    frame, f"{model.names.get(int(c), c)} {cf:.2f}",
                    (int(x1), max(12, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA,
                )

        # Overwrite (not append) the label file so re-runs never duplicate boxes.
        if label_lines:
            (LABELS / f"{stem}.txt").write_text("".join(label_lines), encoding="utf-8")
        else:
            (LABELS / f"{stem}.txt").unlink(missing_ok=True)

        mean_conf = float(np.mean(confs)) if confs else 0.0
        summary.append((stem, n, mean_conf))
        total_boxes += n
        if n > 0:
            with_boxes += 1
        else:
            total_empty += 1

        # Always save a preview (plain image when there are no boxes).
        cv2.imwrite(str(PREVIEWS / f"{stem}.jpg"), frame)

        print(f"  {stem}.jpg: {n} box(es)" + (f"  (mean conf {mean_conf:.2f})" if n else ""))

    # ---- HTML gallery -------------------------------------------------------
    cards = []
    for stem, n, mean_conf in summary:
        src = f"{stem}.jpg"
        cards.append(
            f'<div class="card {"empty" if n == 0 else ""}">'
            f'<img src="{src}" loading="lazy" alt="{stem}">'
            f'<div class="meta"><b>{stem}.jpg</b><span>{n} box'
            f"{'es' if n != 1 else ''}"
            f'{f" · conf {mean_conf:.2f}" if n else " · NO BOXES"}</span></div></div>'
        )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Auto-annotation previews</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; background: #111; color: #eee; }}
  h1 {{ font-size: 18px; }}
  .stats {{ color: #9db; margin-bottom: 16px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }}
  .card {{ background: #1c1c1c; border-radius: 8px; overflow: hidden; border: 1px solid #333; }}
  .card img {{ width: 100%; display: block; }}
  .card.empty img {{ opacity: .45; }}
  .card.empty {{ border-color: #6a3b3b; }}
  .meta {{ padding: 8px 10px; font-size: 12px; display: flex; justify-content: space-between; gap: 8px; }}
  .meta span {{ color: #9dc; }}
  .card.empty .meta span {{ color: #e88; }}
</style></head><body>
<h1>Auto-annotation previews</h1>
<div class="stats">{len(summary)} images · {with_boxes} with boxes · {total_empty} empty ·
{total_boxes} total boxes · model classes: {model.names}</div>
<div class="grid">{''.join(cards)}</div>
</body></html>"""
    (PREVIEWS / "index.html").write_text(html, encoding="utf-8")

    # ---- summary -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"images processed : {len(images)}")
    print(f"with >=1 box     : {with_boxes}")
    print(f"no boxes         : {total_empty}")
    print(f"total boxes      : {total_boxes}")
    print(f"model class use  : {per_class or 'none'}")
    print(f"labels           : {LABELS}")
    print(f"previews         : {PREVIEWS / 'index.html'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_opt()))
